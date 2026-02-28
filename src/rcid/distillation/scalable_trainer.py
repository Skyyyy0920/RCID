"""Scalable distillation trainer for KL-Ratio adaptive distillation.

Supports four training methods:
  - ``standard_kd``     — forward KL baseline
  - ``reverse_kl``      — reverse KL baseline
  - ``standard_kd_akl`` — AKL (Wu et al., COLING 2025)
  - ``standard_kd_klr`` — KL-Ratio adaptive (ours)

All methods share the same training infrastructure: InstructionDataset,
AdamW + cosine LR with warmup, fp16, gradient accumulation.

Usage::

    trainer = ScalableDistillationTrainer(
        teacher=teacher, student=student,
        teacher_adapter=t_adp, student_adapter=s_adp,
        tokenizer=tokenizer,
        main_dataset=instruction_ds,
        config={"method": "standard_kd_klr", "klr_granularity": "batch", ...},
    )
    history = trainer.train(save_dir="outputs/paper/klr_batch_ema")
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rcid.distillation.baselines import StandardKDLoss
from rcid.models.adapter import ModelAdapter

logger = logging.getLogger(__name__)

_VALID_METHODS = {
    "standard_kd", "reverse_kl",
    "standard_kd_akl", "standard_kd_klr",
}
_ADAPTIVE_METHODS = {"standard_kd_akl", "standard_kd_klr"}


class ScalableDistillationTrainer:
    """Large-scale distillation with adaptive KL divergence methods."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        teacher_adapter: ModelAdapter,
        student_adapter: ModelAdapter,
        tokenizer: Any,
        main_dataset: Any,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.t_adapter = teacher_adapter
        self.s_adapter = student_adapter
        self.tokenizer = tokenizer
        self.main_dataset = main_dataset

        cfg: dict[str, Any] = config or {}
        self.config = cfg
        self.device = next(student.parameters()).device
        self.teacher.eval()

        # Upcast student to FP32 if needed
        param_dtype = next(student.parameters()).dtype
        if param_dtype != torch.float32:
            logger.info("Upcasting student %s -> float32", param_dtype)
            student.float()

        # ── Config values ────────────────────────────────────────────
        self.epochs: int = cfg.get("epochs", 3)
        self.batch_size: int = cfg.get("batch_size", 8)
        self.grad_accum: int = cfg.get("gradient_accumulation", 1)
        self.max_grad_norm: float = cfg.get("max_grad_norm", 1.0)
        self.save_every: int = cfg.get("save_every_n_epochs", 1)
        self.log_every: int = cfg.get("log_every", 50)
        self.jsonl_every: int = cfg.get("jsonl_every", 100)

        # ── Method routing ────────────────────────────────────────────
        method = cfg.get("method", "standard_kd")
        assert method in _VALID_METHODS, (
            f"Unknown method {method!r}. Valid: {_VALID_METHODS}"
        )
        self.method = method
        self.use_adaptive = method in _ADAPTIVE_METHODS

        # ── Loss functions ────────────────────────────────────────────
        temperature = cfg.get("temperature", 2.0)
        self.adaptive_loss_fn: nn.Module | None = None
        # kd_loss_fn: always available (plain FKL, used by evaluate())
        self.kd_loss_fn = StandardKDLoss(temperature=temperature)

        if method == "reverse_kl":
            from rcid.distillation.adaptive_kl_losses import KLRatioLoss
            # Reverse KL = KLRatioLoss with fixed_alpha=0.0
            self.adaptive_loss_fn = KLRatioLoss(
                temperature=temperature, fixed_alpha=0.0,
            )
            self.use_adaptive = True
        elif method == "standard_kd_akl":
            from rcid.distillation.adaptive_kl_losses import AKLLoss
            self.adaptive_loss_fn = AKLLoss(
                temperature=temperature,
                mu=cfg.get("akl_mu", 0.5),
            )
        elif method == "standard_kd_klr":
            from rcid.distillation.adaptive_kl_losses import KLRatioLoss
            self.adaptive_loss_fn = KLRatioLoss(
                temperature=temperature,
                granularity=cfg.get("klr_granularity", "token"),
                beta=cfg.get("klr_beta", 0.99),
                fixed_alpha=cfg.get("klr_fixed_alpha", None),
            )

        # ── Optimiser ────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=cfg.get("lr", 2e-5),
            weight_decay=cfg.get("weight_decay", 0.01),
        )

        # ── AMP ──────────────────────────────────────────────────────
        use_amp = cfg.get("fp16", True)
        self.scaler = torch.amp.GradScaler() if use_amp else None

        # ── WandB ────────────────────────────────────────────────────
        self.use_wandb = cfg.get("use_wandb", False)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self, save_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Run the full training loop. Returns per-epoch history."""
        main_loader = DataLoader(
            self.main_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.main_dataset.collate_fn,
        )
        total_steps = self.epochs * math.ceil(len(main_loader) / self.grad_accum)

        # Cosine scheduler with warmup
        warmup_ratio = self.config.get("warmup_ratio", 0.03)
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps - warmup_steps, 1),
        )

        # ── History + JSONL file ──────────────────────────────────────
        history: dict[str, list[float]] = {"loss": [], "kl_loss": []}
        self._adaptive_stat_keys: list[str] = []
        if self.use_adaptive:
            if self.method == "standard_kd_akl":
                self._adaptive_stat_keys = [
                    "alpha_mean", "forward_kl_mean",
                    "reverse_kl_mean", "g_head_mean", "g_tail_mean",
                ]
            else:
                # reverse_kl, standard_kd_klr — all use KLRatioLoss
                self._adaptive_stat_keys = [
                    "alpha_mean", "alpha_std",
                    "forward_kl_mean", "reverse_kl_mean",
                ]
            for k in self._adaptive_stat_keys:
                history[k] = []

        jsonl_path: Path | None = None
        if save_dir:
            jsonl_path = Path(save_dir) / "training_stats.jsonl"
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            # Clear existing file
            jsonl_path.write_text("", encoding="utf-8")

        self.student.train()
        global_step = 0
        use_amp = self.scaler is not None

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_adaptive_stats: dict[str, float] = {
                k: 0.0 for k in self._adaptive_stat_keys
            }
            n_batches = 0

            pbar = tqdm(
                main_loader, desc=f"Epoch {epoch + 1}/{self.epochs}",
                leave=True,
            )
            for step, main_batch in enumerate(pbar):
                input_ids = main_batch["input_ids"].to(self.device)      # (B, L)
                attn_mask = main_batch["attention_mask"].to(self.device).float()

                # ── Forward ───────────────────────────────────────────
                adaptive_stats: dict[str, float] | None = None
                with torch.amp.autocast("cuda", enabled=use_amp):
                    with torch.no_grad():
                        t_logits = self.teacher(input_ids).logits  # (B,L,V)
                    s_logits = self.student(input_ids).logits       # (B,L,V)

                    if self.use_adaptive and self.adaptive_loss_fn is not None:
                        total_loss, adaptive_stats = self.adaptive_loss_fn(
                            t_logits, s_logits, mask=attn_mask,
                        )
                    else:
                        total_loss = self.kd_loss_fn(
                            t_logits, s_logits, mask=attn_mask,
                        )

                    if self.grad_accum > 1:
                        total_loss = total_loss / self.grad_accum

                # ── Backward ──────────────────────────────────────────
                if self.scaler:
                    self.scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

                # ── Optimiser step ────────────────────────────────────
                if (step + 1) % self.grad_accum == 0 or (step + 1) == len(main_loader):
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.student.parameters(), self.max_grad_norm,
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(
                            self.student.parameters(), self.max_grad_norm,
                        )
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                # ── Bookkeeping ───────────────────────────────────────
                loss_val = total_loss.item() * (
                    self.grad_accum if self.grad_accum > 1 else 1
                )
                epoch_loss += loss_val
                n_batches += 1

                if adaptive_stats is not None:
                    for k in self._adaptive_stat_keys:
                        if k in adaptive_stats:
                            epoch_adaptive_stats[k] += adaptive_stats[k]

                # Progress bar
                postfix: dict[str, str] = {
                    "loss": f"{loss_val:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
                if adaptive_stats is not None:
                    postfix["alpha"] = f"{adaptive_stats.get('alpha_mean', 0):.3f}"
                pbar.set_postfix(**postfix)

                # ── JSONL per-step logging ─────────────────────────────
                if jsonl_path and global_step > 0 and global_step % self.jsonl_every == 0:
                    record: dict[str, Any] = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": round(loss_val, 6),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                    if adaptive_stats is not None:
                        record["alpha_mean"] = round(
                            adaptive_stats.get("alpha_mean", 0.0), 6,
                        )
                        record["fkl_mean"] = round(
                            adaptive_stats.get("forward_kl_mean", 0.0), 6,
                        )
                        record["rkl_mean"] = round(
                            adaptive_stats.get("reverse_kl_mean", 0.0), 6,
                        )
                    with open(jsonl_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\n")

                # ── WandB ──────────────────────────────────────────────
                if self.use_wandb and global_step % self.log_every == 0:
                    try:
                        import wandb
                        log_dict: dict[str, Any] = {
                            "loss": loss_val,
                            "step": global_step,
                            "lr": self.optimizer.param_groups[0]["lr"],
                        }
                        if adaptive_stats is not None:
                            log_dict.update({
                                f"adaptive/{k}": v
                                for k, v in adaptive_stats.items()
                            })
                        wandb.log(log_dict)
                    except ImportError:
                        self.use_wandb = False

            # ── Epoch summary ─────────────────────────────────────────
            denom = max(n_batches, 1)
            avg_loss = epoch_loss / denom
            history["loss"].append(avg_loss)
            history["kl_loss"].append(avg_loss)  # same for non-RCID

            if self.use_adaptive:
                for k in self._adaptive_stat_keys:
                    avg_v = epoch_adaptive_stats[k] / denom
                    history[k].append(avg_v)
                avg_alpha = epoch_adaptive_stats.get("alpha_mean", 0) / denom
                avg_fkl = epoch_adaptive_stats.get("forward_kl_mean", 0) / denom
                avg_rkl = epoch_adaptive_stats.get("reverse_kl_mean", 0) / denom
                logger.info(
                    "Epoch %d/%d  loss=%.4f  alpha=%.3f  fwd_kl=%.4f  "
                    "rev_kl=%.4f  lr=%.2e",
                    epoch + 1, self.epochs, avg_loss, avg_alpha,
                    avg_fkl, avg_rkl,
                    self.optimizer.param_groups[0]["lr"],
                )
            else:
                logger.info(
                    "Epoch %d/%d  loss=%.4f  lr=%.2e",
                    epoch + 1, self.epochs, avg_loss,
                    self.optimizer.param_groups[0]["lr"],
                )

            # ── Checkpoint ────────────────────────────────────────────
            if save_dir and (epoch + 1) % self.save_every == 0:
                ckpt = Path(save_dir) / f"student_epoch{epoch + 1}.pt"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.student.state_dict(), ckpt)
                logger.info("Saved checkpoint: %s", ckpt)

        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, eval_dataset: Any | None = None,
    ) -> dict[str, float]:
        """Compute KL loss on *eval_dataset* (or main_dataset if None)."""
        ds = eval_dataset or self.main_dataset
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            collate_fn=ds.collate_fn,
        )
        self.student.eval()
        total_kl = 0.0
        n = 0
        use_amp = self.scaler is not None

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device).float()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    t_logits = self.teacher(input_ids).logits
                    s_logits = self.student(input_ids).logits
                    kl = self.kd_loss_fn(t_logits, s_logits, mask=attn_mask)
                total_kl += kl.item()
                n += 1

        self.student.train()
        avg_kl = total_kl / max(n, 1)
        logger.info("Eval KL loss: %.4f", avg_kl)
        return {"kl_loss": avg_kl}
