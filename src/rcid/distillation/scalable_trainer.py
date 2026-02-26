"""Scalable distillation trainer: full-sequence KL + optional RCID regulariser.

Two data streams:
  1. **Main** — InstructionDataset  → sequence-level KL distillation
  2. **RCID** — GeneratedContrastiveDataset → contrastive-difference alignment

The RCID stream is optional; when disabled, the trainer degenerates to
standard sequence-level KD.

Usage::

    trainer = ScalableDistillationTrainer(
        teacher=teacher, student=student,
        teacher_adapter=t_adp, student_adapter=s_adp,
        tokenizer=tokenizer,
        main_dataset=instruction_ds,
        contrastive_dataset=contrastive_ds,   # or None
        config=config,
        checkpoints=cps, layer_mapping=lmap, W_matrices=Ws,
    )
    history = trainer.train()
"""

from __future__ import annotations

import logging
import math
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rcid.circuit.contrastive import ContrastiveDataset
from rcid.circuit.patching import extract_contrastive_differences
from rcid.distillation.baselines import StandardKDLoss
from rcid.distillation.padd_loss import PADDLoss
from rcid.distillation.rcid_loss import RCIDLoss
from rcid.models.adapter import ModelAdapter

logger = logging.getLogger(__name__)


class ScalableDistillationTrainer:
    """Large-scale distillation: sequence-level KL + optional RCID regulariser."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        teacher_adapter: ModelAdapter,
        student_adapter: ModelAdapter,
        tokenizer: Any,
        main_dataset: Any,  # InstructionDataset (torch Dataset)
        contrastive_dataset: ContrastiveDataset | None = None,
        config: dict[str, Any] | None = None,
        checkpoints: list[tuple[int, int]] | None = None,
        layer_mapping: dict[int, int] | None = None,
        W_matrices: dict[int, torch.Tensor] | None = None,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.t_adapter = teacher_adapter
        self.s_adapter = student_adapter
        self.tokenizer = tokenizer
        self.main_dataset = main_dataset
        self.contrastive_dataset = contrastive_dataset

        cfg: dict[str, Any] = config or {}
        self.config = cfg
        self.checkpoints = checkpoints or []
        self.layer_mapping = layer_mapping or {}
        self.W_matrices = W_matrices or {}

        self.device = next(student.parameters()).device
        self.teacher.eval()

        # Upcast student to FP32 if needed (same logic as UnifiedTrainer)
        param_dtype = next(student.parameters()).dtype
        if param_dtype != torch.float32:
            logger.info("Upcasting student %s → float32", param_dtype)
            student.float()

        # ── Config values ────────────────────────────────────────────
        self.epochs: int = cfg.get("epochs", 3)
        self.batch_size: int = cfg.get("batch_size", 8)
        self.grad_accum: int = cfg.get("gradient_accumulation", 1)
        self.max_grad_norm: float = cfg.get("max_grad_norm", 1.0)
        self.lambda_rcid: float = cfg.get("lambda_rcid", 0.1)
        self.rcid_every: int = cfg.get("rcid_every_n_steps", 5)
        self.save_every: int = cfg.get("save_every_n_epochs", 1)
        self.log_every: int = cfg.get("log_every", 50)

        # ── Method routing ────────────────────────────────────────────
        method = cfg.get("method", "standard_kd")
        self.use_padd = method == "standard_kd_padd"

        # ── Loss ─────────────────────────────────────────────────────
        if self.use_padd:
            self.padd_loss_fn = PADDLoss(
                temperature=cfg.get("temperature", 2.0),
                tau=cfg.get("padd_tau", 1.0),
                alpha_min=cfg.get("padd_alpha_min", 0.1),
                alpha_max=cfg.get("padd_alpha_max", 0.9),
            )
            self.kd_loss_fn = StandardKDLoss(
                temperature=cfg.get("temperature", 2.0),
            )  # kept for evaluate()
        else:
            self.padd_loss_fn: PADDLoss | None = None
            self.kd_loss_fn = StandardKDLoss(
                temperature=cfg.get("temperature", 2.0),
            )

        # ── RCID ─────────────────────────────────────────────────────
        self.use_rcid = (
            contrastive_dataset is not None
            and len(self.checkpoints) > 0
            and len(self.W_matrices) > 0
        )
        self.rcid_loss_fn: RCIDLoss | None = None
        self.teacher_imprints: dict[tuple[int, int], torch.Tensor] = {}

        if self.use_rcid:
            self.rcid_loss_fn = RCIDLoss(
                self.checkpoints, self.layer_mapping, self.W_matrices,
            )
            self._precompute_teacher_imprints()

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
    # Pre-computation
    # ------------------------------------------------------------------

    def _precompute_teacher_imprints(self) -> None:
        """Pre-compute teacher contrastive diffs at checkpoints.

        Uses batched extraction (batch_size=4, pool_seq=False) to avoid OOM.
        Only extracts layers that appear in self.checkpoints.
        Results are stored on CPU; moved to device during _compute_rcid_loss.
        """
        assert self.contrastive_dataset is not None
        t_layers = sorted({cp[0] for cp in self.checkpoints})
        logger.info(
            "Pre-computing teacher imprints: %d checkpoints, %d layers, %d samples",
            len(self.checkpoints), len(t_layers),
            self.contrastive_dataset.clean_ids.shape[0],
        )
        diffs = extract_contrastive_differences(
            self.teacher, self.t_adapter,
            self.contrastive_dataset.clean_ids.to(self.device),
            self.contrastive_dataset.corrupt_ids.to(self.device),
            layers=t_layers,
            batch_size=4,
            pool_seq=False,  # need seq dim for position indexing
        )  # {layer: (N, seq, d_T)} on CPU
        for t_layer, t_pos in self.checkpoints:
            # diffs may be on CPU; index and keep on CPU (moved per-batch later)
            self.teacher_imprints[(t_layer, t_pos)] = (
                diffs[t_layer][:, t_pos, :].detach()  # (N_rcid, d_T)
            )
        # Free the full diffs (only imprints are needed going forward)
        del diffs
        logger.info(
            "Teacher imprints ready: %d checkpoints", len(self.checkpoints),
        )

    # ------------------------------------------------------------------
    # Student residual collection (same pattern as UnifiedTrainer)
    # ------------------------------------------------------------------

    def _collect_student_residuals(
        self, input_ids: torch.Tensor, layers: list[int],
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Forward student, hook residuals at *layers*, return (cache, logits)."""
        cache: dict[int, torch.Tensor] = {}
        handles = []
        for layer_idx in layers:
            hook_point = self.s_adapter.get_residual_hook_point(
                self.student, layer_idx,
            )
            def _make_hook(idx: int):  # noqa: E301
                def _hook(
                    mod: nn.Module, inp: tuple, out: torch.Tensor | tuple,
                ) -> None:
                    cache[idx] = self.s_adapter.parse_layer_output(out)
                return _hook
            handles.append(hook_point.register_forward_hook(_make_hook(layer_idx)))
        try:
            logits = self.student(input_ids).logits
        finally:
            for h in handles:
                h.remove()
        return cache, logits

    # ------------------------------------------------------------------
    # RCID loss for one contrastive batch
    # ------------------------------------------------------------------

    def _compute_rcid_loss(
        self, batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute RCID loss on a contrastive batch."""
        assert self.rcid_loss_fn is not None
        clean = batch["clean_ids"].to(self.device)       # (B, seq)
        corrupt = batch["corrupt_ids"].to(self.device)    # (B, seq)
        indices = batch["index"]                          # (B,) keep on CPU

        s_layers = list(set(self.layer_mapping.values()))
        s_clean_cache, _ = self._collect_student_residuals(clean, s_layers)
        s_corrupt_cache, _ = self._collect_student_residuals(corrupt, s_layers)

        # Imprints live on CPU; index on CPU then move to device
        batch_imprints = {
            key: full[indices].to(self.device)  # (B, d_T)
            for key, full in self.teacher_imprints.items()
        }
        return self.rcid_loss_fn(batch_imprints, s_clean_cache, s_corrupt_cache)

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

        # RCID data iterator (cycles)
        rcid_iter = None
        if self.use_rcid and self.contrastive_dataset is not None:
            rcid_loader = DataLoader(
                self.contrastive_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=ContrastiveDataset.collate_fn,
            )
            rcid_iter = cycle(rcid_loader)

        history: dict[str, list[float]] = {
            "loss": [], "kl_loss": [], "rcid_loss": [],
        }
        if self.use_padd:
            for k in ("alpha_mean", "forward_kl_mean",
                       "reverse_kl_mean", "teacher_entropy_mean"):
                history[k] = []

        self.student.train()
        global_step = 0
        use_amp = self.scaler is not None

        for epoch in range(self.epochs):
            epoch_kl = 0.0
            epoch_rcid = 0.0
            epoch_total = 0.0
            epoch_padd_stats: dict[str, float] = {
                "alpha_mean": 0.0, "forward_kl_mean": 0.0,
                "reverse_kl_mean": 0.0, "teacher_entropy_mean": 0.0,
            }
            n_batches = 0

            pbar = tqdm(
                main_loader, desc=f"Epoch {epoch + 1}/{self.epochs}",
                leave=True,
            )
            for step, main_batch in enumerate(pbar):
                input_ids = main_batch["input_ids"].to(self.device)   # (B, L)
                attn_mask = main_batch["attention_mask"].to(self.device).float()  # (B, L)

                # ── 1. KL loss ───────────────────────────────────────
                padd_stats: dict[str, float] | None = None
                with torch.amp.autocast("cuda", enabled=use_amp):
                    with torch.no_grad():
                        t_logits = self.teacher(input_ids).logits  # (B, L, V)
                    s_logits = self.student(input_ids).logits       # (B, L, V)

                    if self.use_padd and self.padd_loss_fn is not None:
                        kl_loss, padd_stats = self.padd_loss_fn(
                            t_logits, s_logits, mask=attn_mask,
                        )
                    else:
                        kl_loss = self.kd_loss_fn(
                            t_logits, s_logits, mask=attn_mask,
                        )

                    # ── 2. RCID loss (every N steps) ─────────────────
                    rcid_loss = torch.tensor(0.0, device=self.device)
                    if self.use_rcid and rcid_iter is not None and step % self.rcid_every == 0:
                        rcid_batch = next(rcid_iter)
                        rcid_loss = self._compute_rcid_loss(rcid_batch)

                    total_loss = kl_loss + self.lambda_rcid * rcid_loss

                    # Gradient accumulation: scale by accum steps
                    if self.grad_accum > 1:
                        total_loss = total_loss / self.grad_accum

                # ── 3. Backward ──────────────────────────────────────
                if self.scaler:
                    self.scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

                # ── 4. Optimiser step (every grad_accum batches) ─────
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

                # ── Bookkeeping ──────────────────────────────────────
                kl_val = kl_loss.item()
                rcid_val = rcid_loss.item()
                total_val = kl_val + self.lambda_rcid * rcid_val
                epoch_kl += kl_val
                epoch_rcid += rcid_val
                epoch_total += total_val
                n_batches += 1

                if padd_stats is not None:
                    for k, v in padd_stats.items():
                        epoch_padd_stats[k] += v

                postfix: dict[str, str] = {
                    "kl": f"{kl_val:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
                if padd_stats is not None:
                    postfix["alpha"] = f"{padd_stats['alpha_mean']:.3f}"
                else:
                    postfix["rcid"] = f"{rcid_val:.4f}"
                pbar.set_postfix(**postfix)

                if self.use_wandb and global_step % self.log_every == 0:
                    try:
                        import wandb
                        log_dict: dict[str, Any] = {
                            "kl_loss": kl_val, "rcid_loss": rcid_val,
                            "total_loss": total_val, "step": global_step,
                            "lr": self.optimizer.param_groups[0]["lr"],
                        }
                        if padd_stats is not None:
                            log_dict.update({
                                f"padd/{k}": v for k, v in padd_stats.items()
                            })
                        wandb.log(log_dict)
                    except ImportError:
                        self.use_wandb = False

            # ── Epoch summary ────────────────────────────────────────
            denom = max(n_batches, 1)
            avg_kl = epoch_kl / denom
            avg_rcid = epoch_rcid / denom
            avg_total = epoch_total / denom
            history["loss"].append(avg_total)
            history["kl_loss"].append(avg_kl)
            history["rcid_loss"].append(avg_rcid)

            if self.use_padd:
                for k in epoch_padd_stats:
                    avg_v = epoch_padd_stats[k] / denom
                    history[k].append(avg_v)
                logger.info(
                    "Epoch %d/%d  kl=%.4f  alpha=%.3f  fwd_kl=%.4f  "
                    "rev_kl=%.4f  total=%.4f  lr=%.2e",
                    epoch + 1, self.epochs, avg_kl,
                    epoch_padd_stats["alpha_mean"] / denom,
                    epoch_padd_stats["forward_kl_mean"] / denom,
                    epoch_padd_stats["reverse_kl_mean"] / denom,
                    avg_total, self.optimizer.param_groups[0]["lr"],
                )
            else:
                logger.info(
                    "Epoch %d/%d  kl=%.4f  rcid=%.4f  total=%.4f  lr=%.2e",
                    epoch + 1, self.epochs, avg_kl, avg_rcid, avg_total,
                    self.optimizer.param_groups[0]["lr"],
                )

            # ── Checkpoint ───────────────────────────────────────────
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
