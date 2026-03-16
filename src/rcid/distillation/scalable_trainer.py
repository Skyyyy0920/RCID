"""Scalable distillation trainer: KL methods + optional RCID/SaGD.

Supports eight training methods:

KL methods (single data stream):
  - ``standard_kd``     — forward KL baseline
  - ``reverse_kl``      — reverse KL baseline
  - ``standard_kd_akl`` — AKL (Wu et al., COLING 2025)
  - ``standard_kd_klr`` — KL-Ratio adaptive (ours)

RCID methods (dual data stream: KL + contrastive regulariser):
  - ``standard_kd_rcid``             — KL + RCID (contrastive diff matching)
  - ``standard_kd_fitnets``          — KL + FitNets (all-layer repr. matching)
  - ``standard_kd_informed_fitnets`` — KL + InformedFitNets (checkpoint matching)

SaGD methods (single data stream + precomputed teacher saliency):
  - ``standard_kd_sagd``             — KL with saliency-guided sample reweighting
"""

from __future__ import annotations

import json
import logging
import math
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rcid.distillation.baselines import StandardKDLoss
from rcid.models.adapter import ModelAdapter

logger = logging.getLogger(__name__)

_VALID_METHODS = {
    "standard_kd", "reverse_kl",
    "standard_kd_akl", "standard_kd_klr",
    "standard_kd_rcid", "standard_kd_fitnets", "standard_kd_informed_fitnets",
    "standard_kd_sagd",
}
_ADAPTIVE_METHODS = {"standard_kd_akl", "standard_kd_klr"}
_RCID_METHODS = {"standard_kd_rcid", "standard_kd_fitnets", "standard_kd_informed_fitnets"}
_SAGD_METHODS = {"standard_kd_sagd"}


def _extract_residuals_no_grad(
    model: nn.Module, adapter: ModelAdapter,
    input_ids: torch.Tensor, layers: list[int],
) -> dict[int, torch.Tensor]:
    """Forward *model* and capture residuals at *layers* WITHOUT gradients."""
    cache: dict[int, torch.Tensor] = {}
    handles: list[Any] = []
    for layer_idx in layers:
        hp = adapter.get_residual_hook_point(model, layer_idx)

        def _make_hook(idx: int):  # noqa: E301
            def _hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
                cache[idx] = adapter.parse_layer_output(out).detach()
            return _hook

        handles.append(hp.register_forward_hook(_make_hook(layer_idx)))
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in handles:
            h.remove()
    return cache


class ScalableDistillationTrainer:
    """Large-scale distillation with adaptive KL + optional RCID regulariser."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        teacher_adapter: ModelAdapter,
        student_adapter: ModelAdapter,
        tokenizer: Any,
        main_dataset: Any,
        config: dict[str, Any] | None = None,
        # ── RCID-specific (optional) ──
        contrastive_dataset: Any | None = None,
        rcid_loss_fn: nn.Module | None = None,
        lambda_rcid: float = 0.1,
        rcid_every_n_steps: int = 5,
        layer_mapping: dict[int, int] | None = None,
        checkpoints: list[tuple[int, int]] | None = None,
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
        self.use_rcid = method in _RCID_METHODS

        # ── KL loss function ──────────────────────────────────────────
        temperature = cfg.get("temperature", 2.0)
        self.adaptive_loss_fn: nn.Module | None = None
        self.kd_loss_fn = StandardKDLoss(temperature=temperature)

        if method == "reverse_kl":
            from rcid.distillation.adaptive_kl_losses import KLRatioLoss
            self.adaptive_loss_fn = KLRatioLoss(
                temperature=temperature, fixed_alpha=0.0,
            )
            self.use_adaptive = True
        elif method == "standard_kd_akl":
            from rcid.distillation.adaptive_kl_losses import AKLLoss
            self.adaptive_loss_fn = AKLLoss(
                temperature=temperature, mu=cfg.get("akl_mu", 0.5),
            )
        elif method == "standard_kd_klr":
            from rcid.distillation.adaptive_kl_losses import KLRatioLoss
            self.adaptive_loss_fn = KLRatioLoss(
                temperature=temperature,
                granularity=cfg.get("klr_granularity", "token"),
                beta=cfg.get("klr_beta", 0.99),
                fixed_alpha=cfg.get("klr_fixed_alpha", None),
            )

        # ── RCID regulariser setup ───────────────────────────────────
        self.rcid_loss_fn = rcid_loss_fn
        self.lambda_rcid = lambda_rcid
        self.rcid_every = rcid_every_n_steps
        self.layer_mapping = layer_mapping or {}
        self.checkpoints = checkpoints or []
        self.contrastive_dataset = contrastive_dataset

        if self.use_rcid:
            assert contrastive_dataset is not None, (
                f"Method {method!r} requires contrastive_dataset"
            )
            assert rcid_loss_fn is not None, (
                f"Method {method!r} requires rcid_loss_fn"
            )
            # Move RCID loss (contains Procrustes W buffers) to device
            self.rcid_loss_fn = rcid_loss_fn.to(self.device)

        # ── SaGD setup ────────────────────────────────────────────────
        self.use_sagd = (method == "standard_kd_sagd")
        self.saliency_computer: Any = None
        self.teacher_saliency_cache: list[torch.Tensor] = []
        self.sagd_every: int = 1
        self.sagd_tau_w: float = 1.0

        if self.use_sagd:
            from rcid.distillation.saliency import (
                SaliencyComputer,
                SaliencyAlignmentLoss,
            )
            self.saliency_computer = SaliencyComputer(
                temperature=cfg.get("saliency_temperature", 2.0),
            )
            self.saliency_loss_fn = SaliencyAlignmentLoss()
            self.lambda_sal: float = cfg.get("lambda_sal", 0.5)
            sagd_path = cfg.get("teacher_saliency_path")
            assert sagd_path is not None, (
                "SaGD requires config key 'teacher_saliency_path'"
            )
            cache = torch.load(sagd_path, map_location="cpu", weights_only=False)
            self.teacher_saliency_cache = cache["saliency"]
            self.sagd_every = cfg.get("sagd_every_n_steps", 5)
            self.sagd_tau_w = cfg.get("sagd_tau_w", 1.0)
            logger.info(
                "SaGD: loaded %d teacher saliencies, every=%d, tau_w=%.2f, lambda_sal=%.2f",
                len(self.teacher_saliency_cache), self.sagd_every,
                self.sagd_tau_w, self.lambda_sal,
            )

        # ── Optimiser + AMP + WandB ──────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=cfg.get("lr", 2e-5),
            weight_decay=cfg.get("weight_decay", 0.01),
        )
        use_amp = cfg.get("fp16", True)
        self.scaler = torch.amp.GradScaler() if use_amp else None
        self.use_wandb = cfg.get("use_wandb", False)

    # ------------------------------------------------------------------
    # RCID regulariser
    # ------------------------------------------------------------------

    def _compute_rcid_loss(
        self, contrastive_batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute RCID / FitNets / InformedFitNets regulariser loss."""
        from rcid.distillation.rcid_loss import extract_residuals_with_grad

        clean_ids = contrastive_batch["clean_ids"].to(self.device)  # (B, L)
        corrupt_ids = contrastive_batch["corrupt_ids"].to(self.device)

        if self.method == "standard_kd_rcid":
            t_layers = list({l for l, _ in self.checkpoints})
            s_layers = list({self.layer_mapping[l] for l, _ in self.checkpoints
                           if l in self.layer_mapping})
            t_clean = _extract_residuals_no_grad(
                self.teacher, self.t_adapter, clean_ids, t_layers)
            t_corrupt = _extract_residuals_no_grad(
                self.teacher, self.t_adapter, corrupt_ids, t_layers)
            teacher_diffs = {
                l: (t_clean[l] - t_corrupt[l]).detach()
                for l in t_layers if l in t_clean and l in t_corrupt
            }
            s_clean = extract_residuals_with_grad(
                self.student, self.s_adapter, clean_ids, s_layers)
            s_corrupt = extract_residuals_with_grad(
                self.student, self.s_adapter, corrupt_ids, s_layers)
            return self.rcid_loss_fn(teacher_diffs, s_clean, s_corrupt)

        elif self.method == "standard_kd_fitnets":
            t_layers = list(self.layer_mapping.keys())
            s_layers = list(set(self.layer_mapping.values()))
            t_res = _extract_residuals_no_grad(
                self.teacher, self.t_adapter, clean_ids, t_layers)
            s_res = extract_residuals_with_grad(
                self.student, self.s_adapter, clean_ids, s_layers)
            return self.rcid_loss_fn(t_res, s_res)

        else:  # standard_kd_informed_fitnets
            t_layers = list({l for l, _ in self.checkpoints})
            s_layers = list({self.layer_mapping[l] for l, _ in self.checkpoints
                           if l in self.layer_mapping})
            t_res = _extract_residuals_no_grad(
                self.teacher, self.t_adapter, clean_ids, t_layers)
            s_res = extract_residuals_with_grad(
                self.student, self.s_adapter, clean_ids, s_layers)
            return self.rcid_loss_fn(t_res, s_res)

    # ------------------------------------------------------------------
    # SaGD helpers
    # ------------------------------------------------------------------

    def _get_cached_teacher_saliency(
        self,
        indices: torch.Tensor,  # (B,)
        max_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Look up precomputed teacher saliency and pad/truncate to max_len.

        The cached saliency is trimmed to actual (non-padding) length during
        precomputation. This method zero-pads shorter cached vectors and
        truncates if cached length exceeds the current batch's max_len.

        Returns:
            t_sal: (B, max_len) teacher saliency on *device*.
        """
        B = len(indices)
        t_sal = torch.zeros(B, max_len, device=device)
        for j, idx_t in enumerate(indices):
            s = self.teacher_saliency_cache[idx_t.item()]
            L_s = min(len(s), max_len)
            t_sal[j, :L_s] = s[:L_s].to(device)
        return t_sal

    # ------------------------------------------------------------------
    # SaGD per-sample KL
    # ------------------------------------------------------------------

    def _compute_per_sample_kl(
        self,
        t_logits: torch.Tensor,  # (B, L, V)
        s_logits: torch.Tensor,  # (B, L, V)
        mask: torch.Tensor,      # (B, L)
    ) -> torch.Tensor:
        """Per-sample KL divergence (not reduced across batch).

        Returns:
            kl: (B,) — per-sample KL loss.
        """
        T = self.config.get("temperature", 2.0)
        t_probs = F.softmax(t_logits.float() / T, dim=-1)       # (B, L, V)
        s_log_probs = F.log_softmax(s_logits.float() / T, dim=-1)  # (B, L, V)

        per_token_kl = F.kl_div(
            s_log_probs, t_probs, reduction="none",
        ).sum(dim=-1)  # (B, L)

        valid_counts = mask.sum(dim=-1).clamp(min=1)  # (B,)
        per_sample = (per_token_kl * mask).sum(dim=-1) / valid_counts  # (B,)
        per_sample = per_sample * (T * T)

        return per_sample  # (B,)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self, save_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Run the full training loop. Returns per-epoch history."""
        main_loader = DataLoader(
            self.main_dataset, batch_size=self.batch_size,
            shuffle=True, collate_fn=self.main_dataset.collate_fn,
        )
        total_steps = self.epochs * math.ceil(len(main_loader) / self.grad_accum)

        warmup_ratio = self.config.get("warmup_ratio", 0.03)
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps - warmup_steps, 1),
        )

        # ── RCID data iterator ──────────────────────────────────────
        rcid_iter = None
        if self.use_rcid and self.contrastive_dataset is not None:
            rcid_loader = DataLoader(
                self.contrastive_dataset,
                batch_size=min(self.batch_size, len(self.contrastive_dataset)),
                shuffle=True,
                collate_fn=getattr(self.contrastive_dataset, "collate_fn", None),
            )
            rcid_iter = iter(cycle(rcid_loader))

        # ── History ─────────────────────────────────────────────────
        history: dict[str, list[float]] = {"loss": [], "kl_loss": []}
        self._adaptive_stat_keys: list[str] = []
        if self.use_adaptive:
            if self.method == "standard_kd_akl":
                self._adaptive_stat_keys = [
                    "alpha_mean", "forward_kl_mean",
                    "reverse_kl_mean", "g_head_mean", "g_tail_mean",
                ]
            else:
                self._adaptive_stat_keys = [
                    "alpha_mean", "alpha_std",
                    "forward_kl_mean", "reverse_kl_mean",
                ]
            for k in self._adaptive_stat_keys:
                history[k] = []
        if self.use_rcid:
            history["rcid_loss"] = []
        if self.use_sagd:
            history["sagd_mean_jsd"]     = []
            history["sagd_max_weight"]   = []
            history["sagd_min_weight"]   = []
            history["sagd_sal_loss"]     = []
            history["sagd_mean_cos_sim"] = []

        jsonl_path: Path | None = None
        if save_dir:
            jsonl_path = Path(save_dir) / "training_stats.jsonl"
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_path.write_text("", encoding="utf-8")

        self.student.train()
        global_step = 0
        use_amp = self.scaler is not None

        for epoch in range(self.epochs):
            ep_loss = ep_kl = ep_rcid = 0.0
            ep_sagd_jsd = ep_sagd_maxw = ep_sagd_minw = 0.0
            ep_sagd_sal = ep_sagd_cos  = 0.0
            ep_adaptive: dict[str, float] = {k: 0.0 for k in self._adaptive_stat_keys}
            n_batches = 0

            pbar = tqdm(main_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=True)
            for step, main_batch in enumerate(pbar):
                ids = main_batch["input_ids"].to(self.device)       # (B, L)
                mask = main_batch["attention_mask"].to(self.device).float()

                adaptive_stats: dict[str, float] | None = None
                with torch.amp.autocast("cuda", enabled=use_amp):
                    with torch.no_grad():
                        t_logits = self.teacher(ids).logits  # (B,L,V)
                    s_logits = self.student(ids).logits       # (B,L,V)

                    if self.use_adaptive and self.adaptive_loss_fn is not None:
                        kl_loss, adaptive_stats = self.adaptive_loss_fn(
                            t_logits, s_logits, mask=mask)
                    else:
                        kl_loss = self.kd_loss_fn(t_logits, s_logits, mask=mask)

                    total_loss = kl_loss
                    rcid_loss_val = 0.0

                    # ── RCID regulariser ─────────────────────────────
                    if (self.use_rcid and rcid_iter is not None
                            and step % self.rcid_every == 0):
                        c_batch = next(rcid_iter)
                        rcid_loss = self._compute_rcid_loss(c_batch)
                        total_loss = total_loss + self.lambda_rcid * rcid_loss
                        rcid_loss_val = rcid_loss.item()

                    # ── SaGD reweighting ──────────────────────────────
                    sagd_jsd_val      = 0.0
                    sagd_max_w        = sagd_min_w = 0.0
                    sagd_sal_loss_val = 0.0
                    sagd_cos_sim_val  = 0.0
                    if self.use_sagd and step % self.sagd_every == 0:
                        indices = main_batch["index"]  # (B,)
                        labels_mask_b = main_batch["labels_mask"].to(self.device)
                        B_cur = ids.shape[0]
                        max_len = ids.shape[1]

                        # Look up precomputed teacher saliency
                        t_sal = self._get_cached_teacher_saliency(
                            indices, max_len, self.device,
                        )  # (B, max_len)

                        # Compute student saliency on the fly
                        s_sal = self.saliency_computer.compute(
                            self.student, ids, mask.long(), labels_mask_b,
                        )

                        # Convert to distributions → JSD → weights
                        t_dist = self.saliency_computer.to_distribution(
                            t_sal, labels_mask_b)
                        s_dist = self.saliency_computer.to_distribution(
                            s_sal, labels_mask_b)
                        jsd = self.saliency_computer.divergence(
                            t_dist, s_dist, labels_mask_b)  # (B,)
                        weights = F.softmax(
                            jsd / self.sagd_tau_w, dim=0) * B_cur  # mean=1

                        # Per-sample KL with reweighting
                        # logit[j] predicts token[j+1], so shift labels_mask
                        shifted_resp = torch.zeros_like(labels_mask_b, dtype=torch.float)
                        shifted_resp[:, :-1] = labels_mask_b[:, 1:].float()
                        per_sample_kl = self._compute_per_sample_kl(
                            t_logits, s_logits, shifted_resp)  # (B,)
                        kl_loss = (weights.detach() * per_sample_kl).mean()

                        # Saliency alignment loss (first-order matching term)
                        sal_loss, sal_stats = self.saliency_loss_fn(
                            t_sal, s_sal, labels_mask_b,
                        )
                        total_loss = kl_loss + self.lambda_sal * sal_loss

                        sagd_sal_loss_val = sal_loss.item()
                        sagd_cos_sim_val  = sal_stats["mean_cos_sim"]
                        sagd_jsd_val = jsd.mean().item()
                        sagd_max_w = weights.max().item()
                        sagd_min_w = weights.min().item()

                    if self.grad_accum > 1:
                        total_loss = total_loss / self.grad_accum

                # ── Backward + step ──────────────────────────────────
                if self.scaler:
                    self.scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

                if (step + 1) % self.grad_accum == 0 or (step + 1) == len(main_loader):
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                # ── Bookkeeping ──────────────────────────────────────
                loss_val = total_loss.item() * (self.grad_accum if self.grad_accum > 1 else 1)
                kl_val = kl_loss.item()
                ep_loss += loss_val
                ep_kl += kl_val
                ep_rcid += rcid_loss_val
                ep_sagd_jsd  += sagd_jsd_val
                ep_sagd_maxw += sagd_max_w
                ep_sagd_minw += sagd_min_w
                ep_sagd_sal  += sagd_sal_loss_val
                ep_sagd_cos  += sagd_cos_sim_val
                n_batches += 1

                if adaptive_stats is not None:
                    for k in self._adaptive_stat_keys:
                        if k in adaptive_stats:
                            ep_adaptive[k] += adaptive_stats[k]

                postfix: dict[str, str] = {
                    "loss": f"{loss_val:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
                if adaptive_stats is not None:
                    postfix["alpha"] = f"{adaptive_stats.get('alpha_mean', 0):.3f}"
                if self.use_rcid and rcid_loss_val > 0:
                    postfix["rcid"] = f"{rcid_loss_val:.4f}"
                if self.use_sagd and sagd_jsd_val > 0:
                    postfix["jsd"] = f"{sagd_jsd_val:.4f}"
                pbar.set_postfix(**postfix)

                # ── JSONL logging ────────────────────────────────────
                if jsonl_path and global_step > 0 and global_step % self.jsonl_every == 0:
                    rec: dict[str, Any] = {
                        "step": global_step, "epoch": epoch + 1,
                        "loss": round(loss_val, 6),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                    if adaptive_stats is not None:
                        rec["alpha_mean"] = round(adaptive_stats.get("alpha_mean", 0.0), 6)
                        rec["fkl_mean"] = round(adaptive_stats.get("forward_kl_mean", 0.0), 6)
                        rec["rkl_mean"] = round(adaptive_stats.get("reverse_kl_mean", 0.0), 6)
                    if self.use_rcid:
                        rec["rcid_loss"] = round(rcid_loss_val, 6)
                    if self.use_sagd:
                        rec["sagd_jsd"] = round(sagd_jsd_val, 6)
                        rec["sagd_max_w"] = round(sagd_max_w, 4)
                        rec["sagd_min_w"] = round(sagd_min_w, 4)
                        rec["sagd_sal_loss"] = round(sagd_sal_loss_val, 6)
                        rec["sagd_cos_sim"] = round(sagd_cos_sim_val, 4)
                    with open(jsonl_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")

                # ── WandB ────────────────────────────────────────────
                if self.use_wandb and global_step % self.log_every == 0:
                    try:
                        import wandb
                        wd: dict[str, Any] = {"loss": loss_val, "step": global_step,
                                              "lr": self.optimizer.param_groups[0]["lr"]}
                        if adaptive_stats:
                            wd.update({f"adaptive/{k}": v for k, v in adaptive_stats.items()})
                        if self.use_rcid:
                            wd["rcid_loss"] = rcid_loss_val
                        if self.use_sagd:
                            wd["sagd/jsd"] = sagd_jsd_val
                            wd["sagd/max_weight"] = sagd_max_w
                            wd["sagd/sal_loss"] = sagd_sal_loss_val
                            wd["sagd/cos_sim"] = sagd_cos_sim_val
                        wandb.log(wd)
                    except ImportError:
                        self.use_wandb = False

            # ── Epoch summary ────────────────────────────────────────
            denom = max(n_batches, 1)
            avg_loss = ep_loss / denom
            avg_kl = ep_kl / denom
            history["loss"].append(avg_loss)
            history["kl_loss"].append(avg_kl)

            if self.use_rcid:
                history["rcid_loss"].append(ep_rcid / denom)
            if self.use_sagd:
                history["sagd_mean_jsd"].append(ep_sagd_jsd / denom)
                history["sagd_max_weight"].append(ep_sagd_maxw / denom)
                history["sagd_min_weight"].append(ep_sagd_minw / denom)
                history["sagd_sal_loss"].append(ep_sagd_sal / denom)
                history["sagd_mean_cos_sim"].append(ep_sagd_cos / denom)
            if self.use_adaptive:
                for k in self._adaptive_stat_keys:
                    history[k].append(ep_adaptive[k] / denom)
                logger.info(
                    "Epoch %d/%d  loss=%.4f  alpha=%.3f  fwd_kl=%.4f  rev_kl=%.4f  lr=%.2e",
                    epoch+1, self.epochs, avg_loss,
                    ep_adaptive.get("alpha_mean", 0)/denom,
                    ep_adaptive.get("forward_kl_mean", 0)/denom,
                    ep_adaptive.get("reverse_kl_mean", 0)/denom,
                    self.optimizer.param_groups[0]["lr"],
                )
            elif self.use_rcid:
                logger.info(
                    "Epoch %d/%d  loss=%.4f  kl=%.4f  rcid=%.4f  lr=%.2e",
                    epoch+1, self.epochs, avg_loss, avg_kl, ep_rcid/denom,
                    self.optimizer.param_groups[0]["lr"],
                )
            elif self.use_sagd:
                logger.info(
                    "Epoch %d/%d  loss=%.4f  kl=%.4f  sal=%.4f  cos=%.3f  "
                    "jsd=%.4f  max_w=%.2f  lr=%.2e",
                    epoch+1, self.epochs, avg_loss, avg_kl,
                    ep_sagd_sal/denom, ep_sagd_cos/denom,
                    ep_sagd_jsd/denom, ep_sagd_maxw/denom,
                    self.optimizer.param_groups[0]["lr"],
                )
            else:
                logger.info(
                    "Epoch %d/%d  loss=%.4f  lr=%.2e",
                    epoch+1, self.epochs, avg_loss,
                    self.optimizer.param_groups[0]["lr"],
                )

            if save_dir and (epoch + 1) % self.save_every == 0:
                ckpt = Path(save_dir) / f"student_epoch{epoch+1}.pt"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.student.state_dict(), ckpt)
                logger.info("Saved checkpoint: %s", ckpt)

        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, eval_dataset: Any | None = None) -> dict[str, float]:
        """Compute KL loss on *eval_dataset* (or main_dataset if None)."""
        ds = eval_dataset or self.main_dataset
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                            collate_fn=ds.collate_fn)
        self.student.eval()
        total_kl = 0.0
        n = 0
        use_amp = self.scaler is not None
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device).float()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    t_logits = self.teacher(ids).logits
                    s_logits = self.student(ids).logits
                    kl = self.kd_loss_fn(t_logits, s_logits, mask=mask)
                total_kl += kl.item()
                n += 1
        self.student.train()
        avg_kl = total_kl / max(n, 1)
        logger.info("Eval KL loss: %.4f", avg_kl)
        return {"kl_loss": avg_kl}
