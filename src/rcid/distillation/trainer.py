"""Unified trainer for all 4 distillation methods."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rcid.circuit.contrastive import ContrastiveDataset
from rcid.circuit.patching import extract_contrastive_differences, extract_residual_at_layers
from rcid.distillation.baselines import FitNetsLoss, InformedFitNetsLoss, StandardKDLoss
from rcid.distillation.rcid_loss import RCIDLoss
from rcid.models.adapter import ModelAdapter

logger = logging.getLogger(__name__)

VALID_METHODS = ("standard_kd", "fitnets", "informed_fitnets", "rcid")


class UnifiedTrainer:
    """Train a student model using one of 4 distillation methods."""

    def __init__(
        self,
        method: str,
        teacher: nn.Module,
        student: nn.Module,
        teacher_adapter: ModelAdapter,
        student_adapter: ModelAdapter,
        dataset: ContrastiveDataset,
        config: dict[str, Any],
        checkpoints: list[tuple[int, int]] | None = None,
        layer_mapping: dict[int, int] | None = None,
        W_matrices: dict[int, torch.Tensor] | None = None,
    ) -> None:
        assert method in VALID_METHODS, f"Unknown method: {method}"
        self.method = method
        self.teacher = teacher
        self.student = student
        self.t_adapter = teacher_adapter
        self.s_adapter = student_adapter
        self.dataset = dataset
        self.config = config
        self.checkpoints = checkpoints or []
        self.layer_mapping = layer_mapping or {}
        self.W_matrices = W_matrices or {}

        self.device = next(student.parameters()).device
        self.teacher.eval()

        # Build loss functions
        self.kd_loss_fn = StandardKDLoss(temperature=config.get("temperature", 2.0))
        self._build_method_loss()

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            student.parameters(), lr=config.get("lr", 5e-5),
            weight_decay=config.get("weight_decay", 0.01),
        )
        self.scaler = torch.amp.GradScaler() if config.get("fp16", True) else None

        # Pre-compute teacher imprints for RCID
        self.teacher_imprints: dict[tuple[int, int], torch.Tensor] = {}
        if method == "rcid":
            self._precompute_teacher_imprints()

    def _build_method_loss(self) -> None:
        """Instantiate method-specific loss module."""
        if self.method == "fitnets":
            self.method_loss_fn = FitNetsLoss(self.layer_mapping, self.W_matrices)
        elif self.method == "informed_fitnets":
            self.method_loss_fn = InformedFitNetsLoss(
                self.checkpoints, self.layer_mapping, self.W_matrices,
            )
        elif self.method == "rcid":
            self.method_loss_fn = RCIDLoss(
                self.checkpoints, self.layer_mapping, self.W_matrices,
            )
        else:
            self.method_loss_fn = None

    def _precompute_teacher_imprints(self) -> None:
        """Pre-compute teacher contrastive differences at checkpoints."""
        t_layers = list({cp[0] for cp in self.checkpoints})
        diffs = extract_contrastive_differences(
            self.teacher, self.t_adapter,
            self.dataset.clean_ids.to(self.device),
            self.dataset.corrupt_ids.to(self.device),
            layers=t_layers,
        )
        for t_layer, t_pos in self.checkpoints:
            self.teacher_imprints[(t_layer, t_pos)] = (
                diffs[t_layer][:, t_pos, :].detach()  # (N, d_T)
            )

    def _collect_student_residuals(
        self, input_ids: torch.Tensor, layers: list[int]
    ) -> dict[int, torch.Tensor]:
        """Forward student and collect residual activations with gradients."""
        cache: dict[int, torch.Tensor] = {}
        handles = []
        for layer_idx in layers:
            hook_point = self.s_adapter.get_residual_hook_point(
                self.student, layer_idx
            )
            def _make_hook(idx: int):
                def _hook(mod, inp, out):
                    cache[idx] = self.s_adapter.parse_layer_output(out)
                return _hook
            handles.append(hook_point.register_forward_hook(_make_hook(layer_idx)))

        try:
            logits = self.student(input_ids).logits
        finally:
            for h in handles:
                h.remove()
        return cache, logits

    def train(
        self,
        epochs: int | None = None,
        batch_size: int | None = None,
        save_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Run training loop. Returns per-epoch loss history."""
        epochs = epochs or self.config.get("epochs", 20)
        batch_size = batch_size or self.config.get("batch_size", 16)
        lambda_kl = self.config.get("lambda_kl", 1.0)
        lambda_method = self.config.get("lambda_rcid", 1.0)
        grad_clip = self.config.get("grad_clip", 1.0)

        loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True,
            collate_fn=ContrastiveDataset.collate_fn,
        )
        s_layers = list(set(self.layer_mapping.values()))
        history: dict[str, list[float]] = {"loss": [], "kd_loss": [], "method_loss": []}
        self.student.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                clean = batch["clean_ids"].to(self.device)
                corrupt = batch["corrupt_ids"].to(self.device)
                answer_pos = batch["answer_pos"].to(self.device)
                batch_size_actual = clean.shape[0]

                self.optimizer.zero_grad()
                use_amp = self.scaler is not None
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = self._compute_loss(
                        clean, corrupt, answer_pos, s_layers,
                        batch_size_actual, lambda_kl, lambda_method,
                        history,
                    )

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.student.parameters(), grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.student.parameters(), grad_clip)
                    self.optimizer.step()

                epoch_loss += loss.item()

            avg = epoch_loss / max(len(loader), 1)
            history["loss"].append(avg)
            logger.info(f"Epoch {epoch+1}/{epochs} loss={avg:.4f}")

            if save_dir and (epoch + 1) % 5 == 0:
                p = Path(save_dir) / f"student_epoch{epoch+1}.pt"
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.student.state_dict(), p)

        return history

    def _compute_loss(
        self, clean, corrupt, answer_pos, s_layers,
        bs, lambda_kl, lambda_method, history,
    ) -> torch.Tensor:
        """Compute combined loss for one batch."""
        # Teacher forward (no grad)
        with torch.no_grad():
            t_logits = self.teacher(clean).logits  # (batch, seq, vocab)

        # Student clean forward + residuals
        if self.method in ("fitnets", "informed_fitnets", "rcid") and s_layers:
            s_clean_cache, s_logits = self._collect_student_residuals(clean, s_layers)
        else:
            s_logits = self.student(clean).logits
            s_clean_cache = {}

        # KD loss at answer positions
        batch_idx = torch.arange(bs, device=clean.device)
        t_at_ans = t_logits[batch_idx, answer_pos]  # (batch, vocab)
        s_at_ans = s_logits[batch_idx, answer_pos]   # (batch, vocab)
        kd_loss = self.kd_loss_fn(t_at_ans, s_at_ans)

        # Method-specific loss
        method_loss = torch.tensor(0.0, device=clean.device)
        if self.method == "fitnets" and self.method_loss_fn is not None:
            t_residuals = extract_residual_at_layers(
                self.teacher, self.t_adapter, clean,
                layers=list(self.layer_mapping.keys()),
            )
            method_loss = self.method_loss_fn(t_residuals, s_clean_cache)
        elif self.method == "informed_fitnets" and self.method_loss_fn is not None:
            t_residuals = extract_residual_at_layers(
                self.teacher, self.t_adapter, clean,
                layers=list({cp[0] for cp in self.checkpoints}),
            )
            method_loss = self.method_loss_fn(t_residuals, s_clean_cache)
        elif self.method == "rcid" and self.method_loss_fn is not None:
            s_corrupt_cache, _ = self._collect_student_residuals(corrupt, s_layers)
            # Slice teacher imprints for this batch
            batch_imprints = {}
            for key, full_val in self.teacher_imprints.items():
                batch_imprints[key] = full_val[:bs]
            method_loss = self.method_loss_fn(
                batch_imprints, s_clean_cache, s_corrupt_cache,
            )

        loss = lambda_kl * kd_loss + lambda_method * method_loss
        history.setdefault("kd_loss", []).append(kd_loss.item())
        history.setdefault("method_loss", []).append(method_loss.item())
        return loss
