"""Unified trainer for toy data experiments (ContrastiveDataset).

Supports standard_kd only. For adaptive KL methods on instruction data,
use ScalableDistillationTrainer instead.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rcid.circuit.contrastive import ContrastiveDataset
from rcid.distillation.baselines import StandardKDLoss
from rcid.models.adapter import ModelAdapter

logger = logging.getLogger(__name__)

VALID_METHODS = ("standard_kd",)


class UnifiedTrainer:
    """Train a student model using standard KD on contrastive datasets."""

    def __init__(
        self,
        method: str,
        teacher: nn.Module,
        student: nn.Module,
        teacher_adapter: ModelAdapter,
        student_adapter: ModelAdapter,
        dataset: ContrastiveDataset,
        config: dict[str, Any],
        tokenizer: Any | None = None,
        **kwargs: Any,  # Accept extra kwargs for backward compat
    ) -> None:
        assert method in VALID_METHODS, f"Unknown method: {method}"
        self.method = method
        self.teacher = teacher
        self.student = student
        self.t_adapter = teacher_adapter
        self.s_adapter = student_adapter
        self.dataset = dataset
        self.config = config
        self.tokenizer = tokenizer

        self.kl_mode: str = config.get("kl_mode", "sequence")
        self.device = next(student.parameters()).device
        self.teacher.eval()

        # Upcast student to FP32
        param_dtype = next(student.parameters()).dtype
        if param_dtype != torch.float32:
            logger.info("Upcasting student %s -> float32", param_dtype)
            student.float()

        self.kd_loss_fn = StandardKDLoss(
            temperature=config.get("temperature", 2.0),
        )
        self.optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=config.get("lr", 5e-5),
            weight_decay=config.get("weight_decay", 0.01),
        )
        use_amp = config.get("fp16", True)
        self.scaler = torch.amp.GradScaler() if use_amp else None
        self.use_wandb = config.get("use_wandb", False)
        self.log_every = config.get("log_every", 50)

    def train(
        self,
        epochs: int | None = None,
        batch_size: int | None = None,
        save_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Run training loop. Returns per-epoch loss history."""
        epochs = epochs or self.config.get("epochs", 20)
        batch_size = batch_size or self.config.get("batch_size", 16)
        grad_clip = self.config.get("grad_clip", 1.0)

        loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True,
            collate_fn=ContrastiveDataset.collate_fn,
        )
        history: dict[str, list[float]] = {"loss": []}
        self.student.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                clean = batch["clean_ids"].to(self.device)  # (B, seq)
                bs = clean.shape[0]

                self.optimizer.zero_grad()
                use_amp = self.scaler is not None
                with torch.amp.autocast("cuda", enabled=use_amp):
                    with torch.no_grad():
                        t_logits = self.teacher(clean).logits  # (B, seq, V)
                    s_logits = self.student(clean).logits       # (B, seq, V)

                    if self.kl_mode == "sequence":
                        mask: torch.Tensor | None = None
                        if (self.tokenizer is not None
                                and hasattr(self.tokenizer, "pad_token_id")
                                and self.tokenizer.pad_token_id is not None):
                            mask = (clean != self.tokenizer.pad_token_id).float()
                        loss = self.kd_loss_fn(t_logits, s_logits, mask=mask)
                    else:
                        answer_pos = batch["answer_pos"].to(self.device)
                        idx = torch.arange(bs, device=self.device)
                        t_at = t_logits[idx, answer_pos]  # (B, V)
                        s_at = s_logits[idx, answer_pos]  # (B, V)
                        loss = self.kd_loss_fn(t_at, s_at)

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
            logger.info("Epoch %d/%d loss=%.4f", epoch + 1, epochs, avg)

            if save_dir and (epoch + 1) % self.config.get("checkpoint_every", 5) == 0:
                p = Path(save_dir) / f"student_epoch{epoch + 1}.pt"
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.student.state_dict(), p)

        return history
