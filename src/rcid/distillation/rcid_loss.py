"""RCID loss: match contrastive differences at causal checkpoints."""

from __future__ import annotations

import torch
import torch.nn as nn


class RCIDLoss(nn.Module):
    """Residual Causal Imprint Distillation loss.

    Matches normalized contrastive differences between teacher and student
    at causally important checkpoints, after Procrustes alignment.

    W matrices are frozen buffers. Teacher imprints are pre-computed and
    detached. Student forward passes retain gradients.
    """

    def __init__(
        self,
        checkpoints: list[tuple[int, int]],   # [(t_layer, t_pos), ...]
        layer_mapping: dict[int, int],         # {t_layer: s_layer}
        W_matrices: dict[int, torch.Tensor],   # {t_layer: W (d_T, d_S)}
    ) -> None:
        super().__init__()
        self.checkpoints = checkpoints
        self.layer_mapping = layer_mapping
        for t_layer, W in W_matrices.items():
            self.register_buffer(f"W_{t_layer}", W)

    def forward(
        self,
        teacher_imprints: dict[tuple[int, int], torch.Tensor],
        # {(t_layer, t_pos): (batch, d_T)} — detached, no grad
        student_clean_residuals: dict[int, torch.Tensor],
        # {s_layer: (batch, seq_len, d_model)} — has grad
        student_corrupt_residuals: dict[int, torch.Tensor],
        # {s_layer: (batch, seq_len, d_model)} — has grad
    ) -> torch.Tensor:
        eps = 1e-8
        device = next(iter(teacher_imprints.values())).device
        total = torch.tensor(0.0, device=device)

        for t_layer, t_pos in self.checkpoints:
            s_layer = self.layer_mapping[t_layer]
            W = getattr(self, f"W_{t_layer}")  # (d_T, d_S)

            d_T = teacher_imprints[(t_layer, t_pos)]  # (batch, d_T), no grad

            d_S = (
                student_clean_residuals[s_layer][:, t_pos, :]
                - student_corrupt_residuals[s_layer][:, t_pos, :]
            )  # (batch, d_S), has grad

            aligned = d_S @ W.T  # (batch, d_T)
            aligned_n = aligned / aligned.norm(dim=-1, keepdim=True).clamp(min=eps)
            d_T_n = d_T / d_T.norm(dim=-1, keepdim=True).clamp(min=eps)

            total = total + (aligned_n - d_T_n).pow(2).sum(dim=-1).mean()

        total = total / max(len(self.checkpoints), 1)
        assert total.isfinite(), f"RCID loss is {total.item()}"
        return total
