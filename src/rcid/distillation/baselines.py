"""Baseline distillation losses: StandardKD, FitNets, InformedFitNets."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardKDLoss(nn.Module):
    """Standard Knowledge Distillation via KL divergence on logits."""

    def __init__(self, temperature: float = 2.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        teacher_logits: torch.Tensor,  # (batch, seq, vocab) or (batch, vocab)
        student_logits: torch.Tensor,  # same shape as teacher
    ) -> torch.Tensor:
        T = self.temperature
        t_probs = F.softmax(teacher_logits / T, dim=-1)
        s_log_probs = F.log_softmax(student_logits / T, dim=-1)
        loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)
        assert loss.isfinite(), f"KD loss is {loss.item()}"
        return loss


class FitNetsLoss(nn.Module):
    """FitNets: match student representations to teacher at all mapped layers."""

    def __init__(
        self,
        layer_mapping: dict[int, int],         # {t_layer: s_layer}
        W_matrices: dict[int, torch.Tensor],   # {t_layer: W (d_T, d_S)}
    ) -> None:
        super().__init__()
        self.layer_mapping = layer_mapping
        for t_layer, W in W_matrices.items():
            self.register_buffer(f"W_{t_layer}", W)

    def forward(
        self,
        teacher_residuals: dict[int, torch.Tensor],
        # {t_layer: (batch, seq_len, d_T)} — detached
        student_residuals: dict[int, torch.Tensor],
        # {s_layer: (batch, seq_len, d_S)} — has grad
    ) -> torch.Tensor:
        total = torch.tensor(0.0)
        count = 0

        for t_layer, s_layer in self.layer_mapping.items():
            if t_layer not in teacher_residuals or s_layer not in student_residuals:
                continue
            W = getattr(self, f"W_{t_layer}")  # (d_T, d_S)
            h_T = teacher_residuals[t_layer]   # (batch, seq, d_T)
            h_S = student_residuals[s_layer]   # (batch, seq, d_S)

            if total.device != h_T.device:
                total = total.to(h_T.device)

            # Project student: (batch, seq, d_S) @ (d_S, d_T) -> (batch, seq, d_T)
            aligned = h_S @ W.T  # (batch, seq, d_T)
            total = total + F.mse_loss(aligned, h_T)
            count += 1

        if count > 0:
            total = total / count
        assert total.isfinite(), f"FitNets loss is {total.item()}"
        return total


class InformedFitNetsLoss(nn.Module):
    """InformedFitNets: FitNets at causally important checkpoints only.

    Shares checkpoints and W matrices with RCID. Matches h^T_clean
    (full representation) rather than d^T (contrastive difference).
    This is the ablation baseline separating "right positions" from
    "contrastive difference matching".
    """

    def __init__(
        self,
        checkpoints: list[tuple[int, int]],
        layer_mapping: dict[int, int],
        W_matrices: dict[int, torch.Tensor],
    ) -> None:
        super().__init__()
        self.checkpoints = checkpoints
        self.layer_mapping = layer_mapping
        for t_layer, W in W_matrices.items():
            self.register_buffer(f"W_{t_layer}", W)

    def forward(
        self,
        teacher_clean_residuals: dict[int, torch.Tensor],
        # {t_layer: (batch, seq_len, d_T)} — detached
        student_clean_residuals: dict[int, torch.Tensor],
        # {s_layer: (batch, seq_len, d_S)} — has grad
    ) -> torch.Tensor:
        eps = 1e-8
        device = next(iter(teacher_clean_residuals.values())).device
        total = torch.tensor(0.0, device=device)

        for t_layer, t_pos in self.checkpoints:
            s_layer = self.layer_mapping[t_layer]
            W = getattr(self, f"W_{t_layer}")

            h_T = teacher_clean_residuals[t_layer][:, t_pos, :]  # (batch, d_T)
            h_S = student_clean_residuals[s_layer][:, t_pos, :]  # (batch, d_S)

            aligned = h_S @ W.T  # (batch, d_T)
            aligned_n = aligned / aligned.norm(dim=-1, keepdim=True).clamp(min=eps)
            h_T_n = h_T / h_T.norm(dim=-1, keepdim=True).clamp(min=eps)

            total = total + (aligned_n - h_T_n).pow(2).sum(dim=-1).mean()

        total = total / max(len(self.checkpoints), 1)
        assert total.isfinite(), f"InformedFitNets loss is {total.item()}"
        return total
