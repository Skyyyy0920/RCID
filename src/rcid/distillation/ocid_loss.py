"""Output-level Causal Imprint Distillation (OCID) loss.

Aligns teacher and student output sensitivity to minimal input perturbations.
Unlike RCID (residual stream alignment), OCID operates in the shared vocabulary
probability space, requiring no Procrustes alignment, layer mapping, or causal
checkpoint search.

Usage::

    loss_fn = OCIDLoss(temperature=2.0)
    loss, stats = loss_fn(
        teacher_logits_clean,   # (B, L, V)
        teacher_logits_corrupt, # (B, L, V)
        student_logits_clean,   # (B, L, V)
        student_logits_corrupt, # (B, L, V)
        mask=attention_mask,    # (B, L) optional
    )
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OCIDLoss(nn.Module):
    """Output-level Causal Imprint Distillation loss.

    Measures whether the student's output distribution changes in the
    same *direction* as the teacher's when a minimal input perturbation
    is applied.  Uses cosine similarity weighted by the magnitude of
    the teacher's output change at each position.

    Args:
        temperature: Softmax temperature for computing probability
            distributions (default 2.0, same as KD temperature).
        min_delta_norm: Minimum ||Δ_T|| to include a position.
            Positions below this threshold are ignored (cosine is
            numerically unstable near zero vectors). Default 1e-6.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        min_delta_norm: float = 1e-6,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.min_delta_norm = min_delta_norm

    def forward(
        self,
        teacher_logits_clean: torch.Tensor,    # (B, L, V)
        teacher_logits_corrupt: torch.Tensor,  # (B, L, V)
        student_logits_clean: torch.Tensor,    # (B, L, V)
        student_logits_corrupt: torch.Tensor,  # (B, L, V)
        mask: torch.Tensor | None = None,      # (B, L)
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute OCID loss.

        Returns:
            loss: Scalar tensor (with grad).
            stats: Dict with diagnostic values:
                - ``ocid_loss``: loss value (float)
                - ``mean_teacher_norm``: avg ||Δ_T|| at active positions
                - ``mean_cos_sim``: avg cosine similarity at active positions
                - ``n_active_positions``: number of positions above threshold
        """
        assert teacher_logits_clean.dim() == 3  # (B, L, V)
        T = self.temperature

        # ── Probability distributions ─────────────────────────────────
        p_T_clean = F.softmax(teacher_logits_clean.float() / T, dim=-1)
        p_T_corrupt = F.softmax(teacher_logits_corrupt.float() / T, dim=-1)
        p_S_clean = F.softmax(student_logits_clean.float() / T, dim=-1)
        p_S_corrupt = F.softmax(student_logits_corrupt.float() / T, dim=-1)

        # ── Output sensitivity vectors ────────────────────────────────
        delta_T = (p_T_clean - p_T_corrupt).detach()  # (B, L, V) — no grad
        delta_S = p_S_clean - p_S_corrupt              # (B, L, V) — keep grad

        # ── Teacher norm as weight ────────────────────────────────────
        teacher_norm = delta_T.norm(dim=-1)  # (B, L)

        # Active mask: positions where teacher actually changes output
        active = teacher_norm > self.min_delta_norm  # (B, L)
        if mask is not None:
            active = active & (mask > 0)

        # ── Cosine similarity ─────────────────────────────────────────
        cos_sim = F.cosine_similarity(delta_S, delta_T, dim=-1)  # (B, L)
        per_pos_loss = 1.0 - cos_sim  # (B, L), in [0, 2]

        # ── Weighted average over active positions ────────────────────
        weights = teacher_norm * active.float()  # (B, L)
        weight_sum = weights.sum().clamp(min=1e-8)
        loss = (weights * per_pos_loss).sum() / weight_sum

        assert loss.isfinite(), f"OCID loss is {loss.item()}"

        # ── Diagnostics ──────────────────────────────────────────────
        n_active = active.sum().item()
        with torch.no_grad():
            mean_cos = (
                (weights * cos_sim).sum() / weight_sum
            ).item() if n_active > 0 else 0.0
            mean_tnorm = (
                weights.sum() / max(n_active, 1)
            ).item()

        stats: dict[str, float] = {
            "ocid_loss": loss.item(),
            "mean_teacher_norm": mean_tnorm,
            "mean_cos_sim": mean_cos,
            "n_active_positions": n_active,
        }
        return loss, stats
