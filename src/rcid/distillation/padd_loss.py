"""Position-Adaptive Divergence Distillation (PADD) loss.

Adaptively mixes forward KL and reverse KL per token position based on
the teacher's predictive entropy:

- **Low-entropy** positions (teacher is confident): bias toward reverse KL
  (mode-seeking) so the student learns deterministic knowledge precisely.
- **High-entropy** positions (teacher is uncertain): bias toward forward KL
  (mode-covering) so the student preserves dark knowledge across options.

The mixing coefficient for each position *t* is:

    alpha(t) = sigmoid((H(p_T(t)) - mu_H) / tau)

where *mu_H* is the mean teacher entropy over valid positions in the batch,
and *tau* controls how sharply the two modes separate.

Final per-position loss:

    L(t) = alpha(t) * KL(p_T || p_S)  +  (1 - alpha(t)) * KL(p_S || p_T)

When tau -> inf, alpha -> 0.5 everywhere and the loss degenerates to JSD.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PADDLoss(nn.Module):
    """Position-Adaptive Divergence Distillation loss.

    Args:
        temperature: KD temperature for logit scaling (default 2.0).
        tau: Sharpness of the forward/reverse mixing sigmoid (default 1.0).
        alpha_min: Lower clamp for alpha (default 0.1).
        alpha_max: Upper clamp for alpha (default 0.9).
    """

    def __init__(
        self,
        temperature: float = 2.0,
        tau: float = 1.0,
        alpha_min: float = 0.1,
        alpha_max: float = 0.9,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.tau = tau
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        teacher_logits: torch.Tensor,   # (B, L, V)
        student_logits: torch.Tensor,   # (B, L, V)
        mask: torch.Tensor | None = None,  # (B, L), 1=valid 0=pad
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute PADD loss and auxiliary statistics.

        Returns:
            loss: scalar tensor (differentiable w.r.t. student_logits).
            stats: dict with ``alpha_mean``, ``forward_kl_mean``,
                   ``reverse_kl_mean``, ``teacher_entropy_mean``.
        """
        assert teacher_logits.dim() == 3, (
            f"Expected 3D teacher_logits, got {teacher_logits.dim()}D"
        )
        assert student_logits.dim() == 3, (
            f"Expected 3D student_logits, got {student_logits.dim()}D"
        )

        T = self.temperature

        # Upcast to FP32 for numerical stability (vocab ~152 k for Qwen3)
        t_scaled = teacher_logits.float() / T  # (B, L, V)
        s_scaled = student_logits.float() / T  # (B, L, V)

        t_log_probs = F.log_softmax(t_scaled, dim=-1)  # (B, L, V)
        s_log_probs = F.log_softmax(s_scaled, dim=-1)  # (B, L, V)
        t_probs = t_log_probs.exp()                     # (B, L, V)
        s_probs = s_log_probs.exp()                     # (B, L, V)

        # ── Per-position teacher entropy ─────────────────────────────
        # H(p_T) = -sum(p_T * log(p_T));  use t_log_probs for stability
        entropy = -(t_probs * t_log_probs).sum(dim=-1)  # (B, L)

        # ── Adaptive mixing weight alpha ─────────────────────────────
        alpha = self._compute_alpha(entropy, mask)  # (B, L)

        # ── Forward KL: KL(p_T || p_S) ──────────────────────────────
        # = sum(p_T * (log(p_T) - log(p_S)))
        fwd_kl = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)  # (B, L)

        # ── Reverse KL: KL(p_S || p_T) ──────────────────────────────
        # = sum(p_S * (log(p_S) - log(p_T)))
        rev_kl = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)  # (B, L)

        # ── Mixed per-position loss ──────────────────────────────────
        per_pos = alpha * fwd_kl + (1.0 - alpha) * rev_kl  # (B, L)

        # ── Aggregate with mask ──────────────────────────────────────
        if mask is not None:
            mask_f = mask.float()  # (B, L)
            denom = mask_f.sum().clamp(min=1.0)
            loss = (per_pos * mask_f).sum() / denom
            # Stats: averages over valid positions only
            alpha_mean = (alpha * mask_f).sum() / denom
            fwd_mean = (fwd_kl * mask_f).sum() / denom
            rev_mean = (rev_kl * mask_f).sum() / denom
            ent_mean = (entropy * mask_f).sum() / denom
        else:
            loss = per_pos.mean()
            alpha_mean = alpha.mean()
            fwd_mean = fwd_kl.mean()
            rev_mean = rev_kl.mean()
            ent_mean = entropy.mean()

        # Scale by T^2 (standard KD convention)
        loss = loss * (T * T)

        assert loss.isfinite(), f"PADD loss is {loss.item()}"

        stats: dict[str, float] = {
            "alpha_mean": alpha_mean.item(),
            "forward_kl_mean": fwd_mean.item(),
            "reverse_kl_mean": rev_mean.item(),
            "teacher_entropy_mean": ent_mean.item(),
        }
        return loss, stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_alpha(
        self, entropy: torch.Tensor, mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute per-position mixing weight alpha in [alpha_min, alpha_max].

        alpha(t) = clamp(sigmoid((H(t) - mu_H) / tau), alpha_min, alpha_max)

        Args:
            entropy: (B, L) per-position teacher entropy.
            mask: (B, L) or None.

        Returns:
            alpha: (B, L) mixing weights.
        """
        if mask is not None:
            mask_f = mask.float()  # (B, L)
            mu_H = (entropy * mask_f).sum() / mask_f.sum().clamp(min=1.0)
        else:
            mu_H = entropy.mean()

        alpha = torch.sigmoid((entropy - mu_H) / self.tau)  # (B, L)
        alpha = alpha.clamp(min=self.alpha_min, max=self.alpha_max)
        return alpha
