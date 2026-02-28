"""Adaptive KL divergence losses for knowledge distillation.

Implements two adaptive mixing strategies for forward and reverse KL:

1. **AKLLoss** — Reproduces AKL (Wu et al., COLING 2025).  Splits the
   teacher's probability mass into a *head* (top cumulative-prob >= mu) and
   a *tail*, computes L1 gaps in each region, and uses the head-gap ratio
   as the per-position mixing weight alpha.

2. **KLRatioLoss** — Our proposed method.  Uses the ratio
   ``FKL / (FKL + RKL)`` directly as the mixing signal, with two
   granularity options: per-token or batch-level with EMA smoothing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Shared helpers
# ======================================================================

def _compute_fkl_rkl(
    t_log_probs: torch.Tensor,  # (B, L, V)
    s_log_probs: torch.Tensor,  # (B, L, V)
    t_probs: torch.Tensor,      # (B, L, V)
    s_probs: torch.Tensor,      # (B, L, V)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position forward and reverse KL (before T^2 scaling).

    Returns:
        fkl: (B, L) — sum_v p * (log p - log q)
        rkl: (B, L) — sum_v q * (log q - log p)
    """
    diff = t_log_probs - s_log_probs  # (B, L, V)
    fkl = (t_probs * diff).sum(dim=-1)      # (B, L)
    rkl = (s_probs * (-diff)).sum(dim=-1)    # (B, L)
    return fkl, rkl


def _masked_mean(
    values: torch.Tensor, mask: torch.Tensor | None,
) -> torch.Tensor:
    """Mean of *values* over positions where mask == 1."""
    if mask is not None:
        mask_f = mask.float()
        return (values * mask_f).sum() / mask_f.sum().clamp(min=1.0)
    return values.mean()


# ======================================================================
# AKL (Wu et al., COLING 2025)
# ======================================================================

class AKLLoss(nn.Module):
    """Adaptive KL Divergence (Wu et al., COLING 2025).

    For each token position the teacher distribution is split into a *head*
    (tokens whose cumulative sorted probability >= ``mu``) and a *tail*.
    The L1 gap between teacher and student in each region determines alpha::

        alpha = g_head / (g_head + g_tail + eps)
        loss  = alpha * FKL  +  (1 - alpha) * RKL

    Args:
        temperature: KD temperature (default 2.0).
        mu: Cumulative-probability threshold for head/tail split (default 0.5).
    """

    def __init__(
        self, temperature: float = 2.0, mu: float = 0.5,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.mu = mu

    def forward(
        self,
        teacher_logits: torch.Tensor,       # (B, L, V)
        student_logits: torch.Tensor,       # (B, L, V)
        mask: torch.Tensor | None = None,   # (B, L)
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert teacher_logits.dim() == 3
        T = self.temperature
        eps = 1e-8

        # ── Probabilities (FP32 for stability) ──────────────────────
        t_scaled = teacher_logits.float() / T   # (B, L, V)
        s_scaled = student_logits.float() / T   # (B, L, V)

        t_log_probs = F.log_softmax(t_scaled, dim=-1)  # (B, L, V)
        s_log_probs = F.log_softmax(s_scaled, dim=-1)  # (B, L, V)
        t_probs = t_log_probs.exp()                     # (B, L, V)
        s_probs = s_log_probs.exp()                     # (B, L, V)

        # ── Head / tail mask via cumulative probability ──────────────
        # Sort teacher probs descending along vocab dim
        sorted_probs, _ = torch.sort(t_probs, dim=-1, descending=True)  # (B, L, V)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)                  # (B, L, V)

        # Threshold: first index where cumsum >= mu.
        # Use the *original* (unsorted) probs to build the mask:
        # a token belongs to head iff its prob >= the prob at the cutoff rank.
        # Cutoff rank per position: number of tokens needed to reach mu.
        # n_head[b,l] = first k such that cumsum[b,l,k] >= mu  (1-indexed)
        above_mu = (cum_probs >= self.mu).float()               # (B, L, V)
        # argmax gives the first True index along last dim
        cutoff_idx = above_mu.argmax(dim=-1)                     # (B, L)
        # Probability at the cutoff rank
        cutoff_prob = sorted_probs.gather(
            -1, cutoff_idx.unsqueeze(-1),
        ).squeeze(-1)  # (B, L)

        # head_mask[b,l,v] = 1 if t_probs[b,l,v] >= cutoff_prob[b,l]
        head_mask = (t_probs >= cutoff_prob.unsqueeze(-1)).float()  # (B, L, V)

        # Free sorted tensors
        del sorted_probs, cum_probs, above_mu

        # ── L1 gaps ─────────────────────────────────────────────────
        abs_diff = (t_probs - s_probs).abs()             # (B, L, V)
        g_head = (head_mask * abs_diff).sum(dim=-1)       # (B, L)
        g_tail = ((1.0 - head_mask) * abs_diff).sum(dim=-1)  # (B, L)
        del abs_diff, head_mask

        # ── Per-position alpha (detached) ────────────────────────────
        alpha = (g_head / (g_head + g_tail + eps)).detach()  # (B, L)

        # ── FKL / RKL ───────────────────────────────────────────────
        fkl, rkl = _compute_fkl_rkl(t_log_probs, s_log_probs, t_probs, s_probs)

        # ── Mixed loss ──────────────────────────────────────────────
        per_pos = alpha * fkl + (1.0 - alpha) * rkl  # (B, L)

        loss = _masked_mean(per_pos, mask) * (T * T)
        assert loss.isfinite(), f"AKL loss is {loss.item()}"

        stats: dict[str, float] = {
            "alpha_mean": _masked_mean(alpha, mask).item(),
            "forward_kl_mean": _masked_mean(fkl, mask).item(),
            "reverse_kl_mean": _masked_mean(rkl, mask).item(),
            "g_head_mean": _masked_mean(g_head, mask).item(),
            "g_tail_mean": _masked_mean(g_tail, mask).item(),
        }
        return loss, stats


# ======================================================================
# KL-Ratio Adaptive Loss (ours)
# ======================================================================

class KLRatioLoss(nn.Module):
    """KL-Ratio Adaptive Divergence for Knowledge Distillation.

    Uses ``FKL / (FKL + RKL)`` as the adaptive mixing signal — zero
    additional computation since both KL terms are already needed.

    Two granularities:
      * ``'token'`` — per-position alpha (same granularity as AKL).
      * ``'batch'`` — batch-level alpha smoothed with an EMA.

    Special modes via ``fixed_alpha``:
      * ``fixed_alpha=0.5`` → Jeffreys divergence (0.5 FKL + 0.5 RKL).
      * ``fixed_alpha=0.0`` → pure reverse KL.
      * ``fixed_alpha=1.0`` → pure forward KL.
      When ``fixed_alpha`` is set, ``granularity`` and ``beta`` are ignored.

    Args:
        temperature: KD temperature (default 2.0).
        granularity: ``'token'`` or ``'batch'`` (default ``'token'``).
        beta: EMA coefficient, only for ``granularity='batch'`` (default 0.99).
        fixed_alpha: If not None, use this constant alpha (no adaptation).
    """

    def __init__(
        self,
        temperature: float = 2.0,
        granularity: str = "token",
        beta: float = 0.99,
        fixed_alpha: float | None = None,
    ) -> None:
        super().__init__()
        if fixed_alpha is None:
            assert granularity in ("token", "batch"), (
                f"granularity must be 'token' or 'batch', got {granularity!r}"
            )
        self.temperature = temperature
        self.granularity = granularity
        self.beta = beta
        self.fixed_alpha = fixed_alpha
        # EMA state (batch granularity only); stored as buffer so it
        # survives .to(device) and state_dict but does NOT get gradients.
        self.register_buffer("alpha_ema", torch.tensor(0.5))

    def forward(
        self,
        teacher_logits: torch.Tensor,       # (B, L, V)
        student_logits: torch.Tensor,       # (B, L, V)
        mask: torch.Tensor | None = None,   # (B, L)
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert teacher_logits.dim() == 3
        T = self.temperature
        eps = 1e-8

        # ── Probabilities ────────────────────────────────────────────
        t_scaled = teacher_logits.float() / T   # (B, L, V)
        s_scaled = student_logits.float() / T   # (B, L, V)

        t_log_probs = F.log_softmax(t_scaled, dim=-1)  # (B, L, V)
        s_log_probs = F.log_softmax(s_scaled, dim=-1)  # (B, L, V)
        t_probs = t_log_probs.exp()                     # (B, L, V)
        s_probs = s_log_probs.exp()                     # (B, L, V)

        # ── FKL / RKL per position ───────────────────────────────────
        fkl, rkl = _compute_fkl_rkl(
            t_log_probs, s_log_probs, t_probs, s_probs,
        )  # each (B, L)

        # ── Adaptive alpha ───────────────────────────────────────────
        if self.fixed_alpha is not None:
            # Fixed mode: constant alpha (Jeffreys, reverse KL, etc.)
            alpha = self.fixed_alpha  # scalar, broadcast to (B, L)
            alpha_mean = self.fixed_alpha
            alpha_std = 0.0
        elif self.granularity == "token":
            alpha = (fkl / (fkl + rkl + eps)).detach()  # (B, L)
            alpha_mean = _masked_mean(alpha, mask).item()
            # Compute per-position std for diagnostics
            if mask is not None:
                mask_f = mask.float()
                denom = mask_f.sum().clamp(min=1.0)
                mean_t = (alpha * mask_f).sum() / denom
                var_t = ((alpha - mean_t).pow(2) * mask_f).sum() / denom
            else:
                var_t = alpha.var()
            alpha_std = var_t.sqrt().item()
        else:
            # Batch-level with EMA
            fkl_batch = _masked_mean(fkl, mask)  # scalar
            rkl_batch = _masked_mean(rkl, mask)   # scalar
            alpha_instant = (
                fkl_batch / (fkl_batch + rkl_batch + eps)
            ).detach()

            # Update EMA (no grad)
            self.alpha_ema.fill_(
                self.beta * self.alpha_ema.item()
                + (1.0 - self.beta) * alpha_instant.item(),
            )
            alpha = self.alpha_ema.item()   # scalar, broadcast to (B, L)
            alpha_mean = alpha
            alpha_std = 0.0  # single scalar, no variance

        # ── Mixed loss ───────────────────────────────────────────────
        per_pos = alpha * fkl + (1.0 - alpha) * rkl  # (B, L)

        loss = _masked_mean(per_pos, mask) * (T * T)
        assert loss.isfinite(), f"KLRatio loss is {loss.item()}"

        stats: dict[str, float] = {
            "alpha_mean": alpha_mean,
            "alpha_std": alpha_std,
            "forward_kl_mean": _masked_mean(fkl, mask).item(),
            "reverse_kl_mean": _masked_mean(rkl, mask).item(),
        }
        if self.granularity == "batch":
            stats["alpha_ema"] = self.alpha_ema.item()

        return loss, stats
