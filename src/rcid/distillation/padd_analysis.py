"""PADD mechanistic analysis tools (Phase 3).

Provides utilities for:
- Analysing how the adaptive alpha distribution evolves during training.
- Comparing PADD and baseline students' prediction trajectories via
  logit-lens across layers.
- Measuring representation-structure similarity between students and
  the teacher in residual-stream space.

All functions accept pre-loaded models so callers control device placement.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from rcid.models.adapter import ModelAdapter

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Alpha distribution analysis
# ------------------------------------------------------------------

def analyze_alpha_distribution(trainer_stats: dict[str, Any]) -> dict[str, Any]:
    """Analyse the evolution of PADD alpha during training.

    Args:
        trainer_stats: Training history dict returned by
            ``ScalableDistillationTrainer.train()``, expected to contain
            per-epoch lists for ``alpha_mean``, ``forward_kl_mean``,
            ``reverse_kl_mean``, ``teacher_entropy_mean``.

    Returns:
        A summary dict with per-epoch and overall statistics::

            {
                "n_epochs": int,
                "alpha_mean_per_epoch": list[float],
                "forward_kl_per_epoch": list[float],
                "reverse_kl_per_epoch": list[float],
                "entropy_per_epoch": list[float],
                "alpha_trend": "increasing" | "decreasing" | "stable",
            }
    """
    alpha_hist = trainer_stats.get("alpha_mean", [])
    fwd_hist = trainer_stats.get("forward_kl_mean", [])
    rev_hist = trainer_stats.get("reverse_kl_mean", [])
    ent_hist = trainer_stats.get("teacher_entropy_mean", [])
    n = len(alpha_hist)

    trend = "stable"
    if n >= 2:
        delta = alpha_hist[-1] - alpha_hist[0]
        if delta > 0.02:
            trend = "increasing"
        elif delta < -0.02:
            trend = "decreasing"

    return {
        "n_epochs": n,
        "alpha_mean_per_epoch": alpha_hist,
        "forward_kl_per_epoch": fwd_hist,
        "reverse_kl_per_epoch": rev_hist,
        "entropy_per_epoch": ent_hist,
        "alpha_trend": trend,
    }


# ------------------------------------------------------------------
# Logit lens trajectory comparison
# ------------------------------------------------------------------

@torch.no_grad()
def compare_logit_lens_trajectory(
    teacher: nn.Module,
    student_padd: nn.Module,
    student_baseline: nn.Module,
    tokenizer: Any,
    samples: list[str],
    adapter: ModelAdapter,
    device: str | torch.device,
) -> dict[str, Any]:
    """Compare per-layer logit-lens predictions for two students vs teacher.

    For each sample and each transformer layer, applies the language-model
    head to the residual-stream output to obtain intermediate predictions.
    Computes teacher-student prediction agreement per layer.

    Args:
        teacher: Teacher model (eval mode).
        student_padd: PADD-trained student.
        student_baseline: StandardKD-trained student.
        tokenizer: Shared tokenizer (teacher / student use same vocab).
        samples: List of text prompts.
        adapter: ``ModelAdapter`` compatible with all three models.
        device: Compute device.

    Returns:
        ``{ "samples": list of per-sample dicts }`` where each dict
        contains per-layer agreement scores with the teacher for both
        students.
    """
    # Placeholder — full implementation deferred to Phase 3.
    raise NotImplementedError(
        "compare_logit_lens_trajectory is a Phase 3 analysis stub."
    )


# ------------------------------------------------------------------
# Representation structure analysis
# ------------------------------------------------------------------

@torch.no_grad()
def analyze_representation_structure(
    teacher: nn.Module,
    student_padd: nn.Module,
    student_baseline: nn.Module,
    samples: list[str],
    adapter: ModelAdapter,
    device: str | torch.device,
) -> dict[str, Any]:
    """Compare residual-stream structural similarity of two students.

    Uses CKA to measure how closely each student's intermediate
    representations track the teacher's, layer by layer.

    Args:
        teacher: Teacher model (eval mode).
        student_padd: PADD-trained student.
        student_baseline: StandardKD-trained student.
        samples: List of text prompts for probing.
        adapter: ``ModelAdapter`` compatible with all three models.
        device: Compute device.

    Returns:
        ``{ "cka_padd": list[float], "cka_baseline": list[float] }``
        with per-layer CKA scores.
    """
    # Placeholder — full implementation deferred to Phase 3.
    raise NotImplementedError(
        "analyze_representation_structure is a Phase 3 analysis stub."
    )
