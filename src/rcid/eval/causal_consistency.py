"""Causal consistency evaluation between teacher and student."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from rcid.circuit.contrastive import ContrastiveDataset
from rcid.circuit.intervention import compute_causal_effect
from rcid.models.adapter import ModelAdapter

logger = logging.getLogger(__name__)


def _pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pure-torch Pearson correlation coefficient."""
    eps = 1e-10
    x_c = x - x.mean()
    y_c = y - y.mean()
    num = (x_c * y_c).sum()
    den = (x_c.pow(2).sum() * y_c.pow(2).sum()).sqrt().clamp(min=eps)
    return (num / den).item()


class CausalConsistencyEvaluator:
    """Evaluate mechanistic consistency via causal interventions."""

    def evaluate(
        self,
        teacher: nn.Module,
        student: nn.Module,
        teacher_adapter: ModelAdapter,
        student_adapter: ModelAdapter,
        dataset: ContrastiveDataset,
        checkpoints: list[tuple[int, int]],
        layer_mapping: dict[int, int],
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """Compute Pearson correlation of causal effects."""
        device = next(teacher.parameters()).device
        clean = dataset.clean_ids.to(device)
        corrupt = dataset.corrupt_ids.to(device)
        answer_pos = dataset.answer_pos.to(device)
        correct_id = dataset.correct_token_id.to(device)
        wrong_id = dataset.wrong_token_id.to(device)

        results: dict[str, Any] = {"per_checkpoint": {}, "mean_correlation": 0.0}
        correlations: list[float] = []

        for t_layer, t_pos in checkpoints:
            s_layer = layer_mapping.get(t_layer, t_layer)

            delta_T = compute_causal_effect(
                teacher, teacher_adapter,
                clean, corrupt,
                layer=t_layer, token_pos=t_pos,
                answer_pos=answer_pos,
                correct_token_id=correct_id,
                wrong_token_id=wrong_id,
            ).cpu()  # (N,)

            delta_S = compute_causal_effect(
                student, student_adapter,
                clean, corrupt,
                layer=s_layer, token_pos=t_pos,
                answer_pos=answer_pos,
                correct_token_id=correct_id,
                wrong_token_id=wrong_id,
            ).cpu()  # (N,)

            if delta_T.std() < 1e-10 or delta_S.std() < 1e-10:
                r = 0.0
                logger.warning(f"Degenerate at ({t_layer},{t_pos}): near-zero std")
            else:
                r = _pearson_r(delta_T, delta_S)

            results["per_checkpoint"][(t_layer, t_pos)] = {
                "correlation": r,
                "delta_T_mean": delta_T.mean().item(),
                "delta_S_mean": delta_S.mean().item(),
            }
            correlations.append(r)

        results["mean_correlation"] = (
            sum(correlations) / max(len(correlations), 1)
        )
        return results
