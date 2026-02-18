"""Out-of-distribution robustness evaluation."""

from __future__ import annotations

import torch.nn as nn

from rcid.circuit.contrastive import ContrastiveDataset
from rcid.eval.task_accuracy import evaluate_task_accuracy
from rcid.models.adapter import ModelAdapter


def evaluate_ood_robustness(
    model: nn.Module,
    adapter: ModelAdapter,
    ood_dataset: ContrastiveDataset,
    id_accuracy: float,
    batch_size: int = 32,
) -> dict[str, float]:
    """Evaluate OOD robustness by comparing ID and OOD accuracy.

    Args:
        model: Trained student model.
        adapter: Model adapter.
        ood_dataset: Out-of-distribution contrastive dataset.
        id_accuracy: In-distribution accuracy for computing degradation.
        batch_size: Batch size for evaluation.

    Returns:
        Dict with ood_accuracy, degradation (id - ood), and relative_degradation.
    """
    ood_results = evaluate_task_accuracy(model, adapter, ood_dataset, batch_size)
    ood_acc = ood_results["accuracy"]

    degradation = id_accuracy - ood_acc
    relative_deg = degradation / max(id_accuracy, 1e-10)

    return {
        "ood_accuracy": ood_acc,
        "degradation": degradation,
        "relative_degradation": relative_deg,
        "ood_logit_diff_mean": ood_results["logit_diff_mean"],
    }
