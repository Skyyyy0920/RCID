"""Layer matching strategies for mapping teacher layers to student layers."""

from __future__ import annotations

import torch


def match_layers_greedy(
    cka_scores: torch.Tensor,  # (n_teacher, n_student)
) -> dict[int, int]:
    """For each teacher layer, pick the student layer with highest CKA.

    Many-to-one mapping is allowed (multiple teacher layers can map to same
    student layer).
    """
    n_teacher = cka_scores.shape[0]
    mapping: dict[int, int] = {}
    for t in range(n_teacher):
        best_s = cka_scores[t].argmax().item()
        mapping[t] = int(best_s)
    return mapping


def proportional_layer_mapping(
    n_teacher: int,
    n_student: int,
) -> dict[int, int]:
    """Map teacher layers to student layers proportionally.

    teacher layer l -> student layer round(l * n_student / n_teacher).
    """
    mapping: dict[int, int] = {}
    for t in range(n_teacher):
        s = round(t * (n_student - 1) / max(n_teacher - 1, 1))
        mapping[t] = min(s, n_student - 1)
    return mapping


def match_layers(
    cka_scores: torch.Tensor | None = None,
    n_teacher: int | None = None,
    n_student: int | None = None,
    strategy: str = "greedy",
) -> dict[int, int]:
    """Dispatch to the appropriate layer matching strategy.

    Args:
        cka_scores: CKA matrix (n_teacher, n_student). Required for 'greedy'.
        n_teacher: Number of teacher layers. Required for 'proportional'.
        n_student: Number of student layers. Required for 'proportional'.
        strategy: 'greedy' or 'proportional'.

    Returns:
        Dict mapping teacher layer index to student layer index.
    """
    if strategy == "greedy":
        assert cka_scores is not None, "CKA scores required for greedy matching"
        return match_layers_greedy(cka_scores)
    elif strategy == "proportional":
        assert n_teacher is not None and n_student is not None
        return proportional_layer_mapping(n_teacher, n_student)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
