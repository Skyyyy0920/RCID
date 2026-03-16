"""Teacher → student layer matching via CKA similarity or linear mapping.

Given a CKA similarity matrix ``(n_teacher_layers, n_student_layers)``,
finds the best one-to-one mapping.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def match_layers(
    cka_scores: torch.Tensor | None = None,
    n_teacher_layers: int | None = None,
    n_student_layers: int | None = None,
    strategy: str = "greedy",
) -> dict[int, int]:
    """Find teacher → student layer correspondence.

    Parameters
    ----------
    cka_scores : (n_T, n_S) float tensor
        CKA similarity matrix.  Higher = more similar.
    n_teacher_layers, n_student_layers : int
        Used only with ``strategy="linear"`` (uniform spacing).
    strategy : str
        ``"greedy"`` — iteratively pick the (t, s) pair with highest CKA,
        then mask out that student layer.
        ``"linear"`` — uniformly space student layers across teacher layers.

    Returns
    -------
    dict[int, int]
        ``{teacher_layer: student_layer}``.
    """
    if strategy == "linear":
        assert n_teacher_layers is not None and n_student_layers is not None
        mapping: dict[int, int] = {}
        for t in range(n_teacher_layers):
            s = round(t * (n_student_layers - 1) / max(n_teacher_layers - 1, 1))
            mapping[t] = s
        logger.info("Linear layer mapping: %d → %d layers", n_teacher_layers, n_student_layers)
        return mapping

    assert cka_scores is not None, "cka_scores required for greedy strategy"
    n_T, n_S = cka_scores.shape
    scores = cka_scores.float()
    mapping = {}

    # Many-to-one: each teacher layer picks its highest-CKA student layer
    for t in range(n_T):
        s = int(scores[t].argmax().item())
        mapping[t] = s

    logger.info(
        "Greedy layer mapping: %d matched pairs (T=%d, S=%d)",
        len(mapping), n_T, n_S,
    )
    return mapping
