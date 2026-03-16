"""Procrustes alignment: find the optimal linear map between teacher
and student representation spaces.

The Procrustes solution ``W* = V @ U^T`` minimises
``||target - source @ W^T||_F`` subject to ``W`` being orthogonal.
We relax the orthogonality constraint (standard in distillation lit)
and simply return ``W = V @ U^T`` from the SVD of ``target^T @ source``.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def procrustes_align(
    source: torch.Tensor,    # (N, d_S)
    target: torch.Tensor,    # (N, d_T)
) -> torch.Tensor:
    """Compute the Procrustes alignment matrix.

    Parameters
    ----------
    source : (N, d_S) — student representations.
    target : (N, d_T) — teacher representations.

    Returns
    -------
    W : (d_T, d_S) — projection matrix such that
        ``source @ W.T ≈ target``.
    """
    if source.dim() == 3:
        # (B, S, d) → (B*S, d)
        source = source.reshape(-1, source.shape[-1])
    if target.dim() == 3:
        target = target.reshape(-1, target.shape[-1])

    assert source.shape[0] == target.shape[0], (
        f"Sample count mismatch: {source.shape[0]} vs {target.shape[0]}"
    )

    N = source.shape[0]
    if N == 0:
        # No samples — return identity-like matrix (d_T, d_S)
        d_S, d_T = source.shape[1], target.shape[1]
        logger.warning(
            "procrustes_align called with 0 samples; returning zeros (%d, %d)", d_T, d_S,
        )
        return torch.zeros(d_T, d_S)

    # M = target^T @ source   →  (d_T, d_S)
    M = target.float().T @ source.float()
    U, _S, Vt = torch.linalg.svd(M, full_matrices=False)
    W = U @ Vt  # (d_T, d_S)
    return W


def compute_procrustes_matrices(
    teacher_diffs: dict[int, torch.Tensor],
    student_diffs: dict[int, torch.Tensor],
    layer_mapping: dict[int, int],
) -> dict[int, torch.Tensor]:
    """Compute Procrustes W for each mapped (teacher_layer, student_layer) pair.

    Parameters
    ----------
    teacher_diffs : {t_layer: (N, d_T)} — teacher representations (pooled).
    student_diffs : {s_layer: (N, d_S)} — student representations (pooled).
    layer_mapping : {t_layer: s_layer} — mapping from teacher to student layers.

    Returns
    -------
    dict
        ``{t_layer: W_tensor}`` where ``W_tensor`` is ``(d_T, d_S)``.
    """
    W_matrices: dict[int, torch.Tensor] = {}

    for t_layer, s_layer in layer_mapping.items():
        if t_layer not in teacher_diffs or s_layer not in student_diffs:
            continue
        t_data = teacher_diffs[t_layer]  # (N, d_T)
        s_data = student_diffs[s_layer]  # (N, d_S)
        W = procrustes_align(s_data, t_data)  # (d_T, d_S)
        W_matrices[t_layer] = W
        logger.debug(
            "Procrustes W for t_layer=%d → s_layer=%d: shape=%s",
            t_layer, s_layer, W.shape,
        )

    logger.info("Computed %d Procrustes matrices", len(W_matrices))
    return W_matrices
