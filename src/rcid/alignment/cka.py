"""Centered Kernel Alignment (CKA) for comparing layer representations.

CKA measures similarity between two representation matrices using the
Hilbert-Schmidt Independence Criterion (HSIC) with a linear kernel.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def _linear_hsic(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute linear HSIC between centered matrices X and Y.

    Both inputs: ``(N, d)``.
    """
    N = X.shape[0]
    # Center
    X_c = X - X.mean(dim=0, keepdim=True)
    Y_c = Y - Y.mean(dim=0, keepdim=True)
    # HSIC = (1/N^2) * tr(X_c @ X_c^T @ Y_c @ Y_c^T)
    #      = (1/N^2) * ||X_c^T @ Y_c||_F^2
    XtY = X_c.T @ Y_c  # (d_X, d_Y)
    return (XtY * XtY).sum() / (N * N)


def linear_cka(
    X: torch.Tensor,   # (N, d_X)
    Y: torch.Tensor,   # (N, d_Y)
) -> float:
    """Compute linear CKA between two representation matrices.

    Returns a similarity score in ``[0, 1]``.
    Returns ``0.0`` when fewer than 2 samples are provided (undefined).
    """
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
    if Y.dim() == 3:
        Y = Y.reshape(-1, Y.shape[-1])

    X = X.float()
    Y = Y.float()

    if X.shape[0] < 2:
        # CKA is undefined with fewer than 2 samples
        logger.warning("linear_cka called with N=%d samples; returning 0.0", X.shape[0])
        return 0.0

    hsic_xy = _linear_hsic(X, Y)
    hsic_xx = _linear_hsic(X, X)
    hsic_yy = _linear_hsic(Y, Y)

    denom = (hsic_xx * hsic_yy).sqrt().clamp(min=1e-12)
    return (hsic_xy / denom).item()


def cka_matrix(
    teacher_reps: dict[int, torch.Tensor],
    student_reps: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Compute the full CKA similarity matrix between teacher and student layers.

    Parameters
    ----------
    teacher_reps : ``{t_layer: (N, d_T)}`` — teacher representations.
    student_reps : ``{s_layer: (N, d_S)}`` — student representations.

    Returns
    -------
    torch.Tensor
        ``(n_teacher_layers, n_student_layers)`` CKA similarity matrix.
    """
    t_layers = sorted(teacher_reps.keys())
    s_layers = sorted(student_reps.keys())

    matrix = torch.zeros(len(t_layers), len(s_layers))
    for i, tl in enumerate(t_layers):
        for j, sl in enumerate(s_layers):
            matrix[i, j] = linear_cka(teacher_reps[tl], student_reps[sl])

    logger.info(
        "CKA matrix: %d x %d (max=%.3f, mean=%.3f)",
        len(t_layers), len(s_layers), matrix.max().item(), matrix.mean().item(),
    )
    return matrix
