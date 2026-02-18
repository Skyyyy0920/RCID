"""Centered Kernel Alignment (CKA) for comparing representations."""

from __future__ import annotations

import torch


def linear_cka(
    X: torch.Tensor,  # (N, d_X)
    Y: torch.Tensor,  # (N, d_Y)
) -> float:
    """Compute linear CKA between two representation matrices.

    CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    """
    assert X.shape[0] == Y.shape[0], (
        f"Sample count mismatch: {X.shape[0]} vs {Y.shape[0]}"
    )
    eps = 1e-10

    # Center representations
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    YtX = Y.t() @ X       # (d_Y, d_X)
    XtX = X.t() @ X       # (d_X, d_X)
    YtY = Y.t() @ Y       # (d_Y, d_Y)

    numerator = (YtX * YtX).sum()  # ||Y^T X||_F^2
    denom_x = (XtX * XtX).sum()   # ||X^T X||_F^2
    denom_y = (YtY * YtY).sum()   # ||Y^T Y||_F^2

    denominator = torch.sqrt(denom_x * denom_y).clamp(min=eps)
    return (numerator / denominator).item()


def _flatten_to_2d(x: torch.Tensor) -> torch.Tensor:
    """Reshape (batch, seq, d) → (batch*seq, d); leave 2D unchanged."""
    if x.dim() == 3:
        return x.reshape(-1, x.shape[-1])
    return x


def cka_matrix(
    teacher_reps: dict[int, torch.Tensor],  # {layer: (N, d_T) or (B, S, d_T)}
    student_reps: dict[int, torch.Tensor],  # {layer: (N, d_S) or (B, S, d_S)}
) -> torch.Tensor:
    """Compute CKA between all pairs of teacher and student layers.

    Accepts both 2D (N, d) and 3D (batch, seq, d) tensors; 3D inputs
    are flattened to (batch*seq, d) before computing CKA.

    Returns:
        Matrix of shape (n_teacher_layers, n_student_layers).
    """
    t_layers = sorted(teacher_reps.keys())
    s_layers = sorted(student_reps.keys())

    matrix = torch.zeros(len(t_layers), len(s_layers))
    for i, tl in enumerate(t_layers):
        t_flat = _flatten_to_2d(teacher_reps[tl])
        for j, sl in enumerate(s_layers):
            s_flat = _flatten_to_2d(student_reps[sl])
            matrix[i, j] = linear_cka(t_flat, s_flat)

    return matrix
