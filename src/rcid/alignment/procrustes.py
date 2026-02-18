"""Procrustes alignment for mapping student representations to teacher space."""

from __future__ import annotations

import torch


def procrustes_align(
    source: torch.Tensor,   # (N, d_S) — student dimension
    target: torch.Tensor,   # (N, d_T) — teacher dimension
) -> torch.Tensor:          # (d_T, d_S)
    """Compute optimal orthogonal mapping W* = argmin ||target - source @ W^T||_F.

    Uses SVD-based Procrustes solution on centered, normalized inputs.
    Returns W of shape (d_T, d_S) so that: source @ W^T ≈ target.
    """
    eps = 1e-8
    assert source.shape[0] == target.shape[0], (
        f"Sample count mismatch: {source.shape[0]} vs {target.shape[0]}"
    )

    # Center
    source_c = source - source.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)

    # Normalize each sample
    source_c = source_c / source_c.norm(dim=-1, keepdim=True).clamp(min=eps)
    target_c = target_c / target_c.norm(dim=-1, keepdim=True).clamp(min=eps)

    # SVD of cross-covariance M = target^T @ source
    M = target_c.t() @ source_c  # (d_T, d_S)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)

    W = U @ Vh  # (d_T, d_S)
    return W


def compute_procrustes_matrices(
    teacher_diffs: dict[int, torch.Tensor],   # {t_layer: (N, d_T)}
    student_diffs: dict[int, torch.Tensor],   # {s_layer: (N, d_S)}
    layer_mapping: dict[int, int],            # {t_layer: s_layer}
) -> dict[int, torch.Tensor]:                 # {t_layer: W (d_T, d_S)}
    """Compute per-layer Procrustes alignment matrices.

    For each teacher layer in the mapping, computes W that aligns the
    corresponding student layer's representations to the teacher's.
    """
    W_matrices: dict[int, torch.Tensor] = {}

    for t_layer, s_layer in layer_mapping.items():
        if t_layer not in teacher_diffs or s_layer not in student_diffs:
            continue
        t_data = teacher_diffs[t_layer]  # (N, d_T) or (B, S, d_T)
        s_data = student_diffs[s_layer]  # (N, d_S) or (B, S, d_S)
        # Flatten 3D → 2D if needed
        if t_data.dim() == 3:
            t_data = t_data.reshape(-1, t_data.shape[-1])
        if s_data.dim() == 3:
            s_data = s_data.reshape(-1, s_data.shape[-1])
        W_matrices[t_layer] = procrustes_align(source=s_data, target=t_data)

    return W_matrices
