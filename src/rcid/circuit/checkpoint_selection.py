"""Causal checkpoint selection with diversity constraints."""

from __future__ import annotations

import torch

from rcid.circuit.contrastive import ContrastiveDataset


def select_checkpoints(
    contrastive_diffs: dict[int, torch.Tensor],  # {layer: (batch, seq_len, d_model)}
    dataset: ContrastiveDataset,
    top_k: int = 20,
    diversity_ratio: float = 0.5,
    min_layer_fraction: float = 1 / 3,
) -> list[tuple[int, int]]:
    """Select top-k (layer, token_pos) checkpoints from contrastive differences.

    Ensures diversity: both modified and unmodified positions are represented.

    Args:
        contrastive_diffs: Per-layer contrastive difference tensors.
        dataset: The contrastive dataset (provides is_modified and key_positions).
        top_k: Total number of checkpoints to select.
        diversity_ratio: Fraction of checkpoints from unmodified positions.
        min_layer_fraction: Minimum layer fraction for modified positions
            (bottom layers have trivial embedding differences at modified pos).

    Returns:
        List of (layer_index, token_position) tuples, sorted by descending norm.
    """
    layers = sorted(contrastive_diffs.keys())
    n_layers = max(layers) + 1
    min_layer = int(n_layers * min_layer_fraction)

    # Build per-position "is_modified" mask across the sequence
    seq_len = dataset.seq_len
    is_pos_modified = torch.zeros(seq_len, dtype=torch.bool)
    for pos_name, is_mod in dataset.is_modified.items():
        if is_mod and pos_name in dataset.key_positions:
            pos_tensor = dataset.key_positions[pos_name]  # (N,)
            unique_pos = pos_tensor.unique()
            for p in unique_pos:
                if 0 <= p < seq_len:
                    is_pos_modified[p] = True

    # Compute mean norm per (layer, pos) across batch
    candidates_modified: list[tuple[float, int, int]] = []
    candidates_unmodified: list[tuple[float, int, int]] = []

    for layer_idx in layers:
        diff = contrastive_diffs[layer_idx]  # (batch, seq_len, d_model)
        norms = diff.norm(dim=-1).mean(dim=0)  # (seq_len,)

        for pos in range(seq_len):
            norm_val = norms[pos].item()
            if norm_val < 1e-10:
                continue
            if is_pos_modified[pos]:
                if layer_idx >= min_layer:
                    candidates_modified.append((norm_val, layer_idx, pos))
            else:
                candidates_unmodified.append((norm_val, layer_idx, pos))

    # Sort descending by norm
    candidates_modified.sort(reverse=True)
    candidates_unmodified.sort(reverse=True)

    # Allocate budget
    n_unmodified = max(1, int(top_k * diversity_ratio))
    n_modified = top_k - n_unmodified

    # Adjust if one pool is too small
    if len(candidates_unmodified) < n_unmodified:
        n_unmodified = len(candidates_unmodified)
        n_modified = top_k - n_unmodified
    if len(candidates_modified) < n_modified:
        n_modified = len(candidates_modified)
        n_unmodified = min(top_k - n_modified, len(candidates_unmodified))

    selected: list[tuple[int, int]] = []
    for _, layer_idx, pos in candidates_modified[:n_modified]:
        selected.append((layer_idx, pos))
    for _, layer_idx, pos in candidates_unmodified[:n_unmodified]:
        selected.append((layer_idx, pos))

    # Sort final by descending mean norm for consistent ordering
    all_norms: dict[tuple[int, int], float] = {}
    for norm_val, layer_idx, pos in candidates_modified + candidates_unmodified:
        all_norms[(layer_idx, pos)] = norm_val
    selected.sort(key=lambda x: all_norms.get(x, 0.0), reverse=True)

    return selected
