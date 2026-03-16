"""Select causally important checkpoints (layer, token_pos) from teacher.

Ranks all (layer, position) combinations by the L2 norm of the
contrastive difference ``||d_{l,t}|| = ||h_clean - h_corrupt||`` and
selects the top-k, ensuring diversity between modified and unmodified
positions.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def select_checkpoints(
    teacher_diffs: dict[int, torch.Tensor],
    dataset: Any,
    top_k: int = 10,
    diversity_ratio: float = 0.5,
) -> list[tuple[int, int]]:
    """Select top-k causal checkpoints from teacher residual-stream diffs.

    Parameters
    ----------
    teacher_diffs : dict
        ``{layer: (N, seq, d_model)}`` — contrastive differences from teacher.
    dataset : Any
        Must have ``.seq_len``, ``.is_modified`` dict, and ``.key_positions``
        dict.  The key_positions values are ``(N,)`` tensors of token
        indices for named positions.
    top_k : int
        Total number of checkpoints to select.
    diversity_ratio : float
        Fraction of checkpoints that must come from *modified* positions
        vs *unmodified* positions.  ``0.5`` means an even split.

    Returns
    -------
    list[tuple[int, int]]
        Selected ``(layer_idx, token_position)`` pairs sorted by norm
        (descending).
    """
    seq_len = dataset.seq_len

    # ── Collect per-(layer, pos) norms averaged over samples ────────
    scored: list[tuple[float, int, int, bool]] = []  # (norm, layer, pos, is_modified)

    # Determine which token positions are "modified" (differ between
    # clean and corrupt) vs "unmodified" (same token in both).
    modified_positions: set[int] = set()
    for name, is_mod in dataset.is_modified.items():
        if is_mod and name in dataset.key_positions:
            # key_positions[name] is (N,) — take mode as representative
            pos_vals = dataset.key_positions[name]
            if pos_vals.dim() > 0:
                modified_positions.add(int(pos_vals[0].item()))

    for layer_idx, diffs in teacher_diffs.items():
        # diffs: (N, seq, d_model)
        norms = diffs.norm(dim=-1).mean(dim=0)  # (seq,) — avg over samples
        for pos in range(min(norms.shape[0], seq_len)):
            is_mod = pos in modified_positions
            scored.append((norms[pos].item(), layer_idx, pos, is_mod))

    # ── Sort by norm descending ────────────────────────────────────
    scored.sort(key=lambda x: x[0], reverse=True)

    # ── Diversity-constrained selection ────────────────────────────
    n_modified = max(1, int(top_k * diversity_ratio))
    n_unmodified = top_k - n_modified

    modified_cps: list[tuple[int, int]] = []
    unmodified_cps: list[tuple[int, int]] = []

    for norm_val, layer, pos, is_mod in scored:
        if is_mod and len(modified_cps) < n_modified:
            modified_cps.append((layer, pos))
        elif not is_mod and len(unmodified_cps) < n_unmodified:
            unmodified_cps.append((layer, pos))
        if len(modified_cps) >= n_modified and len(unmodified_cps) >= n_unmodified:
            break

    # If not enough of one type, fill from the other
    result = modified_cps + unmodified_cps
    if len(result) < top_k:
        seen = set(result)
        for norm_val, layer, pos, is_mod in scored:
            if (layer, pos) not in seen:
                result.append((layer, pos))
                seen.add((layer, pos))
            if len(result) >= top_k:
                break

    logger.info(
        "Selected %d checkpoints (%d modified, %d unmodified)",
        len(result), len(modified_cps), len(unmodified_cps),
    )
    return result[:top_k]
