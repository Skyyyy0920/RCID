"""Read operation: extract residual stream contrastive differences.

Hooks into transformer layers to capture the residual stream at specified
layers for both clean and corrupt inputs, then computes the difference
``d = h_clean - h_corrupt``.

This is the *Read* operation from the RCID framework — it does NOT alter
model behaviour (cf. ``intervention.py`` for the *Write* operation).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from rcid.models.adapter import ModelAdapter

logger = logging.getLogger(__name__)


def _hook_residuals(
    model: nn.Module,
    adapter: ModelAdapter,
    input_ids: torch.Tensor,       # (batch, seq)
    layers: list[int],
    batch_size: int = 8,
) -> dict[int, torch.Tensor]:
    """Forward-pass *model* and capture residual stream at *layers*.

    Returns
    -------
    dict
        ``{layer_idx: tensor}`` where tensor is ``(N, seq, d_model)`` on CPU.
    """
    cache: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    N = input_ids.shape[0]
    batch_size = max(1, batch_size)  # guard against batch_size=0

    if N == 0:
        # No samples — return empty tensors with correct shape
        logger.warning("_hook_residuals called with 0 samples; returning empty cache.")
        return {l: torch.empty(0) for l in layers}

    for start in range(0, N, batch_size):
        ids = input_ids[start : start + batch_size]       # (B, seq)
        handles: list[torch.utils.hooks.RemovableHook] = []
        batch_cache: dict[int, torch.Tensor] = {}

        for layer_idx in layers:
            hook_point = adapter.get_residual_hook_point(model, layer_idx)

            def _make_hook(idx: int):  # noqa: E301 — closure factory
                def _hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
                    batch_cache[idx] = adapter.parse_layer_output(out).detach().cpu()
                return _hook

            handles.append(hook_point.register_forward_hook(_make_hook(layer_idx)))

        try:
            with torch.no_grad():
                model(ids)
        finally:
            for h in handles:
                h.remove()

        for layer_idx in layers:
            cache[layer_idx].append(batch_cache[layer_idx])   # (B, seq, d)

    # Concatenate across batches → (N, seq, d_model)
    return {l: torch.cat(chunks, dim=0) for l, chunks in cache.items()}


def extract_contrastive_differences(
    model: nn.Module,
    adapter: ModelAdapter,
    clean_ids: torch.Tensor,        # (N, seq)
    corrupt_ids: torch.Tensor,      # (N, seq)
    layers: list[int],
    batch_size: int = 8,
    pool_seq: bool = False,
) -> dict[int, torch.Tensor]:
    """Compute contrastive differences ``d = h_clean - h_corrupt`` per layer.

    Parameters
    ----------
    model : nn.Module
        Transformer model (teacher or student).
    adapter : ModelAdapter
        Architecture adapter.
    clean_ids, corrupt_ids : (N, seq) long tensors
        Tokenised clean and corrupt inputs.
    layers : list[int]
        Which layers to extract from.
    batch_size : int
        Mini-batch size for forward passes (memory management).
    pool_seq : bool
        If True, mean-pool over the sequence dimension → ``(N, d_model)``.
        If False, return full ``(N, seq, d_model)``.

    Returns
    -------
    dict
        ``{layer_idx: difference_tensor}`` on CPU.
    """
    model.eval()
    clean_cache = _hook_residuals(model, adapter, clean_ids, layers, batch_size)
    corrupt_cache = _hook_residuals(model, adapter, corrupt_ids, layers, batch_size)

    diffs: dict[int, torch.Tensor] = {}
    for layer_idx in layers:
        d = clean_cache[layer_idx] - corrupt_cache[layer_idx]  # (N, seq, d)
        if pool_seq:
            d = d.mean(dim=1)  # (N, d)
        diffs[layer_idx] = d

    return diffs
