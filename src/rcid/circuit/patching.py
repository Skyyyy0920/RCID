"""Read operation: extract residual stream values without modifying the model."""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from rcid.models.adapter import ModelAdapter


def extract_residual_at_layers(
    model: nn.Module,
    adapter: ModelAdapter,
    input_ids: torch.Tensor,          # (batch, seq_len)
    layers: list[int] | None = None,
) -> dict[int, torch.Tensor]:         # {layer: (batch, seq_len, d_model)}
    """Hook into specified layers and capture residual stream activations.

    Args:
        model: The transformer model (teacher or student).
        adapter: Model adapter for hook registration.
        input_ids: Token ids to run forward on.
        layers: Layer indices to capture. None = all layers.

    Returns:
        Dict mapping layer index to captured activation tensor.
    """
    if layers is None:
        layers = list(range(adapter.get_num_layers(model)))

    cache: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHook] = []

    for layer_idx in layers:
        hook_point = adapter.get_residual_hook_point(model, layer_idx)

        def _make_hook(idx: int):  # noqa: E301
            def _hook(
                module: nn.Module, input: tuple,
                output: torch.Tensor | tuple,
            ) -> None:
                h = adapter.parse_layer_output(output)  # (batch, seq, d_model)
                cache[idx] = h.detach().clone()
            return _hook

        handle = hook_point.register_forward_hook(_make_hook(layer_idx))
        handles.append(handle)

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    return cache


def extract_contrastive_differences(
    model: nn.Module,
    adapter: ModelAdapter,
    clean_ids: torch.Tensor,           # (N, seq_len)
    corrupt_ids: torch.Tensor,         # (N, seq_len)
    layers: list[int] | None = None,
    batch_size: int = 4,
    pool_seq: bool = False,
) -> dict[int, torch.Tensor]:
    """Compute d = h_clean - h_corrupt at each layer (Read operation).

    Processes data in mini-batches to avoid OOM on large datasets.
    Intermediate per-batch results are moved to CPU; the final concatenated
    tensors live on CPU.

    Args:
        model: Transformer model (teacher or student).
        adapter: Model adapter.
        clean_ids: Clean input token ids, shape (N, seq_len).
        corrupt_ids: Corrupt input token ids, shape (N, seq_len).
        layers: Layer indices to capture. None = all.
        batch_size: Number of samples per forward pass.
        pool_seq: If True, mean-pool over the sequence dimension so that
            each layer's output is (N, d_model) instead of (N, seq_len, d_model).
            Dramatically reduces CPU memory for CKA / Procrustes alignment.

    Returns:
        Dict mapping layer index to contrastive difference tensor on CPU.
        Shape: (N, d_model) if pool_seq else (N, seq_len, d_model).
    """
    N = clean_ids.shape[0]
    if layers is None:
        layers = list(range(adapter.get_num_layers(model)))

    # Small inputs: process in one shot (fast path for tests / toy data)
    if N <= batch_size:
        h_clean = extract_residual_at_layers(model, adapter, clean_ids, layers)
        h_corrupt = extract_residual_at_layers(model, adapter, corrupt_ids, layers)
        diffs: dict[int, torch.Tensor] = {}
        for layer_idx in h_clean:
            d = h_clean[layer_idx] - h_corrupt[layer_idx]  # (N, seq, d)
            diffs[layer_idx] = d.mean(dim=1) if pool_seq else d
        return diffs

    # Large inputs: process in batches, accumulate on CPU
    all_diffs: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    n_batches = (N + batch_size - 1) // batch_size

    for start in tqdm(range(0, N, batch_size), total=n_batches,
                      desc="Contrastive diffs", leave=False):
        end = min(start + batch_size, N)
        c_batch = clean_ids[start:end]    # (B, seq_len)
        x_batch = corrupt_ids[start:end]  # (B, seq_len)

        h_clean = extract_residual_at_layers(model, adapter, c_batch, layers)
        h_corrupt = extract_residual_at_layers(model, adapter, x_batch, layers)

        for l in layers:
            d = h_clean[l] - h_corrupt[l]          # (B, seq_len, d_model)
            if pool_seq:
                d = d.mean(dim=1)                   # (B, d_model)
            all_diffs[l].append(d.cpu())

        # Free GPU cache between batches
        del h_clean, h_corrupt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate all batches on CPU
    diffs = {}
    for l in layers:
        diffs[l] = torch.cat(all_diffs[l], dim=0)

    return diffs


def extract_residual_at_layers_with_grad(
    model: nn.Module,
    adapter: ModelAdapter,
    input_ids: torch.Tensor,           # (batch, seq_len)
    layers: list[int],
) -> dict[int, torch.Tensor]:          # {layer: (batch, seq_len, d_model)}
    """Like extract_residual_at_layers but retains gradients (for student)."""
    cache: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHook] = []

    for layer_idx in layers:
        hook_point = adapter.get_residual_hook_point(model, layer_idx)

        def _make_hook(idx: int):  # noqa: E301
            def _hook(
                module: nn.Module, input: tuple,
                output: torch.Tensor | tuple,
            ) -> None:
                h = adapter.parse_layer_output(output)  # (batch, seq, d_model)
                cache[idx] = h  # keep grad graph
            return _hook

        handle = hook_point.register_forward_hook(_make_hook(layer_idx))
        handles.append(handle)

    try:
        model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    return cache
