"""Read operation: extract residual stream values without modifying the model."""

from __future__ import annotations

import torch
import torch.nn as nn

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
                module: nn.Module, input: tuple, output: tuple
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
    clean_ids: torch.Tensor,           # (batch, seq_len)
    corrupt_ids: torch.Tensor,         # (batch, seq_len)
    layers: list[int] | None = None,
) -> dict[int, torch.Tensor]:          # {layer: (batch, seq_len, d_model)}
    """Compute d = h_clean - h_corrupt at each layer (Read operation).

    Does not modify the model. Runs two forward passes.
    """
    h_clean = extract_residual_at_layers(model, adapter, clean_ids, layers)
    h_corrupt = extract_residual_at_layers(model, adapter, corrupt_ids, layers)

    diffs: dict[int, torch.Tensor] = {}
    for layer_idx in h_clean:
        diffs[layer_idx] = h_clean[layer_idx] - h_corrupt[layer_idx]

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
                module: nn.Module, input: tuple, output: tuple
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
