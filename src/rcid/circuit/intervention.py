"""Write operation: activation patching (causal intervention).

Replaces the residual stream at a specific (layer, token_position) with a
*patched* value and continues the forward pass.  Used for causal
consistency evaluation — NOT for training.

This is the *Write* operation from the RCID framework (Pearl do-operator
in the transformer residual stream).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from rcid.models.adapter import ModelAdapter


def patch_and_run(
    model: nn.Module,
    adapter: ModelAdapter,
    clean_ids: torch.Tensor,         # (1, seq)
    patch_value: torch.Tensor,       # (1, d_model)
    layer: int,
    token_pos: int,
) -> torch.Tensor:
    """Run *model* on *clean_ids*, but replace the residual at
    ``(layer, token_pos)`` with *patch_value*.

    Returns
    -------
    torch.Tensor
        Logits of shape ``(1, seq, vocab)``.
    """
    hook_point = adapter.get_residual_hook_point(model, layer)

    def _patch_hook(
        _mod: nn.Module, _inp: Any, output: Any,
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        h = adapter.parse_layer_output(output)     # (1, seq, d)
        h = h.clone()
        h[:, token_pos, :] = patch_value           # overwrite
        # Re-pack into the original output format
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    handle = hook_point.register_forward_hook(_patch_hook)
    model.eval()
    try:
        with torch.no_grad():
            logits = model(clean_ids).logits        # (1, seq, vocab)
    finally:
        handle.remove()

    return logits


def compute_causal_effect(
    model: nn.Module,
    adapter: ModelAdapter,
    clean_ids: torch.Tensor,         # (1, seq)
    corrupt_ids: torch.Tensor,       # (1, seq)
    layer: int,
    token_pos: int,
    answer_pos: int,
    correct_id: int,
    wrong_id: int,
) -> float:
    """Compute the causal effect of patching at ``(layer, token_pos)``.

    Steps:
    1. Run model on clean → compute ``logit_diff_original``.
    2. Extract the corrupt residual at ``(layer, token_pos)``.
    3. Patch clean with that corrupt value → compute ``logit_diff_patched``.
    4. ``Δ = logit_diff_original - logit_diff_patched``.

    Returns
    -------
    float
        Causal effect (positive = this position is causally important).
    """
    device = clean_ids.device
    model.eval()

    # Step 1: original logit diff
    with torch.no_grad():
        orig_logits = model(clean_ids).logits       # (1, seq, V)
    orig_ld = (
        orig_logits[0, answer_pos, correct_id]
        - orig_logits[0, answer_pos, wrong_id]
    ).item()

    # Step 2: extract corrupt residual at (layer, token_pos)
    hook_point = adapter.get_residual_hook_point(model, layer)
    corrupt_val: list[torch.Tensor] = []

    def _read_hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
        h = adapter.parse_layer_output(out)          # (1, seq, d)
        corrupt_val.append(h[:, token_pos, :].detach())

    handle = hook_point.register_forward_hook(_read_hook)
    try:
        with torch.no_grad():
            model(corrupt_ids)
    finally:
        handle.remove()

    # Step 3: patch clean with corrupt value → forward
    patched_logits = patch_and_run(
        model, adapter, clean_ids,
        patch_value=corrupt_val[0],
        layer=layer, token_pos=token_pos,
    )  # (1, seq, V)
    patched_ld = (
        patched_logits[0, answer_pos, correct_id]
        - patched_logits[0, answer_pos, wrong_id]
    ).item()

    # Step 4: causal effect
    return orig_ld - patched_ld
