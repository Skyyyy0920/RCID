"""Write operation: activation patching and causal effect computation."""

from __future__ import annotations

import torch
import torch.nn as nn

from rcid.models.adapter import ModelAdapter


def patch_and_run(
    model: nn.Module,
    adapter: ModelAdapter,
    clean_input: torch.Tensor,       # (batch, seq_len)
    patch_value: torch.Tensor,       # (batch, d_model)
    layer: int,
    token_pos: int | torch.Tensor,   # scalar or (batch,)
) -> torch.Tensor:                   # (batch, vocab_size)
    """Pearl do-operator: replace residual at (layer, token_pos) and forward.

    Args:
        model: The transformer model.
        adapter: Model adapter.
        clean_input: Input token ids.
        patch_value: Value to patch in at the specified position.
        layer: Layer index to patch.
        token_pos: Token position(s) to patch (scalar or per-sample).

    Returns:
        Logits from the patched forward pass.
    """
    hook_point = adapter.get_residual_hook_point(model, layer)

    def _patch_hook(
        module: nn.Module, input: tuple, output: tuple
    ) -> tuple:
        h = adapter.parse_layer_output(output).clone()  # (batch, seq, d_model)
        if isinstance(token_pos, int):
            h[:, token_pos, :] = patch_value
        else:
            # Per-sample positions: token_pos is (batch,)
            batch_idx = torch.arange(h.shape[0], device=h.device)
            h[batch_idx, token_pos, :] = patch_value
        return (h,) + output[1:]

    handle = hook_point.register_forward_hook(_patch_hook)
    try:
        with torch.no_grad():
            patched_logits = model(clean_input).logits  # (batch, seq, vocab)
    finally:
        handle.remove()

    return patched_logits


def _gather_logit_diff(
    logits: torch.Tensor,               # (batch, seq, vocab)
    answer_pos: int | torch.Tensor,     # scalar or (batch,)
    correct_token_id: torch.Tensor,     # (batch,)
    wrong_token_id: torch.Tensor,       # (batch,)
) -> torch.Tensor:                      # (batch,)
    """Compute logit_diff = logit(correct) - logit(wrong) at answer_pos."""
    batch_size = logits.shape[0]
    if isinstance(answer_pos, int):
        logits_at = logits[:, answer_pos, :]  # (batch, vocab)
    else:
        batch_idx = torch.arange(batch_size, device=logits.device)
        logits_at = logits[batch_idx, answer_pos, :]  # (batch, vocab)

    correct_logit = logits_at.gather(
        1, correct_token_id.unsqueeze(1)
    ).squeeze(1)  # (batch,)
    wrong_logit = logits_at.gather(
        1, wrong_token_id.unsqueeze(1)
    ).squeeze(1)  # (batch,)

    return correct_logit - wrong_logit  # (batch,)


def compute_causal_effect(
    model: nn.Module,
    adapter: ModelAdapter,
    clean_input: torch.Tensor,          # (batch, seq_len)
    corrupt_input: torch.Tensor,        # (batch, seq_len)
    layer: int,
    token_pos: int | torch.Tensor,
    answer_pos: int | torch.Tensor,
    correct_token_id: torch.Tensor,     # (batch,)
    wrong_token_id: torch.Tensor,       # (batch,)
) -> torch.Tensor:                      # (batch,)
    """Compute causal effect: delta = logit_diff(orig) - logit_diff(patched).

    Patches the clean run's residual at (layer, token_pos) with the corrupt
    run's residual value at the same position.
    """
    model.eval()

    # Original logit diff (clean forward, no patching)
    with torch.no_grad():
        orig_logits = model(clean_input).logits  # (batch, seq, vocab)
    orig_diff = _gather_logit_diff(
        orig_logits, answer_pos, correct_token_id, wrong_token_id
    )

    # Get corrupt residual value at (layer, token_pos)
    corrupt_cache: dict[str, torch.Tensor] = {}
    hook_point = adapter.get_residual_hook_point(model, layer)

    def _capture_hook(
        module: nn.Module, input: tuple, output: tuple
    ) -> None:
        h = adapter.parse_layer_output(output)  # (batch, seq, d_model)
        if isinstance(token_pos, int):
            corrupt_cache["h"] = h[:, token_pos, :].detach().clone()
        else:
            batch_idx = torch.arange(h.shape[0], device=h.device)
            corrupt_cache["h"] = h[batch_idx, token_pos, :].detach().clone()

    handle = hook_point.register_forward_hook(_capture_hook)
    try:
        with torch.no_grad():
            model(corrupt_input)
    finally:
        handle.remove()

    # Patched forward
    patched_logits = patch_and_run(
        model, adapter, clean_input,
        corrupt_cache["h"], layer, token_pos,
    )
    patched_diff = _gather_logit_diff(
        patched_logits, answer_pos, correct_token_id, wrong_token_id
    )

    return orig_diff - patched_diff  # (batch,)
