"""RCID loss: match teacher and student contrastive differences at causal checkpoints.

The RCID regulariser compares the *direction* of contrastive differences
(``d = h_clean - h_corrupt``) between teacher and student at causally
important (layer, position) checkpoints, using a frozen Procrustes
alignment matrix to bridge different hidden dimensions.

Loss per checkpoint::

    L = || normalize(W · d_S) - normalize(d_T) ||²

The overall RCID loss is the mean over all checkpoints in ``C``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from rcid.models.adapter import ModelAdapter

logger = logging.getLogger(__name__)


class RCIDLoss(nn.Module):
    """Contrastive difference matching at causal checkpoints.

    Parameters
    ----------
    checkpoints : list[tuple[int, int]]
        ``(teacher_layer, token_position)`` pairs selected by checkpoint search.
    layer_mapping : dict[int, int]
        ``{teacher_layer: student_layer}`` correspondence.
    W_matrices : dict[int, torch.Tensor]
        ``{teacher_layer: W}`` where ``W`` is ``(d_T, d_S)`` Procrustes matrix.
        Registered as frozen buffers (no gradient).
    """

    def __init__(
        self,
        checkpoints: list[tuple[int, int]],
        layer_mapping: dict[int, int],
        W_matrices: dict[int, torch.Tensor],
    ) -> None:
        super().__init__()
        self.checkpoints = checkpoints
        self.layer_mapping = layer_mapping
        for t_layer, W in W_matrices.items():
            self.register_buffer(f"W_{t_layer}", W)  # frozen

    def forward(
        self,
        teacher_diffs: dict[int, torch.Tensor],
        # {t_layer: (batch, seq, d_T)} — detached teacher contrastive diffs
        student_clean_residuals: dict[int, torch.Tensor],
        # {s_layer: (batch, seq, d_S)} — student clean (has grad)
        student_corrupt_residuals: dict[int, torch.Tensor],
        # {s_layer: (batch, seq, d_S)} — student corrupt (has grad)
    ) -> torch.Tensor:
        """Compute RCID loss over all checkpoints.

        The student contrastive difference is computed inline:
        ``d_S = student_clean - student_corrupt`` (gradient preserved).
        """
        eps = 1e-8
        device = next(iter(student_clean_residuals.values())).device
        total = torch.tensor(0.0, device=device)
        count = 0

        for t_layer, t_pos in self.checkpoints:
            if t_layer not in self.layer_mapping:
                continue
            s_layer = self.layer_mapping[t_layer]

            if t_layer not in teacher_diffs:
                continue
            if s_layer not in student_clean_residuals:
                continue
            if s_layer not in student_corrupt_residuals:
                continue

            W = getattr(self, f"W_{t_layer}")  # (d_T, d_S)

            # Teacher diff at this checkpoint (detached)
            d_T = teacher_diffs[t_layer][:, t_pos, :]  # (batch, d_T)

            # Student diff at corresponding position (has grad)
            d_S = (
                student_clean_residuals[s_layer][:, t_pos, :]
                - student_corrupt_residuals[s_layer][:, t_pos, :]
            )  # (batch, d_S)

            # Align student → teacher space
            aligned = d_S @ W.t()  # (batch, d_T)

            # Normalize both
            aligned_n = aligned / aligned.norm(dim=-1, keepdim=True).clamp(min=eps)
            d_T_n = d_T / d_T.norm(dim=-1, keepdim=True).clamp(min=eps)

            # L2 distance between normalised vectors
            total = total + (aligned_n - d_T_n).pow(2).sum(dim=-1).mean()
            count += 1

        if count > 0:
            total = total / count
        assert total.isfinite(), f"RCID loss is {total.item()}"
        return total


def extract_residuals_with_grad(
    model: nn.Module,
    adapter: ModelAdapter,
    input_ids: torch.Tensor,    # (B, seq)
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Forward-pass *model* and capture residuals at *layers* WITH gradients.

    Unlike ``patching.extract_contrastive_differences``, this version
    preserves the computation graph so that the student's loss can
    backpropagate through the captured residuals.

    Returns
    -------
    dict
        ``{layer_idx: (B, seq, d_model)}`` on the same device as model.
    """
    cache: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    for layer_idx in layers:
        hook_point = adapter.get_residual_hook_point(model, layer_idx)

        def _make_hook(idx: int):
            def _hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
                cache[idx] = adapter.parse_layer_output(out)  # keep grad
            return _hook

        handles.append(hook_point.register_forward_hook(_make_hook(layer_idx)))

    try:
        model(input_ids)  # full forward — gradients flow
    finally:
        for h in handles:
            h.remove()

    return cache
