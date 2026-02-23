"""Unified quality validation pipeline for auto-generated contrastive pairs.

Ensures that (clean, corrupt) pairs satisfy all requirements for RCID:
  1. Teacher output actually changes.
  2. Edit distance is small (minimal perturbation).
  3. Sequence lengths are compatible (alignable).
  4. Teacher residual stream shows non-trivial contrastive differences.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from rcid.models.adapter import ModelAdapter

logger = logging.getLogger(__name__)


class ContrastivePairValidator:
    """Validate auto-generated contrastive pairs for RCID quality.

    Parameters
    ----------
    teacher : nn.Module
        Teacher model (kept in eval mode, no grad).
    adapter : ModelAdapter
        Adapter for the teacher model.
    tokenizer : Any
        HuggingFace tokenizer for the teacher.
    causal_effect_threshold : float
        Minimum L2 norm of contrastive difference (averaged over sampled
        layers) for the pair to be considered causally relevant.
    sample_layers : list[int] | None
        Layers to probe for contrastive differences. If None, probes the
        layer at 25 %, 50 %, and 75 % depth.
    max_edit_distance : int
        Maximum number of differing tokens allowed.
    max_length_diff : int
        Maximum difference in token count between clean and corrupt.
    device : str | torch.device
        Device for teacher inference.
    """

    def __init__(
        self,
        teacher: nn.Module,
        adapter: ModelAdapter,
        tokenizer: Any,
        causal_effect_threshold: float = 0.1,
        sample_layers: list[int] | None = None,
        max_edit_distance: int = 5,
        max_length_diff: int = 2,
        device: str | torch.device = "cpu",
    ) -> None:
        self.teacher = teacher
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.threshold = causal_effect_threshold
        self.max_edit_distance = max_edit_distance
        self.max_length_diff = max_length_diff
        self.device = device

        # Default: probe at 25 %, 50 %, 75 % depth
        n_layers = adapter.get_num_layers(teacher)
        self.sample_layers = sample_layers or [
            n_layers // 4,
            n_layers // 2,
            3 * n_layers // 4,
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, clean_text: str, corrupt_text: str) -> dict[str, bool]:
        """Run all quality checks on a single (clean, corrupt) pair.

        Returns a dict mapping check name → pass/fail boolean.
        """
        clean_ids = self._tokenize(clean_text)   # (1, seq_len_c)
        corrupt_ids = self._tokenize(corrupt_text)  # (1, seq_len_x)

        output_changed = self._teacher_output_changed(clean_ids, corrupt_ids)
        len_c, len_x = clean_ids.shape[1], corrupt_ids.shape[1]
        length_ok = abs(len_c - len_x) <= self.max_length_diff

        # Token-level checks require equal length
        if len_c == len_x:
            modified = self.find_modified_positions(clean_ids[0], corrupt_ids[0])
            edit_ok = len(modified) <= self.max_edit_distance
            alignable = len(modified) > 0
        else:
            edit_ok = False
            alignable = False

        causal_ok = False
        if alignable:
            causal_ok = self._causal_effect_exists(clean_ids, corrupt_ids)

        return {
            "teacher_output_changed": output_changed,
            "edit_distance_ok": edit_ok,
            "length_preserved": length_ok,
            "tokens_alignable": alignable,
            "causal_effect_exists": causal_ok,
        }

    def is_valid(self, clean_text: str, corrupt_text: str) -> bool:
        """Return True only if *all* checks pass."""
        return all(self.validate(clean_text, corrupt_text).values())

    @staticmethod
    def find_modified_positions(
        clean_ids: torch.Tensor,   # (seq_len,)
        corrupt_ids: torch.Tensor,  # (seq_len,)
    ) -> list[int]:
        """Return indices where clean and corrupt tokens differ.

        Both tensors must have the same length.
        """
        assert clean_ids.shape == corrupt_ids.shape, (
            f"Length mismatch: {clean_ids.shape} vs {corrupt_ids.shape}"
        )
        diff_mask = clean_ids != corrupt_ids  # (seq_len,)
        return diff_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

    def batch_validate(
        self, pairs: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """Filter *pairs*, keeping only those that pass all checks."""
        valid: list[tuple[str, str]] = []
        for clean, corrupt in pairs:
            if self.is_valid(clean, corrupt):
                valid.append((clean, corrupt))
        logger.info(
            "batch_validate: %d / %d pairs passed", len(valid), len(pairs),
        )
        return valid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize *text* and return input_ids on self.device.  Shape: (1, seq_len)."""
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        )
        return enc.input_ids.to(self.device)  # (1, seq_len)

    def _teacher_output_changed(
        self,
        clean_ids: torch.Tensor,   # (1, seq_len)
        corrupt_ids: torch.Tensor,  # (1, seq_len)
    ) -> bool:
        """True if teacher's argmax at the last position differs."""
        self.teacher.eval()
        with torch.no_grad():
            c_pred = self.teacher(clean_ids).logits[:, -1, :].argmax(dim=-1)
            x_pred = self.teacher(corrupt_ids).logits[:, -1, :].argmax(dim=-1)
        return c_pred.item() != x_pred.item()

    def _causal_effect_exists(
        self,
        clean_ids: torch.Tensor,   # (1, seq_len)
        corrupt_ids: torch.Tensor,  # (1, seq_len)
    ) -> bool:
        """True if residual-stream contrastive diff norm exceeds threshold.

        Hooks into *sample_layers*, computes ||h_clean - h_corrupt|| averaged
        over layers, and checks against *self.threshold*.
        """
        cache_clean: dict[int, torch.Tensor] = {}
        cache_corrupt: dict[int, torch.Tensor] = {}
        handles = []

        for layer_idx in self.sample_layers:
            hook_point = self.adapter.get_residual_hook_point(
                self.teacher, layer_idx,
            )

            def _make_hook(idx: int, store: dict[int, torch.Tensor]):
                def _hook(
                    mod: nn.Module, inp: tuple, out: torch.Tensor | tuple,
                ) -> None:
                    store[idx] = self.adapter.parse_layer_output(out).detach()
                return _hook

            handles.append(
                hook_point.register_forward_hook(_make_hook(layer_idx, cache_clean))
            )

        self.teacher.eval()
        try:
            with torch.no_grad():
                self.teacher(clean_ids)
        finally:
            for h in handles:
                h.remove()

        # Re-register hooks for corrupt pass
        handles = []
        for layer_idx in self.sample_layers:
            hook_point = self.adapter.get_residual_hook_point(
                self.teacher, layer_idx,
            )
            handles.append(
                hook_point.register_forward_hook(
                    _make_hook(layer_idx, cache_corrupt),
                )
            )
        try:
            with torch.no_grad():
                self.teacher(corrupt_ids)
        finally:
            for h in handles:
                h.remove()

        # Mean diff norm across sampled layers
        total_norm = 0.0
        for idx in self.sample_layers:
            diff = cache_clean[idx] - cache_corrupt[idx]  # (1, seq, d_model)
            total_norm += diff.norm(dim=-1).mean().item()
        avg_norm = total_norm / max(len(self.sample_layers), 1)

        return avg_norm > self.threshold
