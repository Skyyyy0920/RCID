"""Logit Lens Monitor: diagnose per-layer causal sensitivity alignment.

Projects intermediate residual streams through each model's own lm_head
to get logits in the shared vocabulary space R^V, then computes OCID-style
cosine similarity on the projected differences.

This avoids Procrustes alignment entirely -- both models project to R^V
through their own lm_heads.

Usage::

    monitor = LogitLensMonitor(teacher, student, t_adapter, s_adapter)
    report = monitor.diagnose(clean_ids, corrupt_ids)
    # report["layer_health"] = {0: 0.82, 1: 0.75, ...}
    # report["overall_health"] = 0.68
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rcid.models.adapter import ModelAdapter

logger = logging.getLogger(__name__)


class LogitLensMonitor:
    """Monitor causal sensitivity alignment at every layer via logit lens.

    For each student layer l:
      1. Extract residual stream h_S^l for clean and corrupt inputs
      2. Project through student's lm_head: z_S^l = lm_head_S(h_S^l)
      3. Compute delta_S^l = softmax(z_S^l_clean) - softmax(z_S^l_corrupt)

    For the mapped teacher layer:
      Same process using teacher's lm_head.

    Then: health(l) = mean cosine_similarity(delta_S^l, delta_T^l_hat)
    over active positions (where teacher delta norm > threshold).

    Args:
        teacher, student: Models (eval mode, no grad).
        t_adapter, s_adapter: ModelAdapter instances.
        temperature: Softmax temperature for logit lens projection.
        sample_layers: Which student layers to monitor.
            Default None = monitor all layers.
        min_delta_norm: Ignore positions with tiny teacher delta.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        t_adapter: ModelAdapter,
        s_adapter: ModelAdapter,
        temperature: float = 2.0,
        sample_layers: list[int] | None = None,
        min_delta_norm: float = 1e-6,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.t_adp = t_adapter
        self.s_adp = s_adapter
        self.temperature = temperature
        self.min_delta_norm = min_delta_norm

        # Determine which layers to monitor
        n_student_layers = s_adapter.get_num_layers(student)
        if sample_layers is not None:
            self.student_layers = sample_layers
        else:
            self.student_layers = list(range(n_student_layers))

        # Linear layer mapping: student layer l -> teacher layer l_hat
        n_teacher_layers = t_adapter.get_num_layers(teacher)
        self.layer_map: dict[int, int] = {}
        for sl in self.student_layers:
            tl = round(sl * n_teacher_layers / n_student_layers)
            tl = min(tl, n_teacher_layers - 1)
            self.layer_map[sl] = tl

    @torch.no_grad()
    def diagnose(
        self,
        clean_ids: torch.Tensor,    # (B, L)
        corrupt_ids: torch.Tensor,  # (B, L)
        mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Run one diagnostic pass on a batch of contrastive pairs.

        Returns dict with:
            layer_health: {student_layer_idx: cosine_similarity} -- higher=better
            overall_health: mean over all monitored layers
            worst_layers: list of (layer_idx, health) sorted ascending
        """
        # Extract teacher residuals at mapped layers
        t_layers_needed = sorted(set(self.layer_map.values()))
        t_residuals_clean = self._extract_residuals(
            self.teacher, self.t_adp, clean_ids, t_layers_needed,
        )
        t_residuals_corrupt = self._extract_residuals(
            self.teacher, self.t_adp, corrupt_ids, t_layers_needed,
        )

        # Extract student residuals
        s_residuals_clean = self._extract_residuals(
            self.student, self.s_adp, clean_ids, self.student_layers,
        )
        s_residuals_corrupt = self._extract_residuals(
            self.student, self.s_adp, corrupt_ids, self.student_layers,
        )

        # Get lm_heads
        t_lm_head = self.t_adp.get_lm_head(self.teacher)
        s_lm_head = self.s_adp.get_lm_head(self.student)

        # Per-layer health
        layer_health: dict[int, float] = {}
        T = self.temperature

        for sl in self.student_layers:
            tl = self.layer_map[sl]

            # Project to vocab space via respective lm_heads
            # Cast residuals to match lm_head weight dtype (may be fp16)
            t_dtype = t_lm_head.weight.dtype
            s_dtype = s_lm_head.weight.dtype
            t_logits_clean = t_lm_head(t_residuals_clean[tl].to(t_dtype))
            t_logits_corrupt = t_lm_head(t_residuals_corrupt[tl].to(t_dtype))
            s_logits_clean = s_lm_head(s_residuals_clean[sl].to(s_dtype))
            s_logits_corrupt = s_lm_head(s_residuals_corrupt[sl].to(s_dtype))

            # Probability deltas (compute in float32 for numerical stability)
            delta_T = (
                F.softmax(t_logits_clean.float() / T, dim=-1)
                - F.softmax(t_logits_corrupt.float() / T, dim=-1)
            )  # (B, L, V)
            delta_S = (
                F.softmax(s_logits_clean.float() / T, dim=-1)
                - F.softmax(s_logits_corrupt.float() / T, dim=-1)
            )  # (B, L, V)

            # Cosine similarity, weighted by teacher norm
            t_norm = delta_T.norm(dim=-1)  # (B, L)
            active = t_norm > self.min_delta_norm
            if mask is not None:
                active = active & (mask > 0)

            cos = F.cosine_similarity(delta_S, delta_T, dim=-1)  # (B, L)
            w = t_norm * active.float()
            w_sum = w.sum().clamp(min=1e-8)
            health = (w * cos).sum() / w_sum
            layer_health[sl] = health.item()

        overall = sum(layer_health.values()) / max(len(layer_health), 1)
        worst = sorted(layer_health.items(), key=lambda x: x[1])

        return {
            "layer_health": layer_health,
            "overall_health": overall,
            "worst_layers": worst[:5],
        }

    def _extract_residuals(
        self,
        model: nn.Module,
        adapter: ModelAdapter,
        input_ids: torch.Tensor,
        layers: list[int],
    ) -> dict[int, torch.Tensor]:
        """Extract residual stream at specified layers via hooks."""
        cache: dict[int, torch.Tensor] = {}
        handles: list[Any] = []

        for layer_idx in layers:
            hook_point = adapter.get_residual_hook_point(model, layer_idx)

            def _make_hook(idx: int):  # noqa: E301
                def _hook(mod: nn.Module, inp: Any, out: Any) -> None:
                    cache[idx] = adapter.parse_layer_output(out).detach()
                return _hook

            handles.append(hook_point.register_forward_hook(_make_hook(layer_idx)))

        try:
            model(input_ids)
        finally:
            for h in handles:
                h.remove()

        return cache
