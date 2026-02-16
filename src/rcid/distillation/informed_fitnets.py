"""Informed FitNets — 关键消融基线。

与 RCID 使用完全相同的因果检查点和 Procrustes 对齐矩阵 W，
但匹配目标不同：

| 方法              | 匹配目标                       | W 矩阵    |
|-------------------|-------------------------------|-----------|
| RCID              | d^T = h^T_clean - h^T_corrupt | 冻结 buffer |
| Informed FitNets  | h^T_clean 本身                 | 冻结 buffer |

这个消融回答的核心问题:
  "RCID 的优势来自因果痕迹差值作为匹配目标，还是仅仅来自在正确位置匹配？"

  - Informed FitNets ≈ RCID → 价值在于检查点选择
  - Informed FitNets << RCID → 价值在于因果痕迹差值

损失公式:
    L_IF = (1/|C|) Σ_{(l,t)∈C} MSE(W @ h^S_{l̂,t}, h^T_{l,t})

其中 h^T_{l,t} 是教师在 clean 输入上的残差流（非差值），
     h^S_{l̂,t} 是学生在 clean 输入上的残差流。
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class InformedFitNetsLoss(nn.Module):
    """Informed FitNets：在 RCID 检查点位置匹配完整表示（非因果差值）。

    与 RCIDLoss 接口高度一致，便于公平对比:
    - 相同的 checkpoints
    - 相同的 layer_mapping
    - 相同的冻结 W（Procrustes 对齐矩阵，buffer）
    - 唯一区别：匹配 h^T_clean 而非 d^T = h^T_clean - h^T_corrupt
    """

    def __init__(
        self,
        W: torch.Tensor,                        # (d_T, d_S), 冻结
        checkpoints: list[tuple[int, int]],      # [(teacher_layer, token_pos), ...]
        layer_mapping: dict[int, int],           # {teacher_layer: student_layer}
        eps: float = 1e-8,
    ) -> None:
        """初始化。参数签名与 RCIDLoss 完全一致。

        Args:
            W: Procrustes 对齐矩阵, shape (d_T, d_S)。注册为 buffer。
            checkpoints: 因果检查点列表 [(teacher_layer, token_pos), ...]。
            layer_mapping: 教师层 → 学生层的映射。
            eps: 归一化的数值稳定常数。
        """
        super().__init__()

        assert W.dim() == 2, f"W should be 2D (d_T, d_S), got {W.dim()}D"
        assert len(checkpoints) > 0, "checkpoints list is empty"

        teacher_layers_needed = set(l for l, t in checkpoints)
        for t_layer in teacher_layers_needed:
            assert t_layer in layer_mapping, (
                f"Teacher layer {t_layer} in checkpoints "
                f"but not in layer_mapping"
            )

        self.register_buffer("W", W.clone())

        self.checkpoints = checkpoints
        self.layer_mapping = layer_mapping
        self.eps = eps

        self._student_layers: list[int] = sorted(
            set(layer_mapping[l] for l, _ in checkpoints)
        )

        logger.info(
            "InformedFitNetsLoss: %d checkpoints, W shape (%d, %d), "
            "student layers %s",
            len(checkpoints), W.shape[0], W.shape[1],
            self._student_layers,
        )

    def forward(
        self,
        teacher_clean_residuals: dict[tuple[int, int], torch.Tensor],
        # 预计算的教师 clean 残差流切片,
        # {(teacher_layer, token_pos): (B, d_T)}，无梯度
        student_model: nn.Module,
        clean_input: torch.Tensor,    # (B, seq_len)
    ) -> torch.Tensor:
        """计算 Informed FitNets 损失。

        与 RCIDLoss.forward 的关键区别:
        1. teacher 传入的是 h^T_clean 而非 d^T = h^T_clean - h^T_corrupt
        2. 不需要 corrupt_input（只用 clean）
        3. 不做归一化（匹配完整表示的绝对值，而非方向）

        Args:
            teacher_clean_residuals: 预计算的教师 clean 残差流字典。
                key = (teacher_layer, token_pos),
                value = (B, d_T) 张量（无梯度）。
            student_model: 学生模型 (GPT-2 风格)。
            clean_input: clean token ids, shape (B, seq_len)。

        Returns:
            Informed FitNets 损失标量。
        """
        assert clean_input.dim() == 2, (
            f"clean_input should be 2D, got {clean_input.dim()}D"
        )

        seq_len = clean_input.shape[1]

        # 过滤越界检查点（variable-length 任务的短 batch）
        active_cps = [
            (l, t) for l, t in self.checkpoints if t < seq_len
        ]
        if not active_cps:
            return torch.tensor(0.0, device=clean_input.device, requires_grad=True)

        for l, t in active_cps:
            assert (l, t) in teacher_clean_residuals, (
                f"Missing teacher residual for checkpoint ({l}, {t})"
            )

        # ── 提取学生 clean 残差流（一次前向，保留梯度）──
        student_residuals = self._extract_student_residuals(
            student_model, clean_input,
        )

        # ── 逐检查点计算 MSE 损失 ──
        total_loss = torch.tensor(
            0.0, device=clean_input.device, dtype=torch.float32,
        )

        for l_t, t_pos in active_cps:
            l_s = self.layer_mapping[l_t]

            # 教师 clean 表示（预计算，无梯度）
            h_T = teacher_clean_residuals[(l_t, t_pos)]  # (B, d_T)

            # 学生 clean 表示（在线计算，有梯度）
            h_S = student_residuals[l_s][:, t_pos, :]    # (B, d_S)

            # Procrustes 对齐: (B, d_S) @ (d_S, d_T) → (B, d_T)
            h_S_aligned = h_S @ self.W.T                  # (B, d_T)

            # MSE（不做归一化 — 这是与 RCID 的关键差异）
            checkpoint_loss = (h_S_aligned - h_T).pow(2).mean()  # scalar

            total_loss = total_loss + checkpoint_loss

        loss = total_loss / len(active_cps)

        assert loss.isfinite(), f"InformedFitNets loss is {loss.item()}"
        return loss

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _extract_student_residuals(
        self,
        student_model: nn.Module,
        input_ids: torch.Tensor,  # (B, seq_len)
    ) -> dict[int, torch.Tensor]:
        """提取学生多层残差流，保留梯度。

        与 RCIDLoss._extract_student_residuals 完全相同。
        """
        residuals: dict[int, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHook] = []

        for layer in self._student_layers:
            def _make_hook(l: int):  # noqa: E741
                def hook(
                    module: nn.Module,
                    input: tuple[torch.Tensor, ...],
                    output: tuple[torch.Tensor, ...],
                ) -> None:
                    residuals[l] = output[0]  # (B, seq_len, d_model)
                return hook

            handle = student_model.transformer.h[layer].register_forward_hook(
                _make_hook(layer),
            )
            handles.append(handle)

        try:
            student_model(input_ids)
        finally:
            for handle in handles:
                handle.remove()

        for layer in self._student_layers:
            assert layer in residuals, (
                f"Failed to extract student residual for layer {layer}"
            )

        return residuals
