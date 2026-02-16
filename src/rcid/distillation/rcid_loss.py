"""L_RCID 损失函数，计算归一化因果痕迹的对齐误差。

核心公式:
    L_RCID = (1/|C|) Σ_{(l,t)∈C} || normalize(W* @ d^S_{l̂,t}) - normalize(d^T_{l,t}) ||²

其中:
    d^T_{l,t} = r^T_l(x_clean)[:,t,:] - r^T_l(x_corrupt)[:,t,:]
        教师因果痕迹，预计算并冻结（外部传入）
    d^S_{l̂,t} = r^S_{l̂}(x_clean)[:,t,:] - r^S_{l̂}(x_corrupt)[:,t,:]
        学生因果痕迹，在线计算，保留梯度
    W*: 冻结的 Procrustes 对齐矩阵 (d_T, d_S)，注册为 buffer
    normalize(x) = x / (||x|| + eps)

梯度流向:
    - W*: buffer，不参与梯度更新
    - d^T: 预计算，无梯度
    - d^S: 在线计算，梯度流回学生模型参数
    - 学生的两次前向传播（clean + corrupt）均保留计算图
"""

from __future__ import annotations

import logging
from typing import Iterator

import torch
from torch import nn

logger = logging.getLogger(__name__)


class RCIDLoss(nn.Module):
    """残差因果印记蒸馏损失。

    使用方式:
        loss_fn = RCIDLoss(W=W_star, checkpoints=C, layer_mapping=mapping)
        loss = loss_fn(teacher_imprints, student_model, clean_ids, corrupt_ids)
        loss.backward()  # 梯度仅流过学生模型
    """

    def __init__(
        self,
        W: torch.Tensor,                        # (d_T, d_S), 冻结
        checkpoints: list[tuple[int, int]],      # [(teacher_layer, token_pos), ...]
        layer_mapping: dict[int, int],           # {teacher_layer: student_layer}
        eps: float = 1e-8,
    ) -> None:
        """初始化 RCID 损失。

        Args:
            W: Procrustes 对齐矩阵, shape (d_T, d_S)。注册为 buffer。
            checkpoints: 因果检查点列表 [(teacher_layer, token_pos), ...]。
            layer_mapping: 教师层 → 学生层的映射 {t_layer: s_layer}。
            eps: 归一化的数值稳定常数。
        """
        super().__init__()

        assert W.dim() == 2, f"W should be 2D (d_T, d_S), got {W.dim()}D"
        assert len(checkpoints) > 0, "checkpoints list is empty"

        # 验证所有教师层都在 layer_mapping 中
        teacher_layers_needed = set(l for l, t in checkpoints)
        for t_layer in teacher_layers_needed:
            assert t_layer in layer_mapping, (
                f"Teacher layer {t_layer} in checkpoints but not in layer_mapping"
            )

        # W 注册为 buffer — 不参与梯度更新，但跟随 .to(device)
        self.register_buffer("W", W.clone())

        self.checkpoints = checkpoints
        self.layer_mapping = layer_mapping
        self.eps = eps

        # 预计算学生需要提取的层列表（去重排序）
        self._student_layers: list[int] = sorted(
            set(layer_mapping[l] for l, _ in checkpoints)
        )

        logger.info(
            "RCIDLoss: %d checkpoints, W shape (%d, %d), "
            "student layers %s, eps=%.0e",
            len(checkpoints), W.shape[0], W.shape[1],
            self._student_layers, eps,
        )

    def forward(
        self,
        teacher_imprints: dict[tuple[int, int], torch.Tensor],
        # 预计算的教师痕迹, {(teacher_layer, token_pos): (B, d_T)}
        student_model: nn.Module,
        clean_input: torch.Tensor,    # (B, seq_len)
        corrupt_input: torch.Tensor,  # (B, seq_len)
    ) -> torch.Tensor:
        """计算 RCID 损失。

        对每个检查点 (l, t)：
        1. 从 teacher_imprints 读取预计算的 d^T_{l,t}
        2. 在线提取学生 d^S_{l̂,t}（两次前向，保留梯度）
        3. 对齐: aligned = d^S @ W^T
        4. 归一化后计算 MSE

        Args:
            teacher_imprints: 预计算的教师痕迹字典。
                key = (teacher_layer, token_pos),
                value = (B, d_T) 张量（无梯度）。
            student_model: 学生模型 (GPT-2 风格, model.transformer.h)。
                前向传播保留梯度。
            clean_input: clean token ids, shape (B, seq_len)。
            corrupt_input: corrupt token ids, shape (B, seq_len)。

        Returns:
            RCID 损失标量（有梯度，可 backward）。
        """
        # ── 输入校验 ──────────────────────────────────────────────────
        assert clean_input.dim() == 2, (
            f"clean_input should be 2D (B, seq_len), got {clean_input.dim()}D"
        )
        assert clean_input.shape == corrupt_input.shape, (
            f"Shape mismatch: clean {clean_input.shape} "
            f"vs corrupt {corrupt_input.shape}"
        )

        seq_len = clean_input.shape[1]

        # 过滤越界检查点（variable-length 任务的短 batch 可能不含最长序列）
        active_cps = [
            (l, t) for l, t in self.checkpoints if t < seq_len
        ]
        if not active_cps:
            return torch.tensor(0.0, device=clean_input.device, requires_grad=True)

        # 验证所有需要的教师痕迹都已提供
        for l, t in active_cps:
            assert (l, t) in teacher_imprints, (
                f"Missing teacher imprint for checkpoint ({l}, {t})"
            )

        # ── 提取学生残差流（两次前向，保留梯度）──────────────────────
        student_residuals_clean = self._extract_student_residuals(
            student_model, clean_input,
        )
        student_residuals_corrupt = self._extract_student_residuals(
            student_model, corrupt_input,
        )

        # ── 逐检查点计算损失 ──────────────────────────────────────────
        total_loss = torch.tensor(
            0.0, device=clean_input.device, dtype=torch.float32,
        )

        for l_t, t_pos in active_cps:
            l_s = self.layer_mapping[l_t]

            # 教师痕迹（预计算，无梯度）
            d_T = teacher_imprints[(l_t, t_pos)]  # (B, d_T)

            # 学生痕迹（在线计算，有梯度）
            d_S = (
                student_residuals_clean[l_s][:, t_pos, :]       # (B, d_S)
                - student_residuals_corrupt[l_s][:, t_pos, :]   # (B, d_S)
            )  # (B, d_S)

            # Procrustes 对齐: (B, d_S) @ (d_S, d_T) → (B, d_T)
            d_S_aligned = d_S @ self.W.T  # (B, d_T)

            # 归一化（避免除以零）
            d_T_norm = d_T / d_T.norm(
                dim=-1, keepdim=True,
            ).clamp(min=self.eps)  # (B, d_T)

            d_S_norm = d_S_aligned / d_S_aligned.norm(
                dim=-1, keepdim=True,
            ).clamp(min=self.eps)  # (B, d_T)

            # 每个检查点的损失: ||normalize(aligned) - normalize(teacher)||²
            checkpoint_loss = (
                (d_T_norm - d_S_norm).pow(2).sum(dim=-1).mean()
            )  # scalar

            total_loss = total_loss + checkpoint_loss

        # 对活跃检查点数取平均
        loss = total_loss / len(active_cps)

        # 数值安全检查
        assert loss.isfinite(), f"RCID loss is {loss.item()}"

        return loss

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _extract_student_residuals(
        self,
        student_model: nn.Module,
        input_ids: torch.Tensor,  # (B, seq_len)
    ) -> dict[int, torch.Tensor]:
        """一次前向传播提取学生多层残差流，保留梯度。

        hook 中 **不** 调用 .detach()，确保梯度链完整。

        Args:
            student_model: 学生模型 (GPT-2 风格)。
            input_ids: token ids, shape (B, seq_len)。

        Returns:
            {student_layer: (B, seq_len, d_model)} 的残差流字典，
            所有 tensor 保留梯度。
        """
        residuals: dict[int, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHook] = []

        for layer in self._student_layers:
            def _make_hook(l: int):  # noqa: E741
                """闭包捕获层索引，避免 late-binding 陷阱。"""
                def hook(
                    module: nn.Module,
                    input: tuple[torch.Tensor, ...],
                    output: tuple[torch.Tensor, ...],
                ) -> None:
                    # GPT-2 block output: (hidden_states, present_kv, ...)
                    # 不 detach — 梯度必须流回学生参数
                    residuals[l] = output[0]  # (B, seq_len, d_model)
                return hook

            handle = student_model.transformer.h[layer].register_forward_hook(
                _make_hook(layer),
            )
            handles.append(handle)

        try:
            # 前向传播（不包裹 no_grad，保留计算图）
            student_model(input_ids)
        finally:
            # 无论是否异常，都移除 hooks
            for handle in handles:
                handle.remove()

        # 验证所有层都提取成功
        for layer in self._student_layers:
            assert layer in residuals, (
                f"Failed to extract student residual for layer {layer}"
            )

        return residuals
