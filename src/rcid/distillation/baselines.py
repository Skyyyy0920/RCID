"""基线蒸馏方法实现：标准 KD / FitNets / Prakash CKA。

| 方法          | 匹配对象     | 对齐矩阵         | 需要对比数据? |
|---------------|------------|------------------|-------------|
| StandardKD    | 输出 logits | 无               | 否          |
| FitNets       | 完整残差流   | 可学习 (nn.Linear)| 否          |
| PrakashCKA    | 残差流 CKA  | 可学习 (nn.Linear)| 否          |
| **RCID**      | 因果痕迹     | 冻结 (buffer)     | **是**      |
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


# ======================================================================
# 1. Standard Knowledge Distillation (Hinton et al., 2015)
# ======================================================================

class StandardKDLoss(nn.Module):
    """标准知识蒸馏：L = KL(softmax(z_T/τ), softmax(z_S/τ)) · τ²。

    只匹配输出层分布，不关心中间表示。不需要对比数据。
    """

    def __init__(self, temperature: float = 4.0) -> None:
        super().__init__()
        assert temperature > 0, f"Temperature must be positive, got {temperature}"
        self.temperature = temperature
        logger.info("StandardKDLoss: temperature=%.1f", temperature)

    def forward(
        self,
        teacher_logits: torch.Tensor,  # (B, seq_len, vocab_size)
        student_logits: torch.Tensor,  # (B, seq_len, vocab_size)
    ) -> torch.Tensor:
        """计算 KL 散度蒸馏损失。teacher_logits 应无梯度。"""
        assert teacher_logits.dim() == 3, (
            f"teacher_logits should be 3D, got {teacher_logits.dim()}D"
        )
        assert student_logits.shape == teacher_logits.shape, (
            f"Shape mismatch: teacher {teacher_logits.shape} "
            f"vs student {student_logits.shape}"
        )

        tau = self.temperature
        teacher_probs = F.softmax(teacher_logits / tau, dim=-1)      # (B, S, V)
        student_log_probs = F.log_softmax(student_logits / tau, dim=-1)  # (B, S, V)

        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        loss = kl * (tau ** 2)  # τ² 抵消温度对梯度的缩放

        assert loss.isfinite(), f"KD loss is {loss.item()}"
        return loss


# ======================================================================
# 2. FitNets: 中间层特征匹配 (Romero et al., 2015)
# ======================================================================

class FitNetsLoss(nn.Module):
    """L = (1/|P|) Σ ||W_l @ h^S_{l_S} - h^T_{l_T}||²。

    W_l 是 **可学习的** nn.Linear（与 RCID 的冻结 W 不同）。
    匹配完整残差流而非因果痕迹，不需要对比数据。
    """

    def __init__(
        self,
        d_teacher: int,
        d_student: int,
        layer_pairs: list[tuple[int, int]],
    ) -> None:
        """Args: layer_pairs = [(teacher_layer, student_layer), ...]。"""
        super().__init__()
        assert d_teacher > 0 and d_student > 0
        assert len(layer_pairs) > 0, "layer_pairs is empty"

        self.layer_pairs = layer_pairs

        # 每个配对一个可学习投影: d_S → d_T
        self.projections = nn.ModuleDict()
        for t_layer, s_layer in layer_pairs:
            self.projections[f"proj_{t_layer}_{s_layer}"] = nn.Linear(
                d_student, d_teacher, bias=False,
            )

        self._teacher_layers = sorted(set(t for t, s in layer_pairs))
        self._student_layers = sorted(set(s for t, s in layer_pairs))

        logger.info(
            "FitNetsLoss: %d pairs, d_T=%d, d_S=%d",
            len(layer_pairs), d_teacher, d_student,
        )

    @staticmethod
    def make_uniform_pairs(
        n_teacher_layers: int,
        n_student_layers: int,
    ) -> list[tuple[int, int]]:
        """等间隔层映射。L_T=12, L_S=4 → [(0,0),(4,1),(8,2),(11,3)]。"""
        assert n_teacher_layers > 0 and n_student_layers > 0
        pairs: list[tuple[int, int]] = []
        for s in range(n_student_layers):
            if n_student_layers == 1:
                t = n_teacher_layers - 1
            else:
                t = round(s * (n_teacher_layers - 1) / (n_student_layers - 1))
            pairs.append((t, s))
        return pairs

    def forward(
        self,
        teacher_residuals: dict[int, torch.Tensor],
        # {teacher_layer: (B, seq_len, d_T)}，预提取，无梯度
        student_model: nn.Module,
        input_ids: torch.Tensor,  # (B, seq_len)
    ) -> torch.Tensor:
        """计算 FitNets 特征匹配损失。"""
        assert input_ids.dim() == 2
        for t in self._teacher_layers:
            assert t in teacher_residuals, f"Missing teacher residual L{t}"

        student_residuals = _extract_residuals(
            student_model, input_ids, self._student_layers,
        )

        total_loss = torch.tensor(0.0, device=input_ids.device)
        for t_layer, s_layer in self.layer_pairs:
            proj = self.projections[f"proj_{t_layer}_{s_layer}"]
            h_T = teacher_residuals[t_layer]   # (B, seq_len, d_T)
            h_S = student_residuals[s_layer]   # (B, seq_len, d_S)
            h_S_proj = proj(h_S)               # (B, seq_len, d_T)
            total_loss = total_loss + (h_S_proj - h_T).pow(2).mean()

        loss = total_loss / len(self.layer_pairs)
        assert loss.isfinite(), f"FitNets loss is {loss.item()}"
        return loss


# ======================================================================
# 3. Prakash CKA 对齐（简化版）
# ======================================================================

class PrakashCKALoss(nn.Module):
    """L = (1/|M|) Σ (1 - CKA(h^T_{l_T}, W_l @ h^S_{l_S}))。

    简化版：层级别 CKA 匹配（原论文是 attention head 级别）。
    CKA 作为可微分损失信号（1 - CKA → 最小化）。
    """

    def __init__(
        self,
        d_teacher: int,
        d_student: int,
        layer_pairs: list[tuple[int, int]],
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        assert d_teacher > 0 and d_student > 0
        assert len(layer_pairs) > 0, "layer_pairs is empty"

        self.layer_pairs = layer_pairs
        self.eps = eps

        self.projections = nn.ModuleDict()
        for t_layer, s_layer in layer_pairs:
            self.projections[f"proj_{t_layer}_{s_layer}"] = nn.Linear(
                d_student, d_teacher, bias=False,
            )

        self._teacher_layers = sorted(set(t for t, s in layer_pairs))
        self._student_layers = sorted(set(s for t, s in layer_pairs))

        logger.info(
            "PrakashCKALoss: %d pairs, d_T=%d, d_S=%d",
            len(layer_pairs), d_teacher, d_student,
        )

    def forward(
        self,
        teacher_residuals: dict[int, torch.Tensor],
        # {layer: (B, seq_len, d_T)}
        student_model: nn.Module,
        input_ids: torch.Tensor,  # (B, seq_len)
    ) -> torch.Tensor:
        """CKA 在 (B*seq_len) 样本维度上计算。"""
        assert input_ids.dim() == 2
        for t in self._teacher_layers:
            assert t in teacher_residuals, f"Missing teacher residual L{t}"

        student_residuals = _extract_residuals(
            student_model, input_ids, self._student_layers,
        )

        total_loss = torch.tensor(0.0, device=input_ids.device)
        for t_layer, s_layer in self.layer_pairs:
            proj = self.projections[f"proj_{t_layer}_{s_layer}"]
            h_T = teacher_residuals[t_layer]   # (B, seq_len, d_T)
            h_S = student_residuals[s_layer]   # (B, seq_len, d_S)
            h_S_proj = proj(h_S)               # (B, seq_len, d_T)

            B, S, d_T = h_T.shape
            X = h_T.reshape(B * S, d_T)        # (N, d_T)
            Y = h_S_proj.reshape(B * S, d_T)   # (N, d_T)

            cka = _differentiable_linear_cka(X, Y, self.eps)
            total_loss = total_loss + (1.0 - cka)

        loss = total_loss / len(self.layer_pairs)
        assert loss.isfinite(), f"PrakashCKA loss is {loss.item()}"
        return loss


# ======================================================================
# 共享内部工具
# ======================================================================

def _extract_residuals(
    model: nn.Module,
    input_ids: torch.Tensor,       # (B, seq_len)
    layers: list[int],
) -> dict[int, torch.Tensor]:     # {layer: (B, seq_len, d_model)}
    """提取 GPT-2 多层残差流，保留梯度（hook 中不 .detach()）。"""
    residuals: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHook] = []

    for layer in layers:
        def _make_hook(l: int):  # noqa: E741
            def hook(mod: nn.Module, inp: tuple, out: tuple) -> None:
                residuals[l] = out[0]  # (B, seq_len, d_model)
            return hook

        handle = model.transformer.h[layer].register_forward_hook(
            _make_hook(layer),
        )
        handles.append(handle)

    try:
        model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    for layer in layers:
        assert layer in residuals, f"Failed to extract residual for L{layer}"
    return residuals


def _differentiable_linear_cka(
    X: torch.Tensor,   # (N, d1)
    Y: torch.Tensor,   # (N, d2)
    eps: float = 1e-8,
) -> torch.Tensor:     # scalar, 保留梯度
    """可微分 Linear CKA（返回 Tensor 而非 float，支持 backward）。

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F · ||Y^T Y||_F)
    """
    X = X - X.mean(dim=0, keepdim=True)  # (N, d1)
    Y = Y - Y.mean(dim=0, keepdim=True)  # (N, d2)

    YtX = Y.T @ X                         # (d2, d1)
    XtX = X.T @ X                         # (d1, d1)
    YtY = Y.T @ Y                         # (d2, d2)

    hsic_xy = (YtX * YtX).sum()           # scalar
    hsic_xx = (XtX * XtX).sum()           # scalar
    hsic_yy = (YtY * YtY).sum()           # scalar

    denominator = torch.sqrt(hsic_xx * hsic_yy).clamp(min=eps)
    return hsic_xy / denominator           # scalar, 有梯度
