"""MiniLM 风格知识蒸馏 (Wang et al., 2020)。

核心思路：匹配 self-attention 中 Value-Value 关系矩阵，
而非直接匹配 hidden states 或 attention weights。

对指定层的每个注意力头：
1. 提取 V vectors: V = hidden_states @ W_v (通过 hook c_attn)
2. 计算 Value 关系矩阵: R = (V @ V^T) / sqrt(d_v)
3. L_VR = KL(softmax(R_T / τ), softmax(R_S / τ))

总损失: L = α·L_VR + β·L_KL

与 TinyBERT 的区别:
- 匹配的是 Value 向量之间的关系矩阵，而非 attention weights
- 更好地保留了 token 间的语义相似性结构
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class MiniLMStyleKD(nn.Module):
    """MiniLM 风格蒸馏：Value 关系矩阵匹配 + KL 散度。

    Args:
        n_head_teacher: 教师注意力头数（如 12）。
        n_head_student: 学生注意力头数（如 6）。
        layer_pairs: (teacher_layer, student_layer) 配对列表。
        temperature: Value 关系矩阵 softmax 的温度。
        kl_temperature: KL 散度（logits）的温度。
        alpha: L_VR 权重。
        beta: L_KL 权重。
    """

    def __init__(
        self,
        n_head_teacher: int = 12,
        n_head_student: int = 6,
        layer_pairs: list[tuple[int, int]] | None = None,
        temperature: float = 1.0,
        kl_temperature: float = 4.0,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        assert n_head_teacher > 0 and n_head_student > 0
        assert temperature > 0 and kl_temperature > 0

        # 默认: 只匹配最后一层
        if layer_pairs is None:
            layer_pairs = [(11, 3)]
        assert len(layer_pairs) > 0, "layer_pairs is empty"

        self.layer_pairs = layer_pairs
        self.temperature = temperature
        self.kl_temperature = kl_temperature
        self.alpha = alpha
        self.beta = beta
        self.n_head_teacher = n_head_teacher
        self.n_head_student = n_head_student

        # 注意力头映射: 均匀分配，student head i → teacher heads [2i, 2i+1]
        heads_per_student = n_head_teacher // n_head_student
        self._head_mapping: list[list[int]] = []
        for sh in range(n_head_student):
            start = sh * heads_per_student
            self._head_mapping.append(
                list(range(start, start + heads_per_student)),
            )

        logger.info(
            "MiniLMStyleKD: %d pairs, heads_T=%d, heads_S=%d, "
            "τ_VR=%.1f, τ_KL=%.1f, α=%.1f, β=%.1f",
            len(layer_pairs), n_head_teacher, n_head_student,
            temperature, kl_temperature, alpha, beta,
        )

    def forward(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        input_ids: torch.Tensor,        # (B, seq_len)
    ) -> dict[str, torch.Tensor]:
        """计算 MiniLM 总损失。

        通过 hook 提取 c_attn 输出来获取 V vectors，
        计算 Value 关系矩阵并用 KL 散度匹配。

        Returns:
            字典包含: loss (总), loss_vr, loss_kl。
        """
        assert input_ids.dim() == 2

        t_layers = sorted(set(t for t, s in self.layer_pairs))
        s_layers = sorted(set(s for t, s in self.layer_pairs))

        # ── 教师: 单次前向同时提取 V + logits（无梯度）──
        teacher_values, t_logits = _extract_values_and_logits(
            teacher_model, input_ids, t_layers, detach=True,
        )

        # ── 学生: 单次前向同时提取 V + logits（保留梯度）──
        student_values, s_logits = _extract_values_and_logits(
            student_model, input_ids, s_layers, detach=False,
        )

        # ── L_VR: Value 关系矩阵 KL 散度 ──
        loss_vr = self._compute_vr_loss(teacher_values, student_values)

        # ── L_KL ──
        loss_kl = self._compute_kl_loss(t_logits, s_logits)

        total = self.alpha * loss_vr + self.beta * loss_kl
        assert total.isfinite(), f"MiniLM total loss is {total.item()}"

        return {
            "loss": total,
            "loss_vr": loss_vr.detach(),
            "loss_kl": loss_kl.detach(),
        }

    # ------------------------------------------------------------------
    # 子损失计算
    # ------------------------------------------------------------------

    def _compute_vr_loss(
        self,
        teacher_values: dict[int, torch.Tensor],
        # {layer: (B, n_head_T, seq_len, head_dim)}
        student_values: dict[int, torch.Tensor],
        # {layer: (B, n_head_S, seq_len, head_dim)}
    ) -> torch.Tensor:
        """L_VR = 对每个层配对、每个 student head，
        计算 KL(softmax(R_S/τ) || softmax(R_T/τ))。

        R = (V @ V^T) / sqrt(d_v)  →  (B, seq_len, seq_len)
        """
        device = next(iter(student_values.values())).device
        total = torch.tensor(0.0, device=device)
        count = 0

        for t_layer, s_layer in self.layer_pairs:
            V_T = teacher_values[t_layer]   # (B, H_T, S, d_v)
            V_S = student_values[s_layer]   # (B, H_S, S, d_v)

            d_v_T = V_T.shape[-1]  # teacher head dim
            d_v_S = V_S.shape[-1]  # student head dim

            for sh, th_indices in enumerate(self._head_mapping):
                # 教师: 对应 heads 的平均 V
                V_T_avg = V_T[:, th_indices, :, :].mean(dim=1)  # (B, S, d_v_T)
                V_S_h = V_S[:, sh, :, :]                        # (B, S, d_v_S)

                # Value 关系矩阵
                R_T = torch.bmm(
                    V_T_avg, V_T_avg.transpose(1, 2),
                ) / math.sqrt(d_v_T)                            # (B, S, S)
                R_S = torch.bmm(
                    V_S_h, V_S_h.transpose(1, 2),
                ) / math.sqrt(d_v_S)                            # (B, S, S)

                # KL 散度: 对每行 softmax
                tau = self.temperature
                R_T_probs = F.softmax(R_T / tau, dim=-1)        # (B, S, S)
                R_S_log_probs = F.log_softmax(R_S / tau, dim=-1)  # (B, S, S)

                # reshape → (B*S, S) 以便用 kl_div
                B, S_len, _ = R_T.shape
                kl = F.kl_div(
                    R_S_log_probs.reshape(B * S_len, S_len),
                    R_T_probs.reshape(B * S_len, S_len).detach(),
                    reduction="batchmean",
                )
                total = total + kl
                count += 1

        return total / max(count, 1)

    def _compute_kl_loss(
        self,
        teacher_logits: torch.Tensor,    # (B, S, V)
        student_logits: torch.Tensor,    # (B, S, V)
    ) -> torch.Tensor:
        """KL 散度蒸馏损失: KL(p_T || p_S) · τ²。"""
        tau = self.kl_temperature
        t_probs = F.softmax(teacher_logits.detach() / tau, dim=-1)  # (B, S, V)
        s_log_probs = F.log_softmax(student_logits / tau, dim=-1)   # (B, S, V)
        kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean")
        return kl * (tau ** 2)


# ======================================================================
# 内部工具: Value vector 提取
# ======================================================================

def _extract_values_and_logits(
    model: nn.Module,
    input_ids: torch.Tensor,       # (B, seq_len)
    layers: list[int],
    detach: bool = True,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """单次前向同时提取 Value vectors 和 logits。

    通过 hook GPT-2 的 c_attn 提取 Value vectors:
    c_attn 输出 (B, S, 3*embed_dim)，split 后第三部分是 V。
    reshape 为 (B, n_heads, S, head_dim)。

    Returns:
        (values_dict, logits) —
        values_dict: {layer: (B, n_heads, seq_len, head_dim)}
        logits: (B, seq_len, vocab_size)
    """
    values: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHook] = []

    for layer in layers:
        attn_module = model.transformer.h[layer].attn
        n_heads = attn_module.num_heads
        head_dim = attn_module.head_dim

        def _make_hook(l: int, nh: int, hd: int, do_detach: bool):  # noqa: E741
            def hook(
                mod: nn.Module,
                inp: tuple,
                out: torch.Tensor,
            ) -> None:
                # out: (B, S, 3*embed_dim) from c_attn
                embed_dim = nh * hd
                v_raw = out[:, :, 2 * embed_dim:]          # (B, S, embed_dim)
                B, S, _ = v_raw.shape
                v = v_raw.view(B, S, nh, hd).permute(0, 2, 1, 3)
                # v: (B, n_heads, S, head_dim)
                if do_detach:
                    v = v.detach()
                values[l] = v
            return hook

        handle = attn_module.c_attn.register_forward_hook(
            _make_hook(layer, n_heads, head_dim, detach),
        )
        handles.append(handle)

    try:
        if detach:
            with torch.no_grad():
                out = model(input_ids)
        else:
            out = model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    for layer in layers:
        assert layer in values, f"Failed to extract values for L{layer}"

    logits = out.logits  # (B, S, V)
    if detach:
        logits = logits.detach()

    return values, logits
