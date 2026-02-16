"""TinyBERT 风格知识蒸馏 (Jiao et al., 2020)。

同时匹配教师和学生的:
1. Hidden states:  L_hidden = MSE(W_h @ h^S, h^T)
2. Attention maps:  L_attn  = MSE(A^S_mapped, A^T_mapped)
3. 输出分布:        L_KL    = KL(softmax(z_T/τ), softmax(z_S/τ)) · τ²

总损失: L = α·L_hidden + β·L_attn + γ·L_KL

与 FitNets 的区别:
- 额外匹配注意力矩阵（不仅仅是隐藏层）
- 注意力匹配需要处理 teacher/student 头数不同的问题

使用 output_hidden_states=True 和 output_attentions=True 实现单次前向
传播提取所有中间表示，避免多次前向或 hook 管理的复杂性。

注意：GPT2SdpaAttention（PyTorch 2.0+默认）在 output_attentions=True
时可能返回 None。需要强制切换到 eager attention 实现。
"""

from __future__ import annotations

import contextlib
import logging

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _force_eager_attention(model: nn.Module):
    """临时强制 GPT-2 模型使用 eager attention（非 SDPA）。

    GPT2SdpaAttention.forward 在 output_attentions=True 时应回退到
    eager attention，但在某些 PyTorch/transformers 版本中仍返回 None。
    本 context manager 直接替换 attention class 确保正确返回注意力权重。
    """
    # 收集需要修补的 attention 模块
    patched: list[tuple[nn.Module, str, type]] = []
    try:
        # 检查是否有 GPT2SdpaAttention
        from transformers.models.gpt2.modeling_gpt2 import (
            GPT2Attention,
        )
        try:
            from transformers.models.gpt2.modeling_gpt2 import (
                GPT2SdpaAttention,
            )
        except ImportError:
            # 旧版 transformers 没有 SDPA，不需要修补
            yield
            return

        for block in getattr(model, "transformer", model).h:
            attn_mod = block.attn
            if isinstance(attn_mod, GPT2SdpaAttention):
                # 替换 __class__ 让 forward 走 GPT2Attention 的路径
                patched.append((attn_mod, "original_cls", type(attn_mod)))
                attn_mod.__class__ = GPT2Attention
        yield
    finally:
        # 恢复原始 class
        for attn_mod, _, orig_cls in patched:
            attn_mod.__class__ = orig_cls


class TinyBERTStyleKD(nn.Module):
    """TinyBERT 风格蒸馏：隐藏层匹配 + 注意力矩阵匹配 + KL 散度。

    Args:
        d_teacher: 教师模型隐藏维度（如 768）。
        d_student: 学生模型隐藏维度（如 384）。
        n_head_teacher: 教师注意力头数（如 12）。
        n_head_student: 学生注意力头数（如 6）。
        layer_pairs: (teacher_layer, student_layer) 配对列表。
        temperature: KL 散度的温度。
        alpha: L_hidden 权重。
        beta: L_attn 权重。
        gamma: L_KL 权重。
    """

    def __init__(
        self,
        d_teacher: int = 768,
        d_student: int = 384,
        n_head_teacher: int = 12,
        n_head_student: int = 6,
        layer_pairs: list[tuple[int, int]] | None = None,
        temperature: float = 4.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        assert d_teacher > 0 and d_student > 0
        assert n_head_teacher > 0 and n_head_student > 0
        assert temperature > 0

        # 默认层配对: student 4层 → teacher {2,5,8,11}
        if layer_pairs is None:
            layer_pairs = [(2, 0), (5, 1), (8, 2), (11, 3)]
        assert len(layer_pairs) > 0, "layer_pairs is empty"

        self.layer_pairs = layer_pairs
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_head_teacher = n_head_teacher
        self.n_head_student = n_head_student

        # 每个层配对一个可学习投影: d_S → d_T
        self.hidden_projections = nn.ModuleDict()
        for t_layer, s_layer in layer_pairs:
            self.hidden_projections[f"proj_{t_layer}_{s_layer}"] = nn.Linear(
                d_student, d_teacher, bias=False,
            )

        # 注意力头映射: 均匀分配，student head i → 平均 teacher heads [2i, 2i+1]
        heads_per_student = n_head_teacher // n_head_student
        self._attn_head_mapping: list[list[int]] = []
        for sh in range(n_head_student):
            start = sh * heads_per_student
            self._attn_head_mapping.append(
                list(range(start, start + heads_per_student)),
            )

        logger.info(
            "TinyBERTStyleKD: %d pairs, d_T=%d, d_S=%d, "
            "heads_T=%d, heads_S=%d, α=%.1f, β=%.1f, γ=%.1f",
            len(layer_pairs), d_teacher, d_student,
            n_head_teacher, n_head_student, alpha, beta, gamma,
        )

    @staticmethod
    def make_uniform_pairs(
        n_teacher_layers: int,
        n_student_layers: int,
    ) -> list[tuple[int, int]]:
        """等间隔层映射。L_T=12, L_S=4 → [(2,0),(5,1),(8,2),(11,3)]。"""
        assert n_teacher_layers > 0 and n_student_layers > 0
        pairs: list[tuple[int, int]] = []
        for s in range(n_student_layers):
            if n_student_layers == 1:
                t = n_teacher_layers - 1
            else:
                t = round((s + 1) * (n_teacher_layers - 1) / n_student_layers)
            pairs.append((t, s))
        return pairs

    def forward(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        input_ids: torch.Tensor,        # (B, seq_len)
    ) -> dict[str, torch.Tensor]:
        """计算 TinyBERT 总损失。

        单次前向传播提取 hidden_states + attentions + logits。
        hidden_states[i] 对应 layer i 的输出（index 0 = embedding 输出）。
        attentions[i] 对应 layer i 的注意力权重。

        Returns:
            字典包含: loss (总), loss_hidden, loss_attn, loss_kl。
        """
        assert input_ids.dim() == 2

        # ── 教师前向（无梯度，单次调用）──
        # 强制 eager attention 确保 output_attentions 返回非 None
        with torch.no_grad(), _force_eager_attention(teacher_model):
            t_out = teacher_model(
                input_ids,
                output_hidden_states=True,
                output_attentions=True,
            )
        # hidden_states: tuple of (B, S, d_T), len = n_layers + 1
        # hidden_states[0] = embedding output, [i+1] = layer i output
        # attentions: tuple of (B, H_T, S, S), len = n_layers
        # attentions[i] = layer i attention weights

        # ── 学生前向（保留梯度，单次调用）──
        with _force_eager_attention(student_model):
            s_out = student_model(
                input_ids,
                output_hidden_states=True,
                output_attentions=True,
            )

        # ── L_hidden: MSE(W_h @ h^S, h^T) ──
        loss_hidden = self._compute_hidden_loss(
            t_out.hidden_states, s_out.hidden_states,
        )

        # ── L_attn: MSE(A^S, avg(A^T)) ──
        loss_attn = self._compute_attn_loss(
            t_out.attentions, s_out.attentions,
        )

        # ── L_KL ──
        loss_kl = self._compute_kl_loss(
            t_out.logits, s_out.logits,  # (B, S, V)
        )

        total = (
            self.alpha * loss_hidden
            + self.beta * loss_attn
            + self.gamma * loss_kl
        )
        assert total.isfinite(), f"TinyBERT total loss is {total.item()}"

        return {
            "loss": total,
            "loss_hidden": loss_hidden.detach(),
            "loss_attn": loss_attn.detach(),
            "loss_kl": loss_kl.detach(),
        }

    # ------------------------------------------------------------------
    # 子损失计算
    # ------------------------------------------------------------------

    def _compute_hidden_loss(
        self,
        teacher_hs: tuple[torch.Tensor, ...],   # len = T_layers + 1
        student_hs: tuple[torch.Tensor, ...],   # len = S_layers + 1
    ) -> torch.Tensor:
        """L_hidden = (1/|P|) Σ MSE(W_h @ h^S_{l_S}, h^T_{l_T})。

        hidden_states[0] = embedding output, [i+1] = layer i 的 block 输出。
        """
        device = student_hs[0].device
        total = torch.tensor(0.0, device=device)

        for t_layer, s_layer in self.layer_pairs:
            proj = self.hidden_projections[f"proj_{t_layer}_{s_layer}"]
            h_T = teacher_hs[t_layer + 1]   # (B, S, d_T) — layer output
            h_S = student_hs[s_layer + 1]   # (B, S, d_S) — layer output
            h_S_proj = proj(h_S)             # (B, S, d_T)
            total = total + F.mse_loss(h_S_proj, h_T.detach())

        return total / len(self.layer_pairs)

    def _compute_attn_loss(
        self,
        teacher_attns: tuple[torch.Tensor, ...],  # each (B, H_T, S, S)
        student_attns: tuple[torch.Tensor, ...],  # each (B, H_S, S, S)
    ) -> torch.Tensor:
        """L_attn: 学生每个 head 匹配教师对应 heads 的平均注意力矩阵。"""
        device = student_attns[0].device
        total = torch.tensor(0.0, device=device)
        count = 0

        for t_layer, s_layer in self.layer_pairs:
            A_T = teacher_attns[t_layer].detach()  # (B, H_T, S, S)
            A_S = student_attns[s_layer]           # (B, H_S, S, S)

            for sh, th_indices in enumerate(self._attn_head_mapping):
                # 教师对应 heads 的平均注意力
                A_T_avg = A_T[:, th_indices, :, :].mean(dim=1)  # (B, S, S)
                A_S_h = A_S[:, sh, :, :]                        # (B, S, S)
                total = total + F.mse_loss(A_S_h, A_T_avg)
                count += 1

        return total / max(count, 1)

    def _compute_kl_loss(
        self,
        teacher_logits: torch.Tensor,    # (B, S, V)
        student_logits: torch.Tensor,    # (B, S, V)
    ) -> torch.Tensor:
        """KL 散度蒸馏损失: KL(p_T || p_S) · τ²。"""
        tau = self.temperature
        t_probs = F.softmax(teacher_logits.detach() / tau, dim=-1)  # (B, S, V)
        s_log_probs = F.log_softmax(student_logits / tau, dim=-1)   # (B, S, V)
        kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean")
        return kl * (tau ** 2)
