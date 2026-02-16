"""MiniLM 风格蒸馏基线单元测试。

测试覆盖：
- MiniLMStyleKD 构造和头映射
- Value vector 提取正确性（shape, per-head）
- Value 关系矩阵 shape 和性质
- KL 散度非负且有限
- 梯度流：只流向学生模型，不流向教师
- Overfit 测试：10 个样本，200 epochs, L_VR 下降
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid.distillation.minilm import MiniLMStyleKD, _extract_values_and_logits


# ======================================================================
# 辅助函数
# ======================================================================

def _make_tiny_gpt2(
    n_layer: int = 2,
    n_embd: int = 32,
    n_head: int = 2,
    vocab_size: int = 100,
) -> GPT2LMHeadModel:
    """创建极小的 GPT-2 模型用于单元测试。"""
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=64,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_embd * 4,
    )
    return GPT2LMHeadModel(config)


# ======================================================================
# TestConstruction — 构造与配置
# ======================================================================

class TestConstruction:
    """MiniLMStyleKD 构造与参数验证。"""

    def test_default_layer_pairs(self) -> None:
        """默认只匹配最后一层: [(11, 3)]。"""
        kd = MiniLMStyleKD()
        assert kd.layer_pairs == [(11, 3)]

    def test_custom_layer_pairs(self) -> None:
        """自定义层配对。"""
        pairs = [(0, 0), (1, 1)]
        kd = MiniLMStyleKD(
            layer_pairs=pairs,
            n_head_teacher=4, n_head_student=2,
        )
        assert kd.layer_pairs == pairs

    def test_head_mapping_4_2(self) -> None:
        """4 teacher → 2 student heads: [[0,1], [2,3]]。"""
        kd = MiniLMStyleKD(
            n_head_teacher=4, n_head_student=2,
            layer_pairs=[(0, 0)],
        )
        assert kd._head_mapping == [[0, 1], [2, 3]]

    def test_head_mapping_12_6(self) -> None:
        """12 teacher → 6 student: 每个映射 2 个。"""
        kd = MiniLMStyleKD(n_head_teacher=12, n_head_student=6)
        assert len(kd._head_mapping) == 6
        for mapping in kd._head_mapping:
            assert len(mapping) == 2

    def test_rejects_empty_layer_pairs(self) -> None:
        """空层配对应报错。"""
        with pytest.raises(AssertionError, match="empty"):
            MiniLMStyleKD(layer_pairs=[])

    def test_no_learnable_parameters(self) -> None:
        """MiniLM 不需要可学习的投影（与 TinyBERT 不同）。"""
        kd = MiniLMStyleKD(layer_pairs=[(0, 0)])
        params = list(kd.parameters())
        assert len(params) == 0, (
            f"MiniLM should have no learnable params, got {len(params)}"
        )


# ======================================================================
# TestValueExtraction — Value vector 提取
# ======================================================================

class TestValueExtraction:
    """验证 Value vector 提取的正确性。"""

    def test_value_shape(self) -> None:
        """提取的 V shape = (B, n_heads, S, head_dim)。"""
        torch.manual_seed(42)
        model = _make_tiny_gpt2(n_layer=2, n_embd=32, n_head=4)
        ids = torch.randint(0, 100, (3, 8))  # (B=3, S=8)

        values, logits = _extract_values_and_logits(
            model, ids, [0, 1], detach=True,
        )

        assert 0 in values and 1 in values
        for layer in [0, 1]:
            v = values[layer]
            assert v.shape == (3, 4, 8, 8), (  # (B, H, S, d_v=32//4=8)
                f"V shape at L{layer}: {v.shape}"
            )

    def test_logits_shape(self) -> None:
        """logits shape = (B, S, vocab_size)。"""
        torch.manual_seed(42)
        model = _make_tiny_gpt2(n_layer=2, n_embd=32, n_head=4)
        ids = torch.randint(0, 100, (3, 8))

        _, logits = _extract_values_and_logits(model, ids, [0], detach=True)
        assert logits.shape == (3, 8, 100)  # (B, S, vocab)

    def test_detach_removes_grad(self) -> None:
        """detach=True 时 V 和 logits 无梯度。"""
        model = _make_tiny_gpt2(n_layer=2, n_embd=32, n_head=2)
        ids = torch.randint(0, 100, (2, 6))

        values, logits = _extract_values_and_logits(
            model, ids, [0], detach=True,
        )
        assert not values[0].requires_grad
        assert not logits.requires_grad

    def test_no_detach_preserves_grad(self) -> None:
        """detach=False 时 V 保留梯度。"""
        model = _make_tiny_gpt2(n_layer=2, n_embd=32, n_head=2)
        model.train()
        ids = torch.randint(0, 100, (2, 6))

        values, logits = _extract_values_and_logits(
            model, ids, [0], detach=False,
        )
        assert values[0].requires_grad or values[0].grad_fn is not None

    def test_value_content_changes_with_input(self) -> None:
        """不同输入产生不同的 V。"""
        torch.manual_seed(42)
        model = _make_tiny_gpt2(n_layer=2, n_embd=32, n_head=2)
        model.eval()

        ids_a = torch.randint(0, 100, (2, 8))
        ids_b = torch.randint(0, 100, (2, 8))

        v_a, _ = _extract_values_and_logits(model, ids_a, [0], detach=True)
        v_b, _ = _extract_values_and_logits(model, ids_b, [0], detach=True)

        assert not torch.allclose(v_a[0], v_b[0], atol=1e-6), (
            "Different inputs should produce different V vectors"
        )


# ======================================================================
# TestValueRelationMatrix — R = V @ V^T / sqrt(d_v)
# ======================================================================

class TestValueRelationMatrix:
    """验证 Value 关系矩阵的性质。"""

    def test_relation_matrix_shape(self) -> None:
        """R 的 shape = (B, S, S)。"""
        B, S, d_v = 2, 5, 8
        V = torch.randn(B, S, d_v)
        R = torch.bmm(V, V.transpose(1, 2)) / math.sqrt(d_v)  # (B, S, S)
        assert R.shape == (B, S, S)

    def test_relation_matrix_symmetric(self) -> None:
        """R = V @ V^T 是对称矩阵。"""
        V = torch.randn(3, 6, 10)
        R = torch.bmm(V, V.transpose(1, 2))  # (3, 6, 6)
        assert torch.allclose(R, R.transpose(1, 2), atol=1e-5)

    def test_kl_on_softmax_R_non_negative(self) -> None:
        """KL(softmax(R_S), softmax(R_T)) >= 0。"""
        R_T = torch.randn(2, 5, 5)
        R_S = torch.randn(2, 5, 5)

        p = F.softmax(R_T, dim=-1)  # (2, 5, 5)
        q = F.log_softmax(R_S, dim=-1)  # (2, 5, 5)

        B, S, _ = R_T.shape
        kl = F.kl_div(
            q.reshape(B * S, S), p.reshape(B * S, S),
            reduction="batchmean",
        )
        assert kl.item() >= -1e-6, f"KL should be non-negative, got {kl.item()}"

    def test_kl_is_zero_when_identical(self) -> None:
        """R_S == R_T 时 KL = 0。"""
        R = torch.randn(2, 5, 5)
        p = F.softmax(R, dim=-1)
        q = F.log_softmax(R, dim=-1)

        B, S, _ = R.shape
        kl = F.kl_div(
            q.reshape(B * S, S), p.reshape(B * S, S),
            reduction="batchmean",
        )
        assert kl.item() < 1e-5, f"KL should be ~0 for identical R, got {kl.item()}"


# ======================================================================
# TestForward — 前向传播基础
# ======================================================================

class TestForward:
    """前向传播和损失计算。"""

    @pytest.fixture
    def models_and_input(self):
        """创建极小的教师和学生模型及输入。"""
        torch.manual_seed(42)
        teacher = _make_tiny_gpt2(n_layer=4, n_embd=32, n_head=4)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        student = _make_tiny_gpt2(n_layer=2, n_embd=16, n_head=2)
        student.train()

        input_ids = torch.randint(0, 100, (4, 16))  # (B=4, S=16)
        layer_pairs = [(1, 0), (3, 1)]

        kd = MiniLMStyleKD(
            n_head_teacher=4, n_head_student=2,
            layer_pairs=layer_pairs,
        )
        return teacher, student, input_ids, kd

    def test_forward_returns_dict(self, models_and_input) -> None:
        """forward 返回包含所有损失键的字典。"""
        teacher, student, ids, kd = models_and_input
        result = kd(teacher, student, ids)
        assert set(result.keys()) == {"loss", "loss_vr", "loss_kl"}

    def test_all_losses_finite(self, models_and_input) -> None:
        """所有损失值应有限。"""
        teacher, student, ids, kd = models_and_input
        result = kd(teacher, student, ids)
        for key, val in result.items():
            assert val.isfinite(), f"{key} is not finite: {val.item()}"

    def test_loss_vr_non_negative(self, models_and_input) -> None:
        """L_VR 应非负（KL 散度）。"""
        teacher, student, ids, kd = models_and_input
        result = kd(teacher, student, ids)
        assert result["loss_vr"].item() >= -1e-6, (
            f"L_VR should be non-negative, got {result['loss_vr'].item()}"
        )

    def test_total_loss_is_weighted_sum(self, models_and_input) -> None:
        """总损失 = α·L_VR + β·L_KL。"""
        teacher, student, ids, kd = models_and_input
        kd.alpha, kd.beta = 2.0, 0.5
        result = kd(teacher, student, ids)

        expected = 2.0 * result["loss_vr"] + 0.5 * result["loss_kl"]
        assert torch.allclose(result["loss"].detach(), expected, atol=1e-4), (
            f"Total {result['loss'].item():.4f} != expected {expected.item():.4f}"
        )

    def test_zero_alpha_eliminates_vr(self, models_and_input) -> None:
        """alpha=0 时 L_VR 不影响总损失。"""
        teacher, student, ids, kd = models_and_input
        kd.alpha = 0.0
        result = kd(teacher, student, ids)
        expected = kd.beta * result["loss_kl"]
        assert torch.allclose(result["loss"].detach(), expected, atol=1e-4)


# ======================================================================
# TestGradientFlow
# ======================================================================

class TestGradientFlow:
    """验证梯度流向。"""

    def test_gradient_flows_to_student(self) -> None:
        """梯度应流回学生模型参数。"""
        torch.manual_seed(42)
        teacher = _make_tiny_gpt2(n_layer=4, n_embd=32, n_head=4)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        student = _make_tiny_gpt2(n_layer=2, n_embd=16, n_head=2)
        student.train()

        kd = MiniLMStyleKD(
            n_head_teacher=4, n_head_student=2,
            layer_pairs=[(1, 0), (3, 1)],
        )

        ids = torch.randint(0, 100, (4, 16))
        student.zero_grad()
        result = kd(teacher, student, ids)
        result["loss"].backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in student.parameters()
        )
        assert has_grad, "No gradient flowed to student parameters"

    def test_no_gradient_to_teacher(self) -> None:
        """教师模型不应有梯度。"""
        torch.manual_seed(42)
        teacher = _make_tiny_gpt2(n_layer=4, n_embd=32, n_head=4)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        student = _make_tiny_gpt2(n_layer=2, n_embd=16, n_head=2)
        student.train()

        kd = MiniLMStyleKD(
            n_head_teacher=4, n_head_student=2,
            layer_pairs=[(1, 0), (3, 1)],
        )

        ids = torch.randint(0, 100, (4, 16))
        result = kd(teacher, student, ids)
        result["loss"].backward()

        for p in teacher.parameters():
            assert p.grad is None, "Teacher should not receive gradients"


# ======================================================================
# TestOverfit — 在小数据上 Overfit
# ======================================================================

@pytest.mark.slow
class TestOverfit:
    """用小型模型验证 overfit 能力（需要较长时间）。"""

    def test_overfit_vr_loss(self) -> None:
        """10 个样本，200 epochs：L_VR 应显著下降。"""
        torch.manual_seed(42)

        teacher = _make_tiny_gpt2(n_layer=4, n_embd=64, n_head=4)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        student = _make_tiny_gpt2(n_layer=2, n_embd=32, n_head=2)
        student.train()

        input_ids = torch.randint(0, 100, (10, 16))  # (10, 16)

        kd = MiniLMStyleKD(
            n_head_teacher=4, n_head_student=2,
            layer_pairs=[(1, 0), (3, 1)],
            alpha=1.0, beta=0.0,  # 只训练 VR
        )

        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

        # 记录初始损失
        with torch.no_grad():
            init_result = kd(teacher, student, input_ids)
        init_vr = init_result["loss_vr"].item()

        print(f"\n=== MiniLM Overfit Test (VR only) ===")
        print(f"Initial: L_VR={init_vr:.6f}")

        for epoch in range(200):
            optimizer.zero_grad()
            result = kd(teacher, student, input_ids)
            result["loss"].backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch+1:3d}: "
                    f"L_VR={result['loss_vr'].item():.6f}, "
                    f"total={result['loss'].item():.6f}"
                )

        final_vr = result["loss_vr"].item()
        print(f"Final: L_VR={final_vr:.6f}")
        print(f"Reduction: {init_vr/max(final_vr,1e-10):.1f}x")

        # L_VR 应显著下降（至少降低 30%）
        assert final_vr < init_vr * 0.7, (
            f"L_VR did not decrease enough: "
            f"{init_vr:.6f} → {final_vr:.6f}"
        )

    def test_overfit_full(self) -> None:
        """包含 KL 损失的完整 overfit: 总损失应下降。"""
        torch.manual_seed(42)

        teacher = _make_tiny_gpt2(n_layer=4, n_embd=64, n_head=4)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        student = _make_tiny_gpt2(n_layer=2, n_embd=32, n_head=2)
        student.train()

        input_ids = torch.randint(0, 100, (10, 16))

        kd = MiniLMStyleKD(
            n_head_teacher=4, n_head_student=2,
            layer_pairs=[(1, 0), (3, 1)],
            alpha=1.0, beta=1.0,
        )

        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

        with torch.no_grad():
            init_loss = kd(teacher, student, input_ids)["loss"].item()

        print(f"\n=== MiniLM Overfit Test (full) ===")
        print(f"Initial total loss: {init_loss:.4f}")

        for epoch in range(200):
            optimizer.zero_grad()
            result = kd(teacher, student, input_ids)
            result["loss"].backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch+1:3d}: "
                    f"L_VR={result['loss_vr'].item():.6f}, "
                    f"L_KL={result['loss_kl'].item():.4f}, "
                    f"total={result['loss'].item():.4f}"
                )

        final_loss = result["loss"].item()
        print(f"Final total: {final_loss:.4f} "
              f"({init_loss/max(final_loss,1e-10):.1f}x reduction)")

        assert final_loss < init_loss * 0.5, (
            f"Total loss did not decrease: {init_loss:.4f} → {final_loss:.4f}"
        )
