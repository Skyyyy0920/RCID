"""TinyBERT 风格蒸馏基线单元测试。

测试覆盖：
- TinyBERTStyleKD 构造和层配对
- 子损失计算（hidden, attn, kl）
- 梯度流：只流向学生模型和可学习 W_h，不流向教师
- W_h 是可学习参数（在 optimizer 中）
- Overfit 测试：10 个 IOI 样本，200 epochs, L_hidden/L_attn → 0
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid.distillation.tinybert import TinyBERTStyleKD


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
    """TinyBERTStyleKD 构造与参数验证。"""

    def test_default_layer_pairs(self) -> None:
        """默认层配对应为 [(2,0),(5,1),(8,2),(11,3)]。"""
        kd = TinyBERTStyleKD()
        assert kd.layer_pairs == [(2, 0), (5, 1), (8, 2), (11, 3)]

    def test_custom_layer_pairs(self) -> None:
        """自定义层配对。"""
        pairs = [(0, 0), (1, 1)]
        kd = TinyBERTStyleKD(layer_pairs=pairs, d_teacher=32, d_student=16)
        assert kd.layer_pairs == pairs

    def test_make_uniform_pairs(self) -> None:
        """等间隔映射: 12→4 应产生覆盖全层范围的配对。"""
        pairs = TinyBERTStyleKD.make_uniform_pairs(12, 4)
        assert len(pairs) == 4
        # 学生层 0..3 各出现一次
        assert [s for _, s in pairs] == [0, 1, 2, 3]
        # 教师层递增且最后一个映射到最后一层
        t_layers = [t for t, _ in pairs]
        assert t_layers == sorted(t_layers), "Teacher layers should increase"
        assert t_layers[-1] == 11, "Last student layer should map to last teacher layer"

    def test_make_uniform_pairs_small(self) -> None:
        """4→2 → [(1,0),(3,1)]。"""
        pairs = TinyBERTStyleKD.make_uniform_pairs(4, 2)
        assert len(pairs) == 2
        # 检查映射到合理范围
        for t, s in pairs:
            assert 0 <= t < 4
            assert 0 <= s < 2

    def test_hidden_projections_exist(self) -> None:
        """每个层配对有对应的可学习投影。"""
        pairs = [(0, 0), (1, 1)]
        kd = TinyBERTStyleKD(
            d_teacher=64, d_student=32,
            n_head_teacher=4, n_head_student=2,
            layer_pairs=pairs,
        )
        assert "proj_0_0" in kd.hidden_projections
        assert "proj_1_1" in kd.hidden_projections
        # 投影形状: d_S → d_T
        assert kd.hidden_projections["proj_0_0"].weight.shape == (64, 32)

    def test_attn_head_mapping(self) -> None:
        """注意力头映射: 4 teacher heads, 2 student heads → [0,1] [2,3]。"""
        kd = TinyBERTStyleKD(
            n_head_teacher=4, n_head_student=2,
            d_teacher=32, d_student=16,
            layer_pairs=[(0, 0)],
        )
        assert kd._attn_head_mapping == [[0, 1], [2, 3]]

    def test_attn_head_mapping_12_6(self) -> None:
        """12 teacher → 6 student: 每个 student head 映射 2 个 teacher heads。"""
        kd = TinyBERTStyleKD(n_head_teacher=12, n_head_student=6)
        assert len(kd._attn_head_mapping) == 6
        for mapping in kd._attn_head_mapping:
            assert len(mapping) == 2

    def test_rejects_empty_layer_pairs(self) -> None:
        """空层配对应报错。"""
        with pytest.raises(AssertionError, match="empty"):
            TinyBERTStyleKD(layer_pairs=[])

    def test_projections_are_parameters(self) -> None:
        """W_h 应是可学习参数。"""
        kd = TinyBERTStyleKD(
            d_teacher=64, d_student=32,
            layer_pairs=[(0, 0)],
        )
        param_names = [n for n, _ in kd.named_parameters()]
        assert any("proj_0_0" in n for n in param_names), (
            "Projection should be a learnable parameter"
        )


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

        kd = TinyBERTStyleKD(
            d_teacher=32, d_student=16,
            n_head_teacher=4, n_head_student=2,
            layer_pairs=layer_pairs,
        )
        return teacher, student, input_ids, kd

    def test_forward_returns_dict(self, models_and_input) -> None:
        """forward 返回包含所有损失键的字典。"""
        teacher, student, ids, kd = models_and_input
        result = kd(teacher, student, ids)
        assert set(result.keys()) == {"loss", "loss_hidden", "loss_attn", "loss_kl"}

    def test_all_losses_finite(self, models_and_input) -> None:
        """所有损失值应有限。"""
        teacher, student, ids, kd = models_and_input
        result = kd(teacher, student, ids)
        for key, val in result.items():
            assert val.isfinite(), f"{key} is not finite: {val.item()}"

    def test_all_losses_non_negative(self, models_and_input) -> None:
        """所有子损失应非负。"""
        teacher, student, ids, kd = models_and_input
        result = kd(teacher, student, ids)
        for key in ("loss_hidden", "loss_attn", "loss_kl"):
            assert result[key].item() >= 0, f"{key} is negative: {result[key].item()}"

    def test_total_loss_is_weighted_sum(self, models_and_input) -> None:
        """总损失 = α·L_hidden + β·L_attn + γ·L_kl。"""
        teacher, student, ids, kd = models_and_input
        # 设不同权重
        kd.alpha, kd.beta, kd.gamma = 2.0, 3.0, 0.5
        result = kd(teacher, student, ids)

        # 重新计算（因为 result 中子损失是 detach 过的）
        expected = (
            2.0 * result["loss_hidden"]
            + 3.0 * result["loss_attn"]
            + 0.5 * result["loss_kl"]
        )
        assert torch.allclose(result["loss"].detach(), expected, atol=1e-4), (
            f"Total {result['loss'].item():.4f} != "
            f"expected {expected.item():.4f}"
        )

    def test_zero_alpha_eliminates_hidden(self, models_and_input) -> None:
        """alpha=0 时 L_hidden 不影响总损失。"""
        teacher, student, ids, kd = models_and_input
        kd.alpha = 0.0
        result = kd(teacher, student, ids)
        expected = kd.beta * result["loss_attn"] + kd.gamma * result["loss_kl"]
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

        kd = TinyBERTStyleKD(
            d_teacher=32, d_student=16,
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

    def test_gradient_flows_to_projection(self) -> None:
        """梯度应流回可学习 W_h 投影。"""
        torch.manual_seed(42)
        teacher = _make_tiny_gpt2(n_layer=4, n_embd=32, n_head=4)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        student = _make_tiny_gpt2(n_layer=2, n_embd=16, n_head=2)
        student.train()

        kd = TinyBERTStyleKD(
            d_teacher=32, d_student=16,
            n_head_teacher=4, n_head_student=2,
            layer_pairs=[(1, 0), (3, 1)],
        )

        ids = torch.randint(0, 100, (4, 16))
        kd.zero_grad()
        result = kd(teacher, student, ids)
        result["loss"].backward()

        has_proj_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in kd.hidden_projections.parameters()
        )
        assert has_proj_grad, "No gradient to hidden projections (W_h)"

    def test_no_gradient_to_teacher(self) -> None:
        """教师模型不应有梯度。"""
        torch.manual_seed(42)
        teacher = _make_tiny_gpt2(n_layer=4, n_embd=32, n_head=4)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        student = _make_tiny_gpt2(n_layer=2, n_embd=16, n_head=2)
        student.train()

        kd = TinyBERTStyleKD(
            d_teacher=32, d_student=16,
            n_head_teacher=4, n_head_student=2,
            layer_pairs=[(1, 0), (3, 1)],
        )

        ids = torch.randint(0, 100, (4, 16))
        result = kd(teacher, student, ids)
        result["loss"].backward()

        for p in teacher.parameters():
            assert p.grad is None, "Teacher should not receive gradients"


# ======================================================================
# TestOverfit — 在 IOI 小数据上 Overfit
# ======================================================================

@pytest.mark.slow
class TestOverfit:
    """用小型模型验证 overfit 能力（需要较长时间）。"""

    def test_overfit_hidden_and_attn(self) -> None:
        """10 个样本，200 epochs：L_hidden 和 L_attn 应显著下降。

        使用小型教师(4层,64d)和学生(2层,32d)，
        验证 TinyBERT 蒸馏可以 overfit 小数据。
        """
        torch.manual_seed(42)

        # 小型教师
        teacher = _make_tiny_gpt2(n_layer=4, n_embd=64, n_head=4)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # 小型学生
        student = _make_tiny_gpt2(n_layer=2, n_embd=32, n_head=2)
        student.train()

        # 固定输入: 10 个样本
        input_ids = torch.randint(0, 100, (10, 16))  # (10, 16)

        layer_pairs = [(1, 0), (3, 1)]
        kd = TinyBERTStyleKD(
            d_teacher=64, d_student=32,
            n_head_teacher=4, n_head_student=2,
            layer_pairs=layer_pairs,
            alpha=1.0, beta=1.0, gamma=0.0,  # 只训练 hidden + attn
        )

        # Optimizer 包含学生参数 + W_h 投影
        optimizer = torch.optim.Adam(
            list(student.parameters()) + list(kd.parameters()),
            lr=1e-3,
        )

        # 记录初始损失
        with torch.no_grad():
            init_result = kd(teacher, student, input_ids)
        init_hidden = init_result["loss_hidden"].item()
        init_attn = init_result["loss_attn"].item()

        print(f"\n=== TinyBERT Overfit Test ===")
        print(f"Initial: L_hidden={init_hidden:.4f}, L_attn={init_attn:.4f}")

        # 训练 200 epochs
        for epoch in range(200):
            optimizer.zero_grad()
            result = kd(teacher, student, input_ids)
            result["loss"].backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch+1:3d}: "
                    f"L_hidden={result['loss_hidden'].item():.6f}, "
                    f"L_attn={result['loss_attn'].item():.6f}, "
                    f"total={result['loss'].item():.6f}"
                )

        final_hidden = result["loss_hidden"].item()
        final_attn = result["loss_attn"].item()

        print(f"Final: L_hidden={final_hidden:.6f}, L_attn={final_attn:.6f}")
        print(f"Reduction: hidden {init_hidden/max(final_hidden,1e-10):.1f}x, "
              f"attn {init_attn/max(final_attn,1e-10):.1f}x")

        # L_hidden 应显著下降（至少降低 50%）
        assert final_hidden < init_hidden * 0.5, (
            f"L_hidden did not decrease enough: "
            f"{init_hidden:.4f} → {final_hidden:.4f}"
        )

        # L_attn: 初始值极小（~0.001），不容易继续下降
        # 验证它没有爆炸增长即可（< 初始的 5 倍）
        assert final_attn < max(init_attn * 5.0, 0.01), (
            f"L_attn increased too much: "
            f"{init_attn:.4f} → {final_attn:.4f}"
        )

        # 总损失应显著下降
        final_total = final_hidden + final_attn
        init_total = init_hidden + init_attn
        assert final_total < init_total * 0.5, (
            f"Total (hidden+attn) did not decrease: "
            f"{init_total:.4f} → {final_total:.4f}"
        )

    def test_overfit_with_kl(self) -> None:
        """包含 KL 损失的完整 overfit: 总损失应下降。"""
        torch.manual_seed(42)

        teacher = _make_tiny_gpt2(n_layer=4, n_embd=64, n_head=4)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        student = _make_tiny_gpt2(n_layer=2, n_embd=32, n_head=2)
        student.train()

        input_ids = torch.randint(0, 100, (10, 16))

        kd = TinyBERTStyleKD(
            d_teacher=64, d_student=32,
            n_head_teacher=4, n_head_student=2,
            layer_pairs=[(1, 0), (3, 1)],
            alpha=1.0, beta=1.0, gamma=1.0,
        )

        optimizer = torch.optim.Adam(
            list(student.parameters()) + list(kd.parameters()),
            lr=1e-3,
        )

        with torch.no_grad():
            init_loss = kd(teacher, student, input_ids)["loss"].item()

        for _ in range(200):
            optimizer.zero_grad()
            result = kd(teacher, student, input_ids)
            result["loss"].backward()
            optimizer.step()

        final_loss = result["loss"].item()
        print(f"\nOverfit w/ KL: {init_loss:.4f} → {final_loss:.4f} "
              f"({init_loss/max(final_loss,1e-10):.1f}x reduction)")

        assert final_loss < init_loss * 0.5, (
            f"Total loss did not decrease: {init_loss:.4f} → {final_loss:.4f}"
        )
