"""Informed FitNets 消融基线单元测试。

测试覆盖：
- 与 RCID 使用完全相同的检查点和 W 矩阵
- 匹配目标差异：h^T_clean（非 d^T = h^T_clean - h^T_corrupt）
- 只需 clean_input（不需 corrupt_input）
- W 是冻结 buffer，不参与梯度更新
- 梯度只流向学生模型
- Overfit 测试
- 与 RCID 的对比：相同设置下损失值不同
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid.distillation.informed_fitnets import InformedFitNetsLoss
from rcid.distillation.rcid_loss import RCIDLoss


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


def _extract_teacher_clean_residuals(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,             # (B, seq_len)
    checkpoints: list[tuple[int, int]],  # [(layer, token_pos), ...]
) -> dict[tuple[int, int], torch.Tensor]:
    """提取教师在 clean 输入上各检查点的残差流（无梯度）。"""
    model.eval()
    layers_needed = sorted(set(l for l, t in checkpoints))
    residuals: dict[tuple[int, int], torch.Tensor] = {}

    with torch.no_grad():
        storage: dict[int, torch.Tensor] = {}
        handles = []
        for layer in layers_needed:
            def _make_hook(l: int):  # noqa: E741
                def hook(mod, inp, out):
                    storage[l] = out[0]
                return hook
            h = model.transformer.h[layer].register_forward_hook(
                _make_hook(layer),
            )
            handles.append(h)

        model(input_ids)
        for h in handles:
            h.remove()

        for l, t in checkpoints:
            residuals[(l, t)] = storage[l][:, t, :]  # (B, d_model)

    return residuals


def _extract_teacher_imprints(
    model: GPT2LMHeadModel,
    clean_ids: torch.Tensor,
    corrupt_ids: torch.Tensor,
    checkpoints: list[tuple[int, int]],
) -> dict[tuple[int, int], torch.Tensor]:
    """提取教师因果痕迹 d^T = h_clean - h_corrupt（用于 RCID 对比）。"""
    clean_res = _extract_teacher_clean_residuals(model, clean_ids, checkpoints)
    corrupt_res = _extract_teacher_clean_residuals(
        model, corrupt_ids, checkpoints,
    )
    return {
        key: clean_res[key] - corrupt_res[key]
        for key in clean_res
    }


# ======================================================================
# TestConstruction — 构造与参数验证
# ======================================================================

class TestConstruction:
    """InformedFitNetsLoss 构造与配置。"""

    def test_creation(self) -> None:
        """基本创建。"""
        loss_fn = InformedFitNetsLoss(
            W=torch.eye(32),
            checkpoints=[(0, 5), (1, 10)],
            layer_mapping={0: 0, 1: 1},
        )
        assert len(loss_fn.checkpoints) == 2

    def test_W_is_buffer(self) -> None:
        """W 应注册为 buffer 而非 parameter。"""
        loss_fn = InformedFitNetsLoss(
            W=torch.eye(32),
            checkpoints=[(0, 5)],
            layer_mapping={0: 0},
        )
        buffer_names = [n for n, _ in loss_fn.named_buffers()]
        assert "W" in buffer_names
        param_names = [n for n, _ in loss_fn.named_parameters()]
        assert "W" not in param_names

    def test_W_not_requires_grad(self) -> None:
        """W 不应 requires_grad。"""
        loss_fn = InformedFitNetsLoss(
            W=torch.eye(32),
            checkpoints=[(0, 5)],
            layer_mapping={0: 0},
        )
        assert not loss_fn.W.requires_grad

    def test_W_is_cloned(self) -> None:
        """传入的 W 被 clone，修改原始不影响 loss_fn。"""
        W_orig = torch.eye(32)
        loss_fn = InformedFitNetsLoss(
            W=W_orig,
            checkpoints=[(0, 5)],
            layer_mapping={0: 0},
        )
        W_orig.fill_(0.0)
        assert loss_fn.W.abs().sum() > 0

    def test_rejects_empty_checkpoints(self) -> None:
        """空检查点应报错。"""
        with pytest.raises(AssertionError, match="empty"):
            InformedFitNetsLoss(
                W=torch.eye(32),
                checkpoints=[],
                layer_mapping={},
            )

    def test_rejects_1d_W(self) -> None:
        """W 不是 2D 应报错。"""
        with pytest.raises(AssertionError, match="2D"):
            InformedFitNetsLoss(
                W=torch.randn(32),
                checkpoints=[(0, 5)],
                layer_mapping={0: 0},
            )

    def test_rejects_missing_layer_mapping(self) -> None:
        """layer_mapping 缺层应报错。"""
        with pytest.raises(AssertionError, match="not in layer_mapping"):
            InformedFitNetsLoss(
                W=torch.eye(32),
                checkpoints=[(0, 5), (5, 10)],
                layer_mapping={0: 0},
            )


# ======================================================================
# TestSameCheckpointsAsRCID — 与 RCID 的一致性
# ======================================================================

class TestSameCheckpointsAsRCID:
    """验证 Informed FitNets 和 RCID 使用相同的检查点和 W。"""

    def test_same_checkpoints_and_W(self) -> None:
        """构造相同配置的 RCID 和 IF，验证参数一致。"""
        W = torch.eye(32)
        ckpts = [(0, 5), (1, 10)]
        mapping = {0: 0, 1: 1}

        rcid = RCIDLoss(W=W, checkpoints=ckpts, layer_mapping=mapping)
        ifn = InformedFitNetsLoss(
            W=W, checkpoints=ckpts, layer_mapping=mapping,
        )

        assert rcid.checkpoints == ifn.checkpoints
        assert rcid.layer_mapping == ifn.layer_mapping
        assert torch.allclose(rcid.W, ifn.W)

    def test_different_loss_values(self) -> None:
        """不同教师和学生模型下，RCID 和 IF 损失值应不同。

        RCID 匹配因果差值 d^T，IF 匹配完整表示 h^T。
        使用不同的教师和学生模型确保损失非零。
        """
        d = 32
        torch.manual_seed(42)
        teacher = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)
        teacher.eval()

        torch.manual_seed(99)
        student = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)
        student.eval()

        W = torch.eye(d)
        ckpts = [(0, 5), (1, 10)]
        mapping = {0: 0, 1: 1}

        torch.manual_seed(42)
        clean = torch.randint(0, 100, (4, 16))
        corrupt = torch.randint(0, 100, (4, 16))

        # RCID 需要因果痕迹
        teacher_imprints = _extract_teacher_imprints(
            teacher, clean, corrupt, ckpts,
        )
        rcid = RCIDLoss(W=W, checkpoints=ckpts, layer_mapping=mapping)
        rcid_loss = rcid(teacher_imprints, student, clean, corrupt)

        # IF 需要 clean 表示
        teacher_clean = _extract_teacher_clean_residuals(
            teacher, clean, ckpts,
        )
        ifn = InformedFitNetsLoss(
            W=W, checkpoints=ckpts, layer_mapping=mapping,
        )
        ifn_loss = ifn(teacher_clean, student, clean)

        # 两者都非零
        assert rcid_loss.item() > 1e-6, (
            f"RCID loss should be > 0, got {rcid_loss.item():.6f}"
        )
        assert ifn_loss.item() > 1e-6, (
            f"IF loss should be > 0, got {ifn_loss.item():.6f}"
        )

        # 两者使用不同目标，损失值应不同
        assert abs(rcid_loss.item() - ifn_loss.item()) > 1e-6, (
            f"RCID ({rcid_loss.item():.6f}) and IF ({ifn_loss.item():.6f}) "
            f"should differ (different matching targets)"
        )


# ======================================================================
# TestForward — 前向传播
# ======================================================================

class TestForward:
    """前向传播和损失计算。"""

    def test_loss_is_finite(self) -> None:
        """损失应有限。"""
        d = 32
        model = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)

        torch.manual_seed(42)
        clean = torch.randint(0, 100, (4, 16))
        ckpts = [(0, 5), (1, 10)]
        mapping = {0: 0, 1: 1}

        teacher_clean = _extract_teacher_clean_residuals(
            model, clean, ckpts,
        )
        loss_fn = InformedFitNetsLoss(
            W=torch.eye(d), checkpoints=ckpts, layer_mapping=mapping,
        )

        model.train()
        loss = loss_fn(teacher_clean, model, clean)
        assert loss.isfinite(), f"Loss not finite: {loss.item()}"

    def test_loss_non_negative(self) -> None:
        """MSE 损失应非负。"""
        d = 32
        model = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)

        torch.manual_seed(42)
        clean = torch.randint(0, 100, (4, 16))
        ckpts = [(0, 5)]
        mapping = {0: 0}

        teacher_clean = _extract_teacher_clean_residuals(
            model, clean, ckpts,
        )
        loss_fn = InformedFitNetsLoss(
            W=torch.eye(d), checkpoints=ckpts, layer_mapping=mapping,
        )

        model.train()
        loss = loss_fn(teacher_clean, model, clean)
        assert loss.item() >= 0.0

    def test_same_model_identity_W_zero_loss(self) -> None:
        """教师=学生，W=I 时，损失应为零。

        同一模型 eval 模式下两次前向输出完全一致，
        W=I 则对齐后相同，MSE=0。
        """
        d = 32
        model = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)

        torch.manual_seed(42)
        clean = torch.randint(0, 100, (4, 16))
        ckpts = [(0, 5), (1, 10)]
        mapping = {0: 0, 1: 1}

        teacher_clean = _extract_teacher_clean_residuals(
            model, clean, ckpts,
        )
        loss_fn = InformedFitNetsLoss(
            W=torch.eye(d), checkpoints=ckpts, layer_mapping=mapping,
        )

        model.eval()
        loss = loss_fn(teacher_clean, model, clean)
        assert loss.item() < 1e-4, (
            f"Loss should be ~0 when teacher=student and W=I, "
            f"got {loss.item():.6f}"
        )

    def test_random_W_positive_loss(self) -> None:
        """随机 W 时损失应 > 0。"""
        d = 32
        model = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)

        torch.manual_seed(42)
        clean = torch.randint(0, 100, (4, 16))
        ckpts = [(0, 5)]
        mapping = {0: 0}

        teacher_clean = _extract_teacher_clean_residuals(
            model, clean, ckpts,
        )

        gen = torch.Generator().manual_seed(99)
        A = torch.randn(d, d, generator=gen)
        W_rand, _ = torch.linalg.qr(A)

        loss_fn = InformedFitNetsLoss(
            W=W_rand, checkpoints=ckpts, layer_mapping=mapping,
        )

        model.train()
        loss = loss_fn(teacher_clean, model, clean)
        assert loss.item() > 0.0

    def test_no_corrupt_input_needed(self) -> None:
        """IF 只需 clean_input，不接受 corrupt_input。"""
        d = 32
        model = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)
        clean = torch.randint(0, 100, (4, 16))
        ckpts = [(0, 5)]
        mapping = {0: 0}

        teacher_clean = _extract_teacher_clean_residuals(
            model, clean, ckpts,
        )
        loss_fn = InformedFitNetsLoss(
            W=torch.eye(d), checkpoints=ckpts, layer_mapping=mapping,
        )

        # forward 签名只接受 3 个参数 (teacher_res, model, clean)
        model.eval()
        loss = loss_fn(teacher_clean, model, clean)
        assert loss.isfinite()

    def test_rejects_missing_teacher_residual(self) -> None:
        """缺少教师残差应报错。"""
        d = 32
        model = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)
        loss_fn = InformedFitNetsLoss(
            W=torch.eye(d),
            checkpoints=[(0, 5), (1, 10)],
            layer_mapping={0: 0, 1: 1},
        )

        # 只提供一个检查点
        teacher_res = {(0, 5): torch.randn(4, d)}

        with pytest.raises(AssertionError, match="Missing teacher residual"):
            loss_fn(teacher_res, model, torch.randint(0, 100, (4, 16)))


# ======================================================================
# TestNonSquareW — d_T ≠ d_S
# ======================================================================

class TestNonSquareW:
    """教师和学生维度不同时的测试。"""

    def test_different_dimensions(self) -> None:
        """d_T=64, d_S=32 能正确计算。"""
        d_t, d_s = 64, 32
        teacher = _make_tiny_gpt2(n_layer=2, n_embd=d_t, n_head=2)
        student = _make_tiny_gpt2(n_layer=2, n_embd=d_s, n_head=2)

        torch.manual_seed(42)
        clean = torch.randint(0, 100, (4, 16))
        ckpts = [(0, 5), (1, 10)]
        mapping = {0: 0, 1: 1}

        teacher_clean = _extract_teacher_clean_residuals(
            teacher, clean, ckpts,
        )

        gen = torch.Generator().manual_seed(42)
        A = torch.randn(d_t, d_s, generator=gen)
        W, _ = torch.linalg.qr(A)

        loss_fn = InformedFitNetsLoss(
            W=W, checkpoints=ckpts, layer_mapping=mapping,
        )

        student.train()
        loss = loss_fn(teacher_clean, student, clean)
        assert loss.isfinite()
        assert loss.item() >= 0.0


# ======================================================================
# TestGradientFlow
# ======================================================================

class TestGradientFlow:
    """验证梯度流向。"""

    def test_gradient_flows_to_student(self) -> None:
        """梯度应流回学生模型。"""
        d = 32
        teacher = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)
        student = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)

        torch.manual_seed(42)
        clean = torch.randint(0, 100, (4, 16))
        ckpts = [(0, 5)]
        mapping = {0: 0}

        teacher_clean = _extract_teacher_clean_residuals(
            teacher, clean, ckpts,
        )

        loss_fn = InformedFitNetsLoss(
            W=torch.eye(d), checkpoints=ckpts, layer_mapping=mapping,
        )

        student.train()
        student.zero_grad()
        loss = loss_fn(teacher_clean, student, clean)
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in student.parameters()
        )
        assert has_grad, "No gradient to student"

    def test_no_gradient_to_W(self) -> None:
        """W 是 buffer，不应有梯度。"""
        d = 32
        student = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)

        torch.manual_seed(42)
        clean = torch.randint(0, 100, (4, 16))
        ckpts = [(0, 5)]
        mapping = {0: 0}

        teacher = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)
        teacher_clean = _extract_teacher_clean_residuals(
            teacher, clean, ckpts,
        )

        loss_fn = InformedFitNetsLoss(
            W=torch.eye(d), checkpoints=ckpts, layer_mapping=mapping,
        )

        student.train()
        loss = loss_fn(teacher_clean, student, clean)
        loss.backward()

        assert loss_fn.W.grad is None, "W should not have gradient"


# ======================================================================
# TestOverfit
# ======================================================================

@pytest.mark.slow
class TestOverfit:
    """用小型模型验证 overfit 能力。"""

    def test_overfit_informed_fitnets(self) -> None:
        """10 个样本, 200 epochs: 损失应显著下降。"""
        torch.manual_seed(42)

        d_t, d_s = 64, 32
        teacher = _make_tiny_gpt2(n_layer=2, n_embd=d_t, n_head=2)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        student = _make_tiny_gpt2(n_layer=2, n_embd=d_s, n_head=2)
        student.train()

        clean = torch.randint(0, 100, (10, 16))
        ckpts = [(0, 5), (1, 10)]
        mapping = {0: 0, 1: 1}

        teacher_clean = _extract_teacher_clean_residuals(
            teacher, clean, ckpts,
        )

        # 用随机正交 W 模拟 Procrustes 结果
        gen = torch.Generator().manual_seed(42)
        A = torch.randn(d_t, d_s, generator=gen)
        W, _ = torch.linalg.qr(A)

        loss_fn = InformedFitNetsLoss(
            W=W, checkpoints=ckpts, layer_mapping=mapping,
        )

        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

        with torch.no_grad():
            init_loss = loss_fn(teacher_clean, student, clean).item()

        print(f"\n=== Informed FitNets Overfit Test ===")
        print(f"Initial loss: {init_loss:.4f}")

        for epoch in range(200):
            optimizer.zero_grad()
            loss = loss_fn(teacher_clean, student, clean)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:3d}: loss={loss.item():.6f}")

        final_loss = loss.item()
        print(f"Final: {final_loss:.6f} "
              f"({init_loss/max(final_loss,1e-10):.1f}x reduction)")

        assert final_loss < init_loss * 0.5, (
            f"Loss did not decrease enough: "
            f"{init_loss:.4f} → {final_loss:.4f}"
        )
