"""RCID 损失函数单元测试。

测试覆盖（参考 CLAUDE.md 3.4 测试要求）：
- 教师=学生，W=I 时，损失应为零
- 随机 W 时损失应 > 0
- 梯度应只流过学生模型，不流过教师痕迹和 W
- 损失值范围合理性
- d_T ≠ d_S（非方阵 W）情况
- 输入校验
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from rcid.distillation.rcid_loss import RCIDLoss


# ======================================================================
# 辅助函数：创建小型 GPT-2 模型用于测试
# ======================================================================

def _make_tiny_gpt2(
    n_layer: int = 2,
    n_embd: int = 32,
    n_head: int = 2,
    vocab_size: int = 100,
) -> GPT2LMHeadModel:
    """创建一个极小的 GPT-2 模型用于单元测试。"""
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=64,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_embd * 4,
    )
    model = GPT2LMHeadModel(config)
    model.train()
    return model


def _make_teacher_imprints(
    model: GPT2LMHeadModel,
    clean_ids: torch.Tensor,    # (B, seq_len)
    corrupt_ids: torch.Tensor,  # (B, seq_len)
    checkpoints: list[tuple[int, int]],
) -> dict[tuple[int, int], torch.Tensor]:
    """用教师模型提取预计算痕迹（无梯度）。"""
    model.eval()
    imprints: dict[tuple[int, int], torch.Tensor] = {}

    layers_needed = sorted(set(l for l, t in checkpoints))

    with torch.no_grad():
        for layer in layers_needed:
            # 提取 clean 残差流
            storage_clean: dict[int, torch.Tensor] = {}

            def _make_hook(l: int, s: dict):
                def hook(mod, inp, out):
                    s[l] = out[0]
                return hook

            handle = model.transformer.h[layer].register_forward_hook(
                _make_hook(layer, storage_clean)
            )
            model(clean_ids)
            handle.remove()

            # 提取 corrupt 残差流
            storage_corrupt: dict[int, torch.Tensor] = {}
            handle = model.transformer.h[layer].register_forward_hook(
                _make_hook(layer, storage_corrupt)
            )
            model(corrupt_ids)
            handle.remove()

            # 计算差值
            positions = [t for l, t in checkpoints if l == layer]
            for t in positions:
                d = (
                    storage_clean[layer][:, t, :]
                    - storage_corrupt[layer][:, t, :]
                )  # (B, d_model)
                imprints[(layer, t)] = d

    return imprints


# ======================================================================
# 核心功能测试
# ======================================================================

class TestRCIDLossCore:
    """RCIDLoss 核心功能测试。"""

    def test_same_model_identity_W_zero_loss(self) -> None:
        """教师=学生，W=I 时，损失应为零。

        当同一个模型同时作为教师和学生，且 W 是单位矩阵时，
        d^T 和 d^S 完全相同，归一化后差值为零。

        注意：必须使用 eval 模式以禁用 dropout，
        否则 train 模式下的随机 dropout 会导致两次前向输出不同。
        """
        d = 32
        model = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)

        torch.manual_seed(42)
        clean_ids = torch.randint(0, 100, (4, 16))    # (B=4, seq_len=16)
        corrupt_ids = torch.randint(0, 100, (4, 16))   # (B=4, seq_len=16)

        checkpoints = [(0, 5), (1, 10)]
        layer_mapping = {0: 0, 1: 1}
        W = torch.eye(d)

        # 预计算教师痕迹（eval 模式，no_grad）
        teacher_imprints = _make_teacher_imprints(
            model, clean_ids, corrupt_ids, checkpoints,
        )

        # 创建损失函数
        loss_fn = RCIDLoss(
            W=W,
            checkpoints=checkpoints,
            layer_mapping=layer_mapping,
        )

        # 计算损失 — 同一模型做学生，也用 eval 以消除 dropout 差异
        model.eval()
        loss = loss_fn(teacher_imprints, model, clean_ids, corrupt_ids)

        assert loss.item() < 1e-4, (
            f"Loss should be ~0 when teacher=student and W=I, got {loss.item():.6f}"
        )

    def test_random_W_positive_loss(self) -> None:
        """随机 W 时损失应 > 0。"""
        d = 32
        model = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)

        torch.manual_seed(42)
        clean_ids = torch.randint(0, 100, (4, 16))
        corrupt_ids = torch.randint(0, 100, (4, 16))

        checkpoints = [(0, 5), (1, 10)]
        layer_mapping = {0: 0, 1: 1}

        # 随机正交 W（不是最优对齐）
        gen = torch.Generator().manual_seed(99)
        A = torch.randn(d, d, generator=gen)
        W_random, _ = torch.linalg.qr(A)

        teacher_imprints = _make_teacher_imprints(
            model, clean_ids, corrupt_ids, checkpoints,
        )

        loss_fn = RCIDLoss(
            W=W_random,
            checkpoints=checkpoints,
            layer_mapping=layer_mapping,
        )

        model.train()
        loss = loss_fn(teacher_imprints, model, clean_ids, corrupt_ids)

        assert loss.item() > 0.0, (
            f"Loss should be > 0 with random W, got {loss.item():.6f}"
        )

    def test_loss_is_finite(self) -> None:
        """损失值应始终有限。"""
        d = 32
        model = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)

        torch.manual_seed(42)
        clean_ids = torch.randint(0, 100, (4, 16))
        corrupt_ids = torch.randint(0, 100, (4, 16))

        checkpoints = [(0, 5)]
        layer_mapping = {0: 0}
        W = torch.eye(d)

        teacher_imprints = _make_teacher_imprints(
            model, clean_ids, corrupt_ids, checkpoints,
        )

        loss_fn = RCIDLoss(W=W, checkpoints=checkpoints, layer_mapping=layer_mapping)

        model.train()
        loss = loss_fn(teacher_imprints, model, clean_ids, corrupt_ids)

        assert loss.isfinite(), f"Loss is not finite: {loss.item()}"

    def test_loss_range(self) -> None:
        """归一化后的 MSE 损失应在 [0, 4] 范围内。

        ||a - b||² 当 ||a||=||b||=1 时最大值为 4（a = -b）。
        """
        d = 32
        model = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)

        torch.manual_seed(42)
        clean_ids = torch.randint(0, 100, (4, 16))
        corrupt_ids = torch.randint(0, 100, (4, 16))

        checkpoints = [(0, 5), (1, 10)]
        layer_mapping = {0: 0, 1: 1}
        W = torch.eye(d)

        teacher_imprints = _make_teacher_imprints(
            model, clean_ids, corrupt_ids, checkpoints,
        )

        loss_fn = RCIDLoss(W=W, checkpoints=checkpoints, layer_mapping=layer_mapping)

        model.train()
        loss = loss_fn(teacher_imprints, model, clean_ids, corrupt_ids)

        assert 0.0 <= loss.item() <= 4.0, (
            f"Loss {loss.item():.4f} outside expected range [0, 4]"
        )


# ======================================================================
# 梯度流测试
# ======================================================================

class TestGradientFlow:
    """验证梯度只流过学生模型。"""

    def test_gradient_flows_to_student(self) -> None:
        """梯度应流回学生模型参数。"""
        d = 32
        student = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)

        torch.manual_seed(42)
        clean_ids = torch.randint(0, 100, (4, 16))
        corrupt_ids = torch.randint(0, 100, (4, 16))

        checkpoints = [(0, 5)]
        layer_mapping = {0: 0}
        W = torch.eye(d)

        # 用另一个模型做教师
        teacher = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)
        teacher_imprints = _make_teacher_imprints(
            teacher, clean_ids, corrupt_ids, checkpoints,
        )

        loss_fn = RCIDLoss(W=W, checkpoints=checkpoints, layer_mapping=layer_mapping)

        student.train()
        student.zero_grad()
        loss = loss_fn(teacher_imprints, student, clean_ids, corrupt_ids)
        loss.backward()

        # 至少有一个学生参数有梯度
        has_grad = False
        for param in student.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradient flowed to student model parameters"

    def test_no_gradient_to_W(self) -> None:
        """W 是 buffer，不应有梯度。"""
        d = 32
        student = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)

        torch.manual_seed(42)
        clean_ids = torch.randint(0, 100, (4, 16))
        corrupt_ids = torch.randint(0, 100, (4, 16))

        checkpoints = [(0, 5)]
        layer_mapping = {0: 0}
        W = torch.eye(d)

        teacher = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)
        teacher_imprints = _make_teacher_imprints(
            teacher, clean_ids, corrupt_ids, checkpoints,
        )

        loss_fn = RCIDLoss(W=W, checkpoints=checkpoints, layer_mapping=layer_mapping)

        student.train()
        loss = loss_fn(teacher_imprints, student, clean_ids, corrupt_ids)
        loss.backward()

        # W 作为 buffer 不应有 grad
        assert loss_fn.W.grad is None, "W should not have gradient (it's a buffer)"

    def test_no_gradient_to_teacher_imprints(self) -> None:
        """教师痕迹不应有梯度（它们是在 no_grad 下预计算的）。"""
        d = 32
        student = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)

        torch.manual_seed(42)
        clean_ids = torch.randint(0, 100, (4, 16))
        corrupt_ids = torch.randint(0, 100, (4, 16))

        checkpoints = [(0, 5)]
        layer_mapping = {0: 0}
        W = torch.eye(d)

        teacher = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)
        teacher_imprints = _make_teacher_imprints(
            teacher, clean_ids, corrupt_ids, checkpoints,
        )

        loss_fn = RCIDLoss(W=W, checkpoints=checkpoints, layer_mapping=layer_mapping)

        student.train()
        loss = loss_fn(teacher_imprints, student, clean_ids, corrupt_ids)
        loss.backward()

        # 教师痕迹没有梯度（在 no_grad 下计算）
        for key, t_imprint in teacher_imprints.items():
            assert t_imprint.grad is None or t_imprint.grad.abs().sum() == 0, (
                f"Teacher imprint at {key} should not have gradient"
            )


# ======================================================================
# 非方阵 W 测试
# ======================================================================

class TestNonSquareW:
    """d_T ≠ d_S 时的非方阵 W 测试。"""

    def test_different_dimensions(self) -> None:
        """教师 d_T=64 和学生 d_S=32 时能正确计算损失。"""
        d_t, d_s = 64, 32
        teacher = _make_tiny_gpt2(n_layer=2, n_embd=d_t, n_head=2)
        student = _make_tiny_gpt2(n_layer=2, n_embd=d_s, n_head=2)

        torch.manual_seed(42)
        clean_ids = torch.randint(0, 100, (4, 16))
        corrupt_ids = torch.randint(0, 100, (4, 16))

        checkpoints = [(0, 5), (1, 10)]
        layer_mapping = {0: 0, 1: 1}

        # 非方阵 W: (d_T, d_S)
        gen = torch.Generator().manual_seed(42)
        A = torch.randn(d_t, d_s, generator=gen)
        W, _ = torch.linalg.qr(A)  # (d_T, d_S) 列正交

        teacher_imprints = _make_teacher_imprints(
            teacher, clean_ids, corrupt_ids, checkpoints,
        )

        loss_fn = RCIDLoss(
            W=W,
            checkpoints=checkpoints,
            layer_mapping=layer_mapping,
        )

        student.train()
        loss = loss_fn(teacher_imprints, student, clean_ids, corrupt_ids)

        assert loss.isfinite(), f"Loss is not finite: {loss.item()}"
        assert loss.item() >= 0.0, f"Loss should be non-negative, got {loss.item()}"

    def test_different_dims_gradient_flows(self) -> None:
        """非方阵 W 下，梯度仍流回学生。"""
        d_t, d_s = 64, 32
        teacher = _make_tiny_gpt2(n_layer=2, n_embd=d_t, n_head=2)
        student = _make_tiny_gpt2(n_layer=2, n_embd=d_s, n_head=2)

        torch.manual_seed(42)
        clean_ids = torch.randint(0, 100, (4, 16))
        corrupt_ids = torch.randint(0, 100, (4, 16))

        checkpoints = [(0, 5)]
        layer_mapping = {0: 0}

        gen = torch.Generator().manual_seed(42)
        A = torch.randn(d_t, d_s, generator=gen)
        W, _ = torch.linalg.qr(A)

        teacher_imprints = _make_teacher_imprints(
            teacher, clean_ids, corrupt_ids, checkpoints,
        )

        loss_fn = RCIDLoss(W=W, checkpoints=checkpoints, layer_mapping=layer_mapping)

        student.train()
        student.zero_grad()
        loss = loss_fn(teacher_imprints, student, clean_ids, corrupt_ids)
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in student.parameters()
        )
        assert has_grad, "No gradient to student with non-square W"


# ======================================================================
# 输入校验测试
# ======================================================================

class TestValidation:
    """输入校验边界测试。"""

    def test_rejects_empty_checkpoints(self) -> None:
        """空检查点列表应报错。"""
        with pytest.raises(AssertionError, match="empty"):
            RCIDLoss(
                W=torch.eye(32),
                checkpoints=[],
                layer_mapping={},
            )

    def test_rejects_1d_W(self) -> None:
        """W 不是 2D 时应报错。"""
        with pytest.raises(AssertionError, match="2D"):
            RCIDLoss(
                W=torch.randn(32),
                checkpoints=[(0, 5)],
                layer_mapping={0: 0},
            )

    def test_rejects_missing_layer_mapping(self) -> None:
        """checkpoints 中的教师层不在 layer_mapping 中应报错。"""
        with pytest.raises(AssertionError, match="not in layer_mapping"):
            RCIDLoss(
                W=torch.eye(32),
                checkpoints=[(0, 5), (5, 10)],
                layer_mapping={0: 0},  # 缺少 5 → ?
            )

    def test_rejects_shape_mismatch(self) -> None:
        """clean 和 corrupt 形状不匹配应报错。"""
        d = 32
        model = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)
        loss_fn = RCIDLoss(
            W=torch.eye(d),
            checkpoints=[(0, 5)],
            layer_mapping={0: 0},
        )

        teacher_imprints = {(0, 5): torch.randn(4, d)}

        with pytest.raises(AssertionError, match="Shape mismatch"):
            loss_fn(
                teacher_imprints,
                model,
                torch.randint(0, 100, (4, 16)),
                torch.randint(0, 100, (4, 20)),  # 不同 seq_len
            )

    def test_rejects_missing_teacher_imprint(self) -> None:
        """缺少教师痕迹应报错。"""
        d = 32
        model = _make_tiny_gpt2(n_layer=2, n_embd=d, n_head=2)
        loss_fn = RCIDLoss(
            W=torch.eye(d),
            checkpoints=[(0, 5), (1, 10)],
            layer_mapping={0: 0, 1: 1},
        )

        # 只提供一个检查点的痕迹
        teacher_imprints = {(0, 5): torch.randn(4, d)}

        with pytest.raises(AssertionError, match="Missing teacher imprint"):
            loss_fn(
                teacher_imprints,
                model,
                torch.randint(0, 100, (4, 16)),
                torch.randint(0, 100, (4, 16)),
            )


# ======================================================================
# W 作为 buffer 的行为测试
# ======================================================================

class TestWBuffer:
    """验证 W 作为 buffer 的行为。"""

    def test_W_is_buffer(self) -> None:
        """W 应注册为 buffer 而非 parameter。"""
        loss_fn = RCIDLoss(
            W=torch.eye(32),
            checkpoints=[(0, 5)],
            layer_mapping={0: 0},
        )

        # W 在 buffers 中
        buffer_names = [name for name, _ in loss_fn.named_buffers()]
        assert "W" in buffer_names, "W should be a registered buffer"

        # W 不在 parameters 中
        param_names = [name for name, _ in loss_fn.named_parameters()]
        assert "W" not in param_names, "W should NOT be a parameter"

    def test_W_not_requires_grad(self) -> None:
        """W 不应 requires_grad。"""
        loss_fn = RCIDLoss(
            W=torch.eye(32),
            checkpoints=[(0, 5)],
            layer_mapping={0: 0},
        )

        assert not loss_fn.W.requires_grad, "W should not require gradient"

    def test_W_follows_device(self) -> None:
        """W 作为 buffer 应跟随 .to(device)。"""
        loss_fn = RCIDLoss(
            W=torch.eye(32),
            checkpoints=[(0, 5)],
            layer_mapping={0: 0},
        )

        # 移到 CPU（已经在 CPU，但验证 .to 不报错）
        loss_fn.to("cpu")
        assert loss_fn.W.device == torch.device("cpu")

    def test_W_is_cloned(self) -> None:
        """传入的 W 应被 clone，修改原始 W 不影响 loss_fn。"""
        W_original = torch.eye(32)
        loss_fn = RCIDLoss(
            W=W_original,
            checkpoints=[(0, 5)],
            layer_mapping={0: 0},
        )

        # 修改原始 W
        W_original.fill_(0.0)

        # loss_fn 中的 W 不受影响
        assert loss_fn.W.abs().sum() > 0, (
            "W in loss_fn should be independent of original W"
        )
