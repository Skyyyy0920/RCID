"""CKA 计算模块单元测试。

验证 Linear CKA 的核心数学性质：
- 自相似度 == 1
- 正交不变性
- 不相关矩阵的 CKA ≈ 0
- 跨维度支持
- Mini-batch CKA 与全量一致
"""

from __future__ import annotations

import torch
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcid.alignment.cka import linear_cka, minibatch_cka


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def rng() -> torch.Generator:
    """固定随机种子，保证测试可复现。"""
    g = torch.Generator()
    g.manual_seed(42)
    return g


@pytest.fixture
def X(rng: torch.Generator) -> torch.Tensor:
    """标准测试矩阵 X: (100, 64)。"""
    return torch.randn(100, 64, generator=rng)


@pytest.fixture
def random_orthogonal(rng: torch.Generator) -> torch.Tensor:
    """随机正交矩阵 R: (64, 64)，通过 QR 分解生成。"""
    A = torch.randn(64, 64, generator=rng)
    Q, _ = torch.linalg.qr(A)
    return Q


# ======================================================================
# 核心性质测试
# ======================================================================

class TestLinearCKA:
    """linear_cka 的数学性质测试。"""

    def test_self_similarity(self, X: torch.Tensor) -> None:
        """CKA(X, X) == 1.0。"""
        cka = linear_cka(X, X)
        assert abs(cka - 1.0) < 1e-5, f"CKA(X, X) = {cka}, expected 1.0"

    def test_orthogonal_invariance(
        self, X: torch.Tensor, random_orthogonal: torch.Tensor,
    ) -> None:
        """CKA(X, X @ R) == 1.0，R 是正交矩阵。CKA 对旋转不变。"""
        Y = X @ random_orthogonal  # (100, 64) @ (64, 64) = (100, 64)
        cka = linear_cka(X, Y)
        assert abs(cka - 1.0) < 1e-5, (
            f"CKA(X, X@R) = {cka}, expected 1.0 (orthogonal invariance)"
        )

    def test_scaling_invariance(self, X: torch.Tensor) -> None:
        """CKA(X, alpha * X) == 1.0，各向同性缩放不变。"""
        Y = 3.7 * X
        cka = linear_cka(X, Y)
        assert abs(cka - 1.0) < 1e-5, (
            f"CKA(X, 3.7*X) = {cka}, expected 1.0 (scaling invariance)"
        )

    def test_uncorrelated_near_zero(self, rng: torch.Generator) -> None:
        """CKA(X, random_Y) ≈ 0，不相关矩阵的 CKA 应接近 0。"""
        n = 2000  # 足够多样本以降低随机波动
        X = torch.randn(n, 64, generator=rng)

        # 用不同的种子生成独立的 Y
        rng2 = torch.Generator()
        rng2.manual_seed(12345)
        Y = torch.randn(n, 64, generator=rng2)

        cka = linear_cka(X, Y)
        assert cka < 0.05, (
            f"CKA(X, independent_Y) = {cka}, expected < 0.05"
        )

    def test_different_dimensions(self, rng: torch.Generator) -> None:
        """X: (n, d1) 和 Y: (n, d2)，d1 ≠ d2 时也能正确计算。"""
        n = 100
        X = torch.randn(n, 64, generator=rng)

        rng2 = torch.Generator()
        rng2.manual_seed(99)
        Y = torch.randn(n, 128, generator=rng2)

        cka = linear_cka(X, Y)
        assert 0.0 <= cka <= 1.0, f"CKA out of range: {cka}"

    def test_different_dims_self_via_projection(
        self, rng: torch.Generator,
    ) -> None:
        """X: (n, 64) 通过线性投影到 (n, 32)，CKA 应仍然较高。"""
        n = 200
        X = torch.randn(n, 64, generator=rng)
        W = torch.randn(64, 32, generator=rng)
        Y = X @ W  # (n, 32) — 是 X 的线性函数

        cka = linear_cka(X, Y)
        # 线性投影保留了大部分结构，CKA 应 > 0.5
        assert cka > 0.5, (
            f"CKA(X, X@W) = {cka}, expected > 0.5 for linear projection"
        )

    def test_symmetry(self, rng: torch.Generator) -> None:
        """CKA(X, Y) == CKA(Y, X)。"""
        n = 100
        X = torch.randn(n, 64, generator=rng)

        rng2 = torch.Generator()
        rng2.manual_seed(77)
        Y = torch.randn(n, 32, generator=rng2)

        cka_xy = linear_cka(X, Y)
        cka_yx = linear_cka(Y, X)
        assert abs(cka_xy - cka_yx) < 1e-6, (
            f"CKA not symmetric: CKA(X,Y)={cka_xy}, CKA(Y,X)={cka_yx}"
        )

    def test_range(self, X: torch.Tensor, rng: torch.Generator) -> None:
        """CKA 的值域应始终在 [0, 1]。"""
        rng2 = torch.Generator()
        rng2.manual_seed(55)
        Y = torch.randn(100, 48, generator=rng2)

        cka = linear_cka(X, Y)
        assert 0.0 <= cka <= 1.0, f"CKA out of [0, 1]: {cka}"

    def test_center_flag(self, rng: torch.Generator) -> None:
        """center=False 跳过中心化，结果可能不同。"""
        n = 100
        # 构造有非零均值的数据
        X = torch.randn(n, 32, generator=rng) + 5.0
        Y = torch.randn(n, 32, generator=rng) + 3.0

        cka_centered = linear_cka(X, Y, center=True)
        cka_uncentered = linear_cka(X, Y, center=False)

        # 两者应该不同（除非恰好相等，概率极低）
        # 至少验证都在合理范围内
        assert 0.0 <= cka_centered <= 1.0
        assert 0.0 <= cka_uncentered <= 1.0


# ======================================================================
# 输入校验测试
# ======================================================================

class TestLinearCKAValidation:
    """linear_cka 的输入校验测试。"""

    def test_rejects_1d_input(self) -> None:
        """1D tensor 应被拒绝。"""
        with pytest.raises(AssertionError, match="2D"):
            linear_cka(torch.randn(10), torch.randn(10))

    def test_rejects_sample_mismatch(self) -> None:
        """样本数不一致应被拒绝。"""
        with pytest.raises(AssertionError, match="Sample count"):
            linear_cka(torch.randn(10, 5), torch.randn(20, 5))

    def test_rejects_single_sample(self) -> None:
        """单样本应被拒绝。"""
        with pytest.raises(AssertionError, match="at least 2"):
            linear_cka(torch.randn(1, 5), torch.randn(1, 5))


# ======================================================================
# Mini-batch CKA 测试
# ======================================================================

class TestMinibatchCKA:
    """minibatch_cka 的测试。"""

    def test_equals_full_when_small(self, X: torch.Tensor) -> None:
        """当 N <= batch_size 时，mini-batch CKA 应与全量 CKA 完全一致。"""
        Y = torch.randn_like(X)
        full = linear_cka(X, Y)
        mini = minibatch_cka(X, Y, batch_size=1000)  # N=100 < 1000
        assert abs(full - mini) < 1e-6, (
            f"full={full}, mini={mini} should be identical"
        )

    def test_close_to_full(self, rng: torch.Generator) -> None:
        """分批结果应与全量结果接近（对有相关性的数据）。"""
        n = 500
        X = torch.randn(n, 64, generator=rng)
        # 用线性变换生成 Y，确保 X 和 Y 有真实相关性
        rng2 = torch.Generator()
        rng2.manual_seed(88)
        W = torch.randn(64, 64, generator=rng2)
        noise = torch.randn(n, 64, generator=rng2) * 0.1
        Y = X @ W + noise  # 有强相关性

        full = linear_cka(X, Y)
        mini = minibatch_cka(X, Y, batch_size=128)

        # 对有相关性的数据，mini-batch 估计应接近全量
        assert abs(full - mini) < 0.15, (
            f"full={full:.4f}, mini={mini:.4f}, diff too large"
        )

    def test_self_similarity_minibatch(self, X: torch.Tensor) -> None:
        """Mini-batch CKA(X, X) 也应接近 1.0。"""
        cka = minibatch_cka(X, X, batch_size=32)
        assert abs(cka - 1.0) < 1e-4, (
            f"minibatch CKA(X, X) = {cka}, expected ~1.0"
        )

    def test_rejects_batch_size_one(self) -> None:
        """batch_size < 2 应被拒绝。"""
        with pytest.raises(AssertionError, match="batch_size"):
            minibatch_cka(torch.randn(10, 5), torch.randn(10, 5), batch_size=1)
