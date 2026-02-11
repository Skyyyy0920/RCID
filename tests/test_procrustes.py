"""Procrustes 对齐模块单元测试。

测试覆盖：
- 已知旋转恢复（方阵 d_T == d_S）
- 非方阵情况（d_T != d_S）
- 含噪声的旋转恢复
- 对齐质量评估
- 输入校验
- 中心化行为
- SVD 回退机制
"""

from __future__ import annotations

import math

import pytest
import torch

from rcid.alignment.procrustes import (
    AlignmentQuality,
    compute_alignment_quality,
    procrustes_align,
)


# ======================================================================
# 辅助函数
# ======================================================================

def _random_orthogonal(d: int, seed: int = 42) -> torch.Tensor:
    """生成一个随机正交矩阵 (d, d)。"""
    gen = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=gen)
    Q, _ = torch.linalg.qr(A)
    return Q  # (d, d)


def _random_column_orthogonal(d_t: int, d_s: int, seed: int = 42) -> torch.Tensor:
    """生成一个随机列正交矩阵 (d_t, d_s)，其中 d_t > d_s。

    W^T @ W = I_{d_s}，但 W @ W^T ≠ I_{d_t}。
    """
    assert d_t > d_s, "d_t must be > d_s for non-square case"
    gen = torch.Generator().manual_seed(seed)
    A = torch.randn(d_t, d_s, generator=gen)
    Q, _ = torch.linalg.qr(A)
    return Q  # (d_t, d_s)


# ======================================================================
# TestProcrustesAlign — 核心对齐函数测试
# ======================================================================

class TestProcrustesAlign:
    """procrustes_align 的单元测试。"""

    def test_known_rotation_recovery(self) -> None:
        """构造已知旋转 target = source @ R.T，验证恢复的 W ≈ R。"""
        d = 64
        n = 200
        gen = torch.Generator().manual_seed(42)

        R = _random_orthogonal(d, seed=42)         # (d, d)
        source = torch.randn(n, d, generator=gen)  # (N, d)
        target = source @ R.T                       # (N, d)

        W = procrustes_align(source, target, center=False)

        # W 应该恢复 R
        assert W.shape == (d, d)
        error = (W - R).norm().item()
        assert error < 1e-4, f"Rotation recovery error = {error:.6f}"

    def test_known_rotation_recovery_with_center(self) -> None:
        """中心化后的已知旋转恢复。"""
        d = 64
        n = 200
        gen = torch.Generator().manual_seed(123)

        R = _random_orthogonal(d, seed=99)
        source = torch.randn(n, d, generator=gen)
        # 添加偏移（中心化应消除）
        source_shifted = source + 5.0
        target = source @ R.T + 3.0  # (N, d)

        W = procrustes_align(source_shifted, target, center=True)

        # W 应该恢复 R（中心化消除偏移）
        assert W.shape == (d, d)
        error = (W - R).norm().item()
        assert error < 1e-4, f"Rotation recovery error with center = {error:.6f}"

    def test_known_rotation_with_noise(self) -> None:
        """含噪声的旋转恢复：target = source @ R.T + noise。"""
        d = 64
        n = 500
        gen = torch.Generator().manual_seed(42)

        R = _random_orthogonal(d, seed=42)
        source = torch.randn(n, d, generator=gen)
        noise = 0.1 * torch.randn(n, d, generator=gen)
        target = source @ R.T + noise  # (N, d)

        W = procrustes_align(source, target, center=False)

        # 有噪声时误差更大，但 W 应该接近 R
        assert W.shape == (d, d)
        error = (W - R).norm().item()
        assert error < 1.0, f"Noisy rotation recovery error = {error:.6f}"

    def test_non_square_d_t_gt_d_s(self) -> None:
        """d_T > d_S 时验证 W 维度和列正交性。

        模拟教师 768 维 → 学生 384 维 的实际场景。
        """
        d_t, d_s = 128, 64
        n = 200
        gen = torch.Generator().manual_seed(42)

        # 构造列正交矩阵 R: (d_t, d_s)
        R = _random_column_orthogonal(d_t, d_s, seed=42)

        source = torch.randn(n, d_s, generator=gen)  # (N, d_S)
        target = source @ R.T                          # (N, d_T)

        W = procrustes_align(source, target, center=False)

        # 维度检查
        assert W.shape == (d_t, d_s), f"W shape {W.shape} != ({d_t}, {d_s})"

        # 列正交性：W^T W ≈ I_{d_s}
        WtW = W.T @ W  # (d_S, d_S)
        I_ds = torch.eye(d_s)
        ortho_error = (WtW - I_ds).norm().item()
        assert ortho_error < 1e-4, (
            f"Column orthogonality error = {ortho_error:.6f}"
        )

    def test_non_square_d_t_lt_d_s(self) -> None:
        """d_T < d_S 时验证 W 维度和列正交性。"""
        d_t, d_s = 32, 64
        n = 200
        gen = torch.Generator().manual_seed(42)

        source = torch.randn(n, d_s, generator=gen)  # (N, d_S)
        target = torch.randn(n, d_t, generator=gen)   # (N, d_T)

        W = procrustes_align(source, target, center=False)

        # 维度检查
        assert W.shape == (d_t, d_s), f"W shape {W.shape} != ({d_t}, {d_s})"

        # 当 d_t < d_s 时，k = min(d_t, d_s) = d_t
        # W^T W 应该是 (d_s, d_s) 但 rank 至多 d_t，不再是 I
        # 但 W W^T ≈ I_{d_t} 应该成立
        WWt = W @ W.T  # (d_T, d_T)
        I_dt = torch.eye(d_t)
        ortho_error = (WWt - I_dt).norm().item()
        assert ortho_error < 1e-4, (
            f"Row orthogonality error = {ortho_error:.6f}"
        )

    def test_orthogonality_square(self) -> None:
        """方阵时，W 是完全正交矩阵 (W W^T = W^T W = I)。"""
        d = 32
        n = 100
        gen = torch.Generator().manual_seed(42)

        source = torch.randn(n, d, generator=gen)
        target = torch.randn(n, d, generator=gen)

        W = procrustes_align(source, target, center=False)

        # W^T W = I
        WtW = W.T @ W
        error_WtW = (WtW - torch.eye(d)).norm().item()
        assert error_WtW < 1e-4, f"W^T W error = {error_WtW:.6f}"

        # W W^T = I
        WWt = W @ W.T
        error_WWt = (WWt - torch.eye(d)).norm().item()
        assert error_WWt < 1e-4, f"W W^T error = {error_WWt:.6f}"

    def test_identity_mapping(self) -> None:
        """source == target 时，W 应为单位矩阵。"""
        d = 32
        n = 100
        gen = torch.Generator().manual_seed(42)

        data = torch.randn(n, d, generator=gen)

        W = procrustes_align(data, data.clone(), center=False)

        error = (W - torch.eye(d)).norm().item()
        assert error < 1e-4, f"Identity mapping error = {error:.6f}"

    def test_realistic_dimensions(self) -> None:
        """模拟实际场景：d_T=768, d_S=384 的维度。"""
        d_t, d_s = 768, 384
        n = 100
        gen = torch.Generator().manual_seed(42)

        source = torch.randn(n, d_s, generator=gen)
        target = torch.randn(n, d_t, generator=gen)

        W = procrustes_align(source, target, center=True)

        assert W.shape == (d_t, d_s)

        # 列正交性
        WtW = W.T @ W
        ortho_error = (WtW - torch.eye(d_s)).norm().item()
        assert ortho_error < 1e-3, (
            f"Column orthogonality error for (768, 384) = {ortho_error:.6f}"
        )


# ======================================================================
# TestAlignmentQuality — 对齐质量评估测试
# ======================================================================

class TestAlignmentQuality:
    """compute_alignment_quality 的单元测试。"""

    def test_perfect_alignment(self) -> None:
        """完美对齐时，R² ≈ 1, error ≈ 0, cosine ≈ 1。"""
        d = 32
        n = 100
        gen = torch.Generator().manual_seed(42)

        R = _random_orthogonal(d, seed=42)
        source = torch.randn(n, d, generator=gen)
        target = source @ R.T

        W = procrustes_align(source, target, center=False)
        quality = compute_alignment_quality(source, target, W, center=False)

        assert isinstance(quality, AlignmentQuality)
        assert quality.r_squared > 0.999, f"R² = {quality.r_squared:.6f}"
        assert quality.normalized_error < 1e-3, (
            f"normalized_error = {quality.normalized_error:.6f}"
        )
        assert quality.mean_cosine > 0.999, (
            f"mean_cosine = {quality.mean_cosine:.6f}"
        )

    def test_noisy_alignment(self) -> None:
        """含噪声时，各指标应合理但不完美。"""
        d = 64
        n = 500
        gen = torch.Generator().manual_seed(42)

        R = _random_orthogonal(d, seed=42)
        source = torch.randn(n, d, generator=gen)
        noise = 0.3 * torch.randn(n, d, generator=gen)
        target = source @ R.T + noise

        W = procrustes_align(source, target, center=False)
        quality = compute_alignment_quality(source, target, W, center=False)

        # 有噪声但 R² 应该仍然较高
        assert quality.r_squared > 0.5, f"R² = {quality.r_squared:.6f}"
        assert quality.normalized_error < 1.0, (
            f"normalized_error = {quality.normalized_error:.6f}"
        )
        assert quality.mean_cosine > 0.5, (
            f"mean_cosine = {quality.mean_cosine:.6f}"
        )

    def test_non_square_alignment_quality(self) -> None:
        """d_T > d_S 时评估质量。"""
        d_t, d_s = 128, 64
        n = 200
        gen = torch.Generator().manual_seed(42)

        R = _random_column_orthogonal(d_t, d_s, seed=42)
        source = torch.randn(n, d_s, generator=gen)
        target = source @ R.T

        W = procrustes_align(source, target, center=False)
        quality = compute_alignment_quality(source, target, W, center=False)

        assert quality.r_squared > 0.99, f"R² = {quality.r_squared:.6f}"
        assert quality.mean_cosine > 0.99, (
            f"mean_cosine = {quality.mean_cosine:.6f}"
        )

    def test_random_W_poor_quality(self) -> None:
        """使用随机 W（非 Procrustes 解），质量应较差。"""
        d = 32
        n = 100
        gen = torch.Generator().manual_seed(42)

        source = torch.randn(n, d, generator=gen)
        target = torch.randn(n, d, generator=gen)

        # 随机正交矩阵，不是最优对齐
        W_random = _random_orthogonal(d, seed=99)
        W_optimal = procrustes_align(source, target, center=False)

        q_random = compute_alignment_quality(source, target, W_random, center=False)
        q_optimal = compute_alignment_quality(source, target, W_optimal, center=False)

        # 最优解的 R² 应高于随机
        assert q_optimal.r_squared >= q_random.r_squared, (
            f"Optimal R²={q_optimal.r_squared:.4f} < Random R²={q_random.r_squared:.4f}"
        )


# ======================================================================
# TestValidation — 输入校验测试
# ======================================================================

class TestValidation:
    """输入校验的边界测试。"""

    def test_rejects_1d_source(self) -> None:
        """source 不是 2D 时应报错。"""
        with pytest.raises(AssertionError, match="2D"):
            procrustes_align(torch.randn(10), torch.randn(10, 5))

    def test_rejects_1d_target(self) -> None:
        """target 不是 2D 时应报错。"""
        with pytest.raises(AssertionError, match="2D"):
            procrustes_align(torch.randn(10, 5), torch.randn(10))

    def test_rejects_sample_mismatch(self) -> None:
        """样本数不匹配应报错。"""
        with pytest.raises(AssertionError, match="Sample count"):
            procrustes_align(torch.randn(10, 5), torch.randn(20, 5))

    def test_rejects_single_sample(self) -> None:
        """单样本应报错。"""
        with pytest.raises(AssertionError, match="at least 2"):
            procrustes_align(torch.randn(1, 5), torch.randn(1, 5))

    def test_quality_rejects_wrong_W_shape(self) -> None:
        """W 维度不兼容时 compute_alignment_quality 应报错。"""
        source = torch.randn(10, 5)
        target = torch.randn(10, 8)
        W_wrong = torch.randn(5, 5)  # 应为 (8, 5)

        with pytest.raises(AssertionError, match="incompatible"):
            compute_alignment_quality(source, target, W_wrong)


# ======================================================================
# TestCenterBehavior — 中心化行为测试
# ======================================================================

class TestCenterBehavior:
    """center 参数行为测试。"""

    def test_center_removes_mean_shift(self) -> None:
        """中心化应消除均值偏移对对齐结果的影响。"""
        d = 32
        n = 200
        gen = torch.Generator().manual_seed(42)

        R = _random_orthogonal(d, seed=42)
        source = torch.randn(n, d, generator=gen)
        target = source @ R.T

        # 加大偏移
        source_shifted = source + 100.0
        target_shifted = target + 200.0

        W_centered = procrustes_align(
            source_shifted, target_shifted, center=True,
        )
        W_original = procrustes_align(source, target, center=False)

        # 两者应产生相同的 W
        error = (W_centered - W_original).norm().item()
        assert error < 1e-4, f"Center invariance error = {error:.6f}"

    def test_no_center_preserves_mean(self) -> None:
        """center=False 时不做中心化，均值偏移会影响结果。"""
        d = 32
        n = 200
        gen = torch.Generator().manual_seed(42)

        R = _random_orthogonal(d, seed=42)
        source = torch.randn(n, d, generator=gen)
        target = source @ R.T

        # 不加偏移
        W_no_shift = procrustes_align(source, target, center=False)

        # 加偏移
        W_with_shift = procrustes_align(
            source + 100.0, target + 200.0, center=False,
        )

        # 两者应不同（偏移改变了交叉协方差矩阵）
        diff = (W_no_shift - W_with_shift).norm().item()
        assert diff > 0.01, (
            f"Expected different W without centering, diff = {diff:.6f}"
        )
