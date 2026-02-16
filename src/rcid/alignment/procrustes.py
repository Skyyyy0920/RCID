"""Procrustes 正交对齐，求解冻结对齐矩阵 W*。

将学生差值空间 (d_S) 映射到教师差值空间 (d_T)。

核心数学：
    W* = argmin_{W} ||W @ source.T - target.T||_F
    解析解：M = target.T @ source, M = U S Vh (SVD), W = U @ Vh

当 d_T ≠ d_S 时（例如教师 768, 学生 384），W 为非方阵 (d_T, d_S)，
此时 W 的列具有正交性 (W^T @ W ≈ I_{d_S})，但行不具有正交性。

提供两个公开 API：
- procrustes_align: 求解最优对齐矩阵 W*
- compute_alignment_quality: 评估对齐质量（R² 和归一化误差）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class AlignmentQuality:
    """对齐质量统计。"""

    r_squared: float           # 决定系数 R²，越接近 1 对齐越好
    normalized_error: float    # 归一化残差 ||aligned - target||_F / ||target||_F
    mean_cosine: float         # 平均余弦相似度（每样本对齐向量 vs 目标向量）


# ======================================================================
# 核心 API
# ======================================================================

def procrustes_align(
    source: torch.Tensor,   # (N, d_S) — 学生侧的对比差值
    target: torch.Tensor,   # (N, d_T) — 教师侧的对比差值
    center: bool = True,
) -> torch.Tensor:          # (d_T, d_S) — 对齐矩阵
    """求解最优正交对齐矩阵 W*。

    最小化 ||W @ source.T - target.T||_F，其中 W 的列正交。
    解析解：M = target.T @ source = U S Vh (SVD)，W = U @ Vh。

    当 d_T > d_S 时，W 的形状为 (d_T, d_S)，列正交 (W^T W ≈ I)。
    当 d_T == d_S 时，W 为正交矩阵 (W W^T = W^T W = I)。

    Args:
        source: 学生侧对比差值, shape (N, d_S).
        target: 教师侧对比差值, shape (N, d_T).
        center: 是否中心化（减去列均值）。

    Returns:
        W: 对齐矩阵, shape (d_T, d_S)。
            使用方式: aligned = source @ W.T  →  (N, d_T)
    """
    # ── 输入校验 ──────────────────────────────────────────────────────
    assert source.dim() == 2, f"source should be 2D (N, d_S), got {source.dim()}D"
    assert target.dim() == 2, f"target should be 2D (N, d_T), got {target.dim()}D"
    assert source.shape[0] == target.shape[0], (
        f"Sample count mismatch: source has {source.shape[0]}, "
        f"target has {target.shape[0]}"
    )
    assert source.shape[0] >= 2, (
        f"Need at least 2 samples, got {source.shape[0]}"
    )

    n, d_s = source.shape
    _, d_t = target.shape

    # ── 中心化 ────────────────────────────────────────────────────────
    if center:
        source = source - source.mean(dim=0, keepdim=True)  # (N, d_S)
        target = target - target.mean(dim=0, keepdim=True)  # (N, d_T)

    # ── 交叉协方差矩阵 ────────────────────────────────────────────────
    M = target.T @ source  # (d_T, d_S)

    # ── SVD 求解，带数值稳定性处理 ────────────────────────────────────
    # full_matrices=False → U: (d_T, k), S: (k,), Vh: (k, d_S)，k = min(d_T, d_S)
    try:
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    except torch.linalg.LinAlgError:
        logger.warning("SVD did not converge, falling back to regularized version")
        # 对角正则化：在较小的维度上加微扰
        eye_size = min(d_t, d_s)
        M_reg = M.clone()
        M_reg[:eye_size, :eye_size] += 1e-6 * torch.eye(
            eye_size, device=M.device, dtype=M.dtype,
        )
        U, S, Vh = torch.linalg.svd(M_reg, full_matrices=False)

    # ── 构造对齐矩阵 ─────────────────────────────────────────────────
    W = U @ Vh  # (d_T, d_S) — 始终正交（W = U @ Vh 的固有性质）

    assert W.shape == (d_t, d_s), (
        f"W shape {W.shape} != expected ({d_t}, {d_s})"
    )

    # ── 正交性验证 ────────────────────────────────────────────────────
    WtW = W.T @ W  # (d_S, d_S)
    I_ds = torch.eye(d_s, device=W.device, dtype=W.dtype)  # (d_S, d_S)
    orthogonality_error = (WtW - I_ds).norm().item()

    if orthogonality_error > 1e-4:
        logger.warning(
            "W column-orthogonality error = %.6f (threshold=1e-4). "
            "W shape=(%d, %d)",
            orthogonality_error, d_t, d_s,
        )
    else:
        logger.info(
            "Procrustes alignment: W (%d, %d), orthogonality error = %.2e",
            d_t, d_s, orthogonality_error,
        )

    # 日志输出奇异值分布和条件数（诊断对齐质量）
    cond = S.max() / S.min().clamp(min=1e-12)
    logger.info(
        "Singular values: min=%.4f, max=%.4f, mean=%.4f, cond=%.2e",
        S.min().item(), S.max().item(), S.mean().item(), cond.item(),
    )
    if cond > 1e6:
        logger.warning(
            "High condition number (%.2e) — alignment signal is weak in some "
            "directions. RCID loss may converge slowly on this task.",
            cond.item(),
        )

    return W


def compute_alignment_quality(
    source: torch.Tensor,   # (N, d_S)
    target: torch.Tensor,   # (N, d_T)
    W: torch.Tensor,        # (d_T, d_S)
    center: bool = True,
    eps: float = 1e-8,
) -> AlignmentQuality:
    """评估 Procrustes 对齐质量。

    将 source 用 W 映射后，与 target 比较，计算多个质量指标。

    Args:
        source: 学生侧对比差值, shape (N, d_S).
        target: 教师侧对比差值, shape (N, d_T).
        W: 对齐矩阵, shape (d_T, d_S).
        center: 是否中心化后再评估。
        eps: 数值稳定常数。

    Returns:
        AlignmentQuality 包含 R², 归一化误差, 平均余弦相似度。
    """
    # ── 输入校验 ──────────────────────────────────────────────────────
    assert source.dim() == 2, f"source should be 2D, got {source.dim()}D"
    assert target.dim() == 2, f"target should be 2D, got {target.dim()}D"
    assert W.dim() == 2, f"W should be 2D, got {W.dim()}D"
    assert source.shape[0] == target.shape[0], (
        f"Sample count mismatch: source {source.shape[0]} vs target {target.shape[0]}"
    )
    assert W.shape == (target.shape[1], source.shape[1]), (
        f"W shape {W.shape} incompatible with d_T={target.shape[1]}, d_S={source.shape[1]}"
    )

    # ── 中心化 ────────────────────────────────────────────────────────
    if center:
        source = source - source.mean(dim=0, keepdim=True)  # (N, d_S)
        target = target - target.mean(dim=0, keepdim=True)  # (N, d_T)

    # ── 对齐 ──────────────────────────────────────────────────────────
    aligned = source @ W.T  # (N, d_T)

    # ── R² (决定系数) ─────────────────────────────────────────────────
    # R² = 1 - SS_res / SS_tot
    ss_res = (target - aligned).pow(2).sum()                    # scalar
    ss_tot = (target - target.mean(dim=0, keepdim=True)).pow(2).sum()  # scalar
    r_squared = 1.0 - (ss_res / ss_tot.clamp(min=eps)).item()

    # ── 归一化误差 ────────────────────────────────────────────────────
    # ||aligned - target||_F / ||target||_F
    residual_norm = (aligned - target).norm().item()
    target_norm = target.norm().clamp(min=eps).item()
    normalized_error = residual_norm / target_norm

    # ── 平均余弦相似度 ────────────────────────────────────────────────
    # 对每个样本计算 cos(aligned_i, target_i)
    aligned_norms = aligned.norm(dim=-1, keepdim=True).clamp(min=eps)  # (N, 1)
    target_norms = target.norm(dim=-1, keepdim=True).clamp(min=eps)    # (N, 1)

    cosines = (aligned / aligned_norms * target / target_norms).sum(dim=-1)  # (N,)
    mean_cosine = cosines.mean().item()

    quality = AlignmentQuality(
        r_squared=r_squared,
        normalized_error=normalized_error,
        mean_cosine=mean_cosine,
    )

    logger.info(
        "Alignment quality: R²=%.4f, normalized_error=%.4f, mean_cosine=%.4f",
        quality.r_squared, quality.normalized_error, quality.mean_cosine,
    )

    return quality
