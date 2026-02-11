"""Centered Kernel Alignment (CKA) 计算。

Linear CKA 用于度量两组表示之间的相似性，具有以下性质：
- 对正交变换不变：CKA(X, X @ R) == 1.0（R 是正交矩阵）
- 对各向同性缩放不变：CKA(X, alpha * X) == 1.0
- 允许 X 和 Y 的特征维度不同

数学定义（中心化后）：
    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F · ||Y^T Y||_F)
其中 X: (n, d1), Y: (n, d2)。
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)


def linear_cka(
    X: torch.Tensor,   # (n, d1)
    Y: torch.Tensor,   # (n, d2)
    center: bool = True,
    eps: float = 1e-8,
) -> float:
    """计算 Linear CKA 相似度。

    Args:
        X: 第一组表示矩阵, shape (n, d1).
        Y: 第二组表示矩阵, shape (n, d2). n 必须与 X 相同。
        center: 是否对 X, Y 做列中心化（减去均值）。
        eps: 分母的数值稳定常数。

    Returns:
        CKA 相似度标量，范围 [0, 1]。
    """
    assert X.dim() == 2, f"X should be 2D (n, d1), got {X.dim()}D"
    assert Y.dim() == 2, f"Y should be 2D (n, d2), got {Y.dim()}D"
    assert X.shape[0] == Y.shape[0], (
        f"Sample count mismatch: X has {X.shape[0]}, Y has {Y.shape[0]}"
    )

    n = X.shape[0]
    assert n >= 2, f"Need at least 2 samples, got {n}"

    # 中心化：减去列均值
    if center:
        X = X - X.mean(dim=0, keepdim=True)  # (n, d1)
        Y = Y - Y.mean(dim=0, keepdim=True)  # (n, d2)

    # HSIC 估计（unnormalized）
    # HSIC(X, Y) = ||Y^T X||_F^2 / (n-1)^2
    # 但因为分子分母的 (n-1)^2 消掉了，直接用 Frobenius 范数比值
    YtX = Y.T @ X                            # (d2, d1)
    XtX = X.T @ X                            # (d1, d1)
    YtY = Y.T @ Y                            # (d2, d2)

    # ||M||_F^2 = trace(M^T M) = sum of squared elements
    hsic_xy = (YtX * YtX).sum()              # scalar: ||Y^T X||_F^2
    hsic_xx = (XtX * XtX).sum()              # scalar: ||X^T X||_F^2
    hsic_yy = (YtY * YtY).sum()              # scalar: ||Y^T Y||_F^2

    denominator = torch.sqrt(hsic_xx * hsic_yy).clamp(min=eps)  # scalar
    cka = (hsic_xy / denominator).item()     # scalar float

    # Clamp 到 [0, 1] — 数值误差可能导致微小越界
    cka = max(0.0, min(1.0, cka))

    return cka


def minibatch_cka(
    X: torch.Tensor,     # (N, d1)
    Y: torch.Tensor,     # (N, d2)
    batch_size: int = 256,
    center: bool = True,
    eps: float = 1e-8,
) -> float:
    """Mini-batch CKA：当样本数太大时分批计算再平均。

    将 N 个样本分成大小为 batch_size 的 mini-batch，
    对每个 batch 计算 CKA，返回加权平均。

    注意：mini-batch CKA 是全量 CKA 的有偏估计，但在实践中足够准确。
    当 N <= batch_size 时等价于直接调用 linear_cka。

    Args:
        X: 第一组表示矩阵, shape (N, d1).
        Y: 第二组表示矩阵, shape (N, d2).
        batch_size: 每个 mini-batch 的样本数。
        center: 是否对每个 mini-batch 做列中心化。
        eps: 分母的数值稳定常数。

    Returns:
        CKA 相似度标量，范围 [0, 1]。
    """
    assert X.shape[0] == Y.shape[0], (
        f"Sample count mismatch: X has {X.shape[0]}, Y has {Y.shape[0]}"
    )
    assert batch_size >= 2, f"batch_size must be >= 2, got {batch_size}"

    N = X.shape[0]

    # 小于等于 batch_size 时直接计算
    if N <= batch_size:
        return linear_cka(X, Y, center=center, eps=eps)

    # 分批计算
    cka_sum = 0.0
    total_weight = 0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        n_batch = end - start

        if n_batch < 2:
            continue  # 单样本无法计算 CKA

        X_batch = X[start:end]  # (n_batch, d1)
        Y_batch = Y[start:end]  # (n_batch, d2)

        cka_batch = linear_cka(X_batch, Y_batch, center=center, eps=eps)
        cka_sum += cka_batch * n_batch
        total_weight += n_batch

    assert total_weight > 0, "No valid batches to compute CKA"

    return cka_sum / total_weight
