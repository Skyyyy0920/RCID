"""Activation patching — 提取残差流因果痕迹向量。

因果痕迹定义：
    d_{l,t} = r_l(x_clean)[:, t, :] - r_l(x_corrupt)[:, t, :]
其中 r_l 是第 l 层 transformer block 输出端的残差流。

GPT-2 架构说明：
- model.transformer.h[i] 的 output[0] 是该 block 的残差流 (B, seq_len, d_model)
- GPT-2 使用 pre-LN，block 输出已经包含残差连接
- 对 RCID，我们需要的是 block 输出端（包含该层的贡献）
"""

from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ======================================================================
# Hook 管理 — context manager 确保 hook 不泄漏
# ======================================================================

@contextmanager
def _residual_hooks(
    model: nn.Module,
    layers: list[int],
    storage: dict[int, torch.Tensor],
) -> Iterator[dict[int, torch.Tensor]]:
    """临时注册多层 forward hook，提取 block 输出端的残差流。

    GPT-2: model.transformer.h[i] 的 output[0] 是该 block 的残差流，
    形状为 (batch, seq_len, d_model)。

    Args:
        model: GPT-2 风格模型，需具备 model.transformer.h 属性。
        layers: 需要提取的层索引列表。
        storage: 存放结果的字典，key=layer, value=(B, seq_len, d_model)。

    Yields:
        storage 引用，前向传播后即可读取。
    """
    n_layers = len(model.transformer.h)
    handles: list[torch.utils.hooks.RemovableHook] = []

    for layer in layers:
        assert 0 <= layer < n_layers, (
            f"Layer {layer} out of range [0, {n_layers})"
        )

        def _make_hook(l: int):  # noqa: E741
            """闭包捕获层索引，避免 late-binding 陷阱。"""
            def hook(
                module: nn.Module,
                input: tuple[torch.Tensor, ...],
                output: tuple[torch.Tensor, ...],
            ) -> None:
                # GPT-2 block output: (hidden_states, present_kv, ...)
                storage[l] = output[0]  # (B, seq_len, d_model)
            return hook

        handle = model.transformer.h[layer].register_forward_hook(
            _make_hook(layer)
        )
        handles.append(handle)

    try:
        yield storage
    finally:
        for handle in handles:
            handle.remove()


# ======================================================================
# 核心提取函数
# ======================================================================

def extract_causal_imprints(
    model: nn.Module,
    clean_inputs: torch.Tensor,                     # (N, seq_len)
    corrupt_inputs: torch.Tensor,                   # (N, seq_len)
    checkpoints: list[tuple[int, int]],             # [(layer, token_pos), ...]
) -> dict[tuple[int, int], torch.Tensor]:           # {(l, t): (N, d_model)}
    """在所有检查点提取因果痕迹。

    对每个检查点 (l, t):
        d_{l,t} = residual_l(x_clean)[:, t, :] - residual_l(x_corrupt)[:, t, :]

    两次前向传播使用完全相同的模型状态（eval + no_grad）。
    按层分组提取，最小化 hook 注册次数。

    Args:
        model: GPT-2 风格模型，需具备 model.transformer.h 属性。
        clean_inputs: clean 版本的 token ids, shape (N, seq_len).
        corrupt_inputs: corrupt 版本的 token ids, shape (N, seq_len).
        checkpoints: 要提取痕迹的 (layer, token_pos) 列表.

    Returns:
        字典 {(layer, token_pos): imprint_tensor}，
        每个 imprint_tensor 的 shape 为 (N, d_model)。
    """
    # ── 输入校验 ──────────────────────────────────────────────────────
    assert clean_inputs.dim() == 2, (
        f"clean_inputs should be 2D (N, seq_len), got {clean_inputs.dim()}D"
    )
    assert corrupt_inputs.dim() == 2, (
        f"corrupt_inputs should be 2D (N, seq_len), got {corrupt_inputs.dim()}D"
    )
    assert clean_inputs.shape == corrupt_inputs.shape, (
        f"Shape mismatch: clean {clean_inputs.shape} vs corrupt {corrupt_inputs.shape}"
    )
    assert len(checkpoints) > 0, "checkpoints list is empty"

    seq_len = clean_inputs.shape[1]
    for layer, token_pos in checkpoints:
        assert 0 <= token_pos < seq_len, (
            f"Token position {token_pos} out of range [0, {seq_len})"
        )

    # ── 按层分组检查点，减少 hook 注册次数 ────────────────────────────
    layer_to_positions: dict[int, list[int]] = defaultdict(list)
    for layer, token_pos in checkpoints:
        if token_pos not in layer_to_positions[layer]:
            layer_to_positions[layer].append(token_pos)

    layers_needed = sorted(layer_to_positions.keys())

    # ── eval + no_grad 下提取 ─────────────────────────────────────────
    model.eval()
    imprints: dict[tuple[int, int], torch.Tensor] = {}

    with torch.no_grad():
        for layer in layers_needed:
            positions = layer_to_positions[layer]

            # 每层注册一次 hook，做两次前向（clean + corrupt）
            storage_clean: dict[int, torch.Tensor] = {}
            storage_corrupt: dict[int, torch.Tensor] = {}

            with _residual_hooks(model, [layer], storage_clean):
                model(clean_inputs)
            # storage_clean[layer] 现在是 (N, seq_len, d_model)

            with _residual_hooks(model, [layer], storage_corrupt):
                model(corrupt_inputs)
            # storage_corrupt[layer] 现在是 (N, seq_len, d_model)

            residual_clean = storage_clean[layer]    # (N, seq_len, d_model)
            residual_corrupt = storage_corrupt[layer]  # (N, seq_len, d_model)

            assert residual_clean.dim() == 3, (
                f"Expected 3D residual, got {residual_clean.dim()}D"
            )
            assert residual_clean.shape == residual_corrupt.shape, (
                f"Residual shape mismatch at layer {layer}: "
                f"{residual_clean.shape} vs {residual_corrupt.shape}"
            )

            for t in positions:
                d = residual_clean[:, t, :] - residual_corrupt[:, t, :]  # (N, d_model)
                imprints[(layer, t)] = d

    logger.info(
        "Extracted %d causal imprints across %d layers, batch_size=%d",
        len(imprints), len(layers_needed), clean_inputs.shape[0],
    )

    return imprints


# ======================================================================
# 批量扫描：对所有 (layer, token_pos) 组合计算痕迹范数
# ======================================================================

def compute_imprint_norms(
    model: nn.Module,
    clean_inputs: torch.Tensor,               # (N, seq_len)
    corrupt_inputs: torch.Tensor,             # (N, seq_len)
    layers: list[int] | None = None,
    token_positions: list[int] | None = None,
) -> list[tuple[int, int, float]]:
    """计算所有 (layer, token_pos) 组合的平均痕迹 L2 范数并排序。

    用于选择因果效应最大的 top-k 检查点。

    Args:
        model: GPT-2 风格模型。
        clean_inputs: shape (N, seq_len).
        corrupt_inputs: shape (N, seq_len).
        layers: 要扫描的层列表；None 则扫描全部层。
        token_positions: 要扫描的 token 位置列表；None 则扫描全部位置。

    Returns:
        按范数降序排列的列表 [(layer, token_pos, mean_norm), ...]。
    """
    assert clean_inputs.shape == corrupt_inputs.shape, (
        f"Shape mismatch: clean {clean_inputs.shape} vs corrupt {corrupt_inputs.shape}"
    )

    n_layers = len(model.transformer.h)
    seq_len = clean_inputs.shape[1]

    if layers is None:
        layers = list(range(n_layers))
    if token_positions is None:
        token_positions = list(range(seq_len))

    # 构建所有检查点
    checkpoints = [
        (l, t) for l in layers for t in token_positions
    ]

    if len(checkpoints) == 0:
        return []

    logger.info(
        "Scanning %d checkpoints (%d layers x %d positions)...",
        len(checkpoints), len(layers), len(token_positions),
    )

    # 提取所有痕迹
    imprints = extract_causal_imprints(
        model, clean_inputs, corrupt_inputs, checkpoints,
    )

    # 计算每个检查点的平均 L2 范数
    results: list[tuple[int, int, float]] = []
    for (layer, token_pos), d in imprints.items():
        mean_norm = d.norm(dim=-1).mean().item()  # scalar: mean over batch
        results.append((layer, token_pos, mean_norm))

    # 按范数降序排列
    results.sort(key=lambda x: x[2], reverse=True)

    if len(results) > 0:
        top = results[0]
        logger.info(
            "Top checkpoint: layer=%d, pos=%d, mean_norm=%.4f",
            top[0], top[1], top[2],
        )

    return results


def select_top_checkpoints(
    model: nn.Module,
    clean_inputs: torch.Tensor,               # (N, seq_len)
    corrupt_inputs: torch.Tensor,             # (N, seq_len)
    top_k: int = 5,
    layers: list[int] | None = None,
    token_positions: list[int] | None = None,
) -> list[tuple[int, int]]:
    """选择因果效应最大的 top-k 检查点。

    基于 compute_imprint_norms 的排序结果，返回前 k 个 (layer, token_pos)。
    这是 RCID 流水线 Step 1 的核心：确定哪些残差流位置携带了最多因果信息。

    Args:
        model: GPT-2 风格模型。
        clean_inputs: shape (N, seq_len).
        corrupt_inputs: shape (N, seq_len).
        top_k: 要选择的检查点数量。
        layers: 要扫描的层列表；None 则扫描全部层。
        token_positions: 要扫描的 token 位置列表；None 则扫描全部位置。

    Returns:
        top-k 检查点列表 [(layer, token_pos), ...]，按因果效应从大到小排列。
    """
    assert top_k > 0, f"top_k must be positive, got {top_k}"

    ranked = compute_imprint_norms(
        model, clean_inputs, corrupt_inputs, layers, token_positions,
    )

    selected = [(layer, pos) for layer, pos, _ in ranked[:top_k]]

    logger.info(
        "Selected top-%d checkpoints: %s",
        top_k, selected,
    )

    return selected
