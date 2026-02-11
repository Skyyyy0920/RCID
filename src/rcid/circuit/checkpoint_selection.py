"""因果检查点选择，基于因果效应排序选取 top-k 检查点。

从所有 (layer, token_pos) 组合中，计算平均因果痕迹范数，
选择效应最大的 top-k 作为 RCID 蒸馏检查点集合 C。

只搜索数据集标注的关键 token 位置（如 IOI 的 IO/S2/末尾位置），
避免对全序列做昂贵的全量扫描。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from rcid.circuit.patching import extract_causal_imprints

logger = logging.getLogger(__name__)


@dataclass
class CheckpointResult:
    """单个检查点的因果效应统计。"""

    layer: int
    token_pos: int
    mean_norm: float        # 平均 L2 范数（跨 batch）
    std_norm: float         # 范数的标准差（跨 batch）


def select_checkpoints(
    model: nn.Module,
    clean_inputs: torch.Tensor,             # (N, seq_len)
    corrupt_inputs: torch.Tensor,           # (N, seq_len)
    key_positions: list[int],
    top_k: int = 5,
    layers: list[int] | None = None,
) -> list[CheckpointResult]:
    """选择因果效应最大的 top-k 检查点。

    只在 key_positions 指定的 token 位置上搜索，而非全序列。
    这些位置应由数据集提供（如 IOI 的 io_token_pos、s2_token_pos、末尾位置）。

    Args:
        model: GPT-2 风格模型，需具备 model.transformer.h 属性。
        clean_inputs: clean 版本的 token ids, shape (N, seq_len).
        corrupt_inputs: corrupt 版本的 token ids, shape (N, seq_len).
        key_positions: 要搜索的 token 位置列表（去重后使用）。
        top_k: 选择的检查点数量。
        layers: 要搜索的层列表；None 则搜索全部层。

    Returns:
        按 mean_norm 降序排列的 top-k 个 CheckpointResult 列表。
    """
    assert clean_inputs.dim() == 2, (
        f"clean_inputs should be 2D (N, seq_len), got {clean_inputs.dim()}D"
    )
    assert clean_inputs.shape == corrupt_inputs.shape, (
        f"Shape mismatch: clean {clean_inputs.shape} vs corrupt {corrupt_inputs.shape}"
    )
    assert top_k > 0, f"top_k must be positive, got {top_k}"
    assert len(key_positions) > 0, "key_positions is empty"

    n_layers = len(model.transformer.h)
    if layers is None:
        layers = list(range(n_layers))

    # 去重并排序关键位置
    unique_positions = sorted(set(key_positions))

    # 构建所有待评估的 (layer, token_pos) 组合
    checkpoints = [
        (l, t) for l in layers for t in unique_positions
    ]

    logger.info(
        "Scanning %d checkpoints (%d layers x %d key positions)...",
        len(checkpoints), len(layers), len(unique_positions),
    )

    # 提取因果痕迹
    imprints = extract_causal_imprints(
        model, clean_inputs, corrupt_inputs, checkpoints,
    )

    # 计算每个检查点的范数统计
    results: list[CheckpointResult] = []
    for (layer, token_pos), d in imprints.items():
        norms = d.norm(dim=-1)                   # (N,)
        mean_norm = norms.mean().item()           # scalar
        std_norm = norms.std().item()             # scalar
        results.append(CheckpointResult(
            layer=layer,
            token_pos=token_pos,
            mean_norm=mean_norm,
            std_norm=std_norm,
        ))

    # 按 mean_norm 降序排序
    results.sort(key=lambda r: r.mean_norm, reverse=True)

    # 截取 top-k
    selected = results[:top_k]

    # 打印 summary
    _log_selection_summary(selected, n_layers, unique_positions)

    return selected


def collect_key_positions_ioi(
    batch: dict[str, torch.Tensor],
) -> list[int]:
    """从 IOI batch 中收集关键 token 位置。

    收集三种关键位置（去重）：
    - io_token_pos: indirect object 位置
    - s2_token_pos: subject 第二次出现位置
    - 末尾位置 (seq_len - 1): 模型最终预测位置

    Args:
        batch: IOI DataLoader 产生的 batch 字典，包含
               clean_ids (B, seq_len), io_token_pos (B,), s2_token_pos (B,).

    Returns:
        去重后的关键位置列表。
    """
    positions: set[int] = set()

    # IO 位置
    for pos in batch["io_token_pos"].tolist():
        positions.add(int(pos))

    # S2 位置
    for pos in batch["s2_token_pos"].tolist():
        positions.add(int(pos))

    # 末尾位置：找到每个样本的实际长度（非 pad 部分）
    clean_ids = batch["clean_ids"]  # (B, seq_len)
    for i in range(clean_ids.shape[0]):
        # 找最后一个非零 token 的位置
        nonzero_mask = clean_ids[i] != 0  # (seq_len,)
        if nonzero_mask.any():
            last_pos = int(nonzero_mask.nonzero()[-1].item())
            positions.add(last_pos)

    return sorted(positions)


def collect_key_positions_greater_than(
    batch: dict[str, torch.Tensor],
) -> list[int]:
    """从 Greater-Than batch 中收集关键 token 位置。

    收集两种关键位置（去重）：
    - year_token_pos: 起始年份后两位所在位置
    - 末尾位置 (seq_len - 1): 模型最终预测位置

    Args:
        batch: Greater-Than DataLoader 产生的 batch 字典，
               包含 clean_ids (B, seq_len), year_token_pos (B,).

    Returns:
        去重后的关键位置列表。
    """
    positions: set[int] = set()

    # 年份位置
    for pos in batch["year_token_pos"].tolist():
        positions.add(int(pos))

    # 末尾位置
    clean_ids = batch["clean_ids"]  # (B, seq_len)
    for i in range(clean_ids.shape[0]):
        nonzero_mask = clean_ids[i] != 0  # (seq_len,)
        if nonzero_mask.any():
            last_pos = int(nonzero_mask.nonzero()[-1].item())
            positions.add(last_pos)

    return sorted(positions)


def checkpoints_to_tuples(
    results: list[CheckpointResult],
) -> list[tuple[int, int]]:
    """将 CheckpointResult 列表转换为 (layer, token_pos) 元组列表。

    方便直接传给 extract_causal_imprints 等函数。

    Args:
        results: CheckpointResult 列表。

    Returns:
        [(layer, token_pos), ...] 列表，保持原有顺序。
    """
    return [(r.layer, r.token_pos) for r in results]


# ======================================================================
# 内部工具
# ======================================================================

def _log_selection_summary(
    selected: list[CheckpointResult],
    n_layers: int,
    searched_positions: list[int],
) -> None:
    """打印检查点选择的 summary。"""
    if not selected:
        logger.warning("No checkpoints selected.")
        return

    logger.info("=" * 60)
    logger.info("Checkpoint Selection Summary")
    logger.info("=" * 60)
    logger.info("Searched: %d layers x %d positions", n_layers, len(searched_positions))
    logger.info("Selected top-%d checkpoints:", len(selected))
    logger.info("-" * 60)
    logger.info("%-6s %-10s %-12s %-12s", "Rank", "Layer", "Token Pos", "Mean Norm")
    logger.info("-" * 60)

    for rank, r in enumerate(selected, 1):
        logger.info(
            "%-6d %-10d %-12d %-12.4f (±%.4f)",
            rank, r.layer, r.token_pos, r.mean_norm, r.std_norm,
        )

    logger.info("-" * 60)

    # 汇总被选中的层和位置
    selected_layers = sorted(set(r.layer for r in selected))
    selected_positions = sorted(set(r.token_pos for r in selected))
    logger.info("Layers involved:    %s", selected_layers)
    logger.info("Positions involved: %s", selected_positions)
    logger.info("=" * 60)
