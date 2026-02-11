"""基于 CKA 的教师-学生层匹配搜索。

对每个教师检查点层，在学生的所有层中找到因果痕迹 CKA 最高的层。
注意：比较的是因果痕迹（对比差值向量），不是完整残差流。

提供两级 API：
- find_best_layer_mapping: 从预提取的痕迹 dict 直接搜索（轻量）
- find_layer_mapping_from_models: 端到端接口，接收模型 + 对比数据
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from rcid.alignment.cka import linear_cka
from rcid.circuit.patching import extract_causal_imprints

logger = logging.getLogger(__name__)


# ======================================================================
# 核心搜索（从预提取的痕迹 dict）
# ======================================================================

def find_best_layer_mapping(
    teacher_imprints_by_layer: dict[int, torch.Tensor],  # {l: (N, d_T)}
    student_imprints_by_layer: dict[int, torch.Tensor],  # {l': (N, d_S)}
) -> tuple[dict[int, int], np.ndarray]:
    """对每个教师检查点层，找到 CKA 最高的学生层。

    Args:
        teacher_imprints_by_layer: 教师每层的因果痕迹。
            key = 层索引，value = (N, d_T) 的痕迹张量。
        student_imprints_by_layer: 学生每层的因果痕迹。
            key = 层索引，value = (N, d_S) 的痕迹张量。

    Returns:
        mapping: {teacher_layer: best_student_layer} 字典。
        cka_matrix: (n_teacher_layers, n_student_layers) 的完整 CKA 矩阵。
    """
    assert len(teacher_imprints_by_layer) > 0, "Teacher imprints dict is empty"
    assert len(student_imprints_by_layer) > 0, "Student imprints dict is empty"

    t_layers = sorted(teacher_imprints_by_layer.keys())
    s_layers = sorted(student_imprints_by_layer.keys())

    # 验证样本数一致
    n_samples_set: set[int] = set()
    for d in teacher_imprints_by_layer.values():
        n_samples_set.add(d.shape[0])
    for d in student_imprints_by_layer.values():
        n_samples_set.add(d.shape[0])
    assert len(n_samples_set) == 1, (
        f"All imprints must have the same sample count, got {n_samples_set}"
    )

    # 计算完整 CKA 矩阵
    cka_matrix = np.zeros((len(t_layers), len(s_layers)))  # (n_T, n_S)

    for i, t_layer in enumerate(t_layers):
        t_data = teacher_imprints_by_layer[t_layer]  # (N, d_T)
        for j, s_layer in enumerate(s_layers):
            s_data = student_imprints_by_layer[s_layer]  # (N, d_S)
            cka_matrix[i, j] = linear_cka(t_data, s_data)

    # 对每个教师层，选 CKA 最高的学生层
    mapping: dict[int, int] = {}
    for i, t_layer in enumerate(t_layers):
        best_j = int(np.argmax(cka_matrix[i]))
        best_s_layer = s_layers[best_j]
        mapping[t_layer] = best_s_layer

    # 打印结果
    _log_cka_matrix(cka_matrix, t_layers, s_layers, mapping)

    return mapping, cka_matrix


# ======================================================================
# 端到端接口（从模型 + 对比数据）
# ======================================================================

def find_layer_mapping_from_models(
    teacher_model: nn.Module,
    student_model: nn.Module,
    clean_inputs: torch.Tensor,           # (N, seq_len)
    corrupt_inputs: torch.Tensor,         # (N, seq_len)
    teacher_layers: list[int],
    token_positions: list[int],
    student_layers: list[int] | None = None,
) -> tuple[dict[int, int], np.ndarray]:
    """端到端层匹配：从模型和对比数据直接计算。

    对教师和学生模型分别提取因果痕迹，然后用 CKA 搜索最佳映射。
    对于给定的多个 token_positions，每层的痕迹在 token 维度上拼接
    后再做 CKA，以综合考虑多个关键位置的信息。

    Args:
        teacher_model: 教师模型（GPT-2 风格）。
        student_model: 学生模型（GPT-2 风格）。
        clean_inputs: clean 版本的 token ids, shape (N, seq_len).
        corrupt_inputs: corrupt 版本的 token ids, shape (N, seq_len).
        teacher_layers: 教师检查点层列表。
        token_positions: 关键 token 位置列表（用于提取痕迹）。
        student_layers: 学生搜索层列表；None 则搜索全部层。

    Returns:
        mapping: {teacher_layer: best_student_layer} 字典。
        cka_matrix: (n_teacher_layers, n_student_layers) 的完整 CKA 矩阵。
    """
    assert clean_inputs.shape == corrupt_inputs.shape, (
        f"Shape mismatch: clean {clean_inputs.shape} vs corrupt {corrupt_inputs.shape}"
    )
    assert len(teacher_layers) > 0, "teacher_layers is empty"
    assert len(token_positions) > 0, "token_positions is empty"

    n_student_layers = len(student_model.transformer.h)
    if student_layers is None:
        student_layers = list(range(n_student_layers))

    # ── 提取教师痕迹 ──────────────────────────────────────────────────
    logger.info("Extracting teacher imprints for %d layers...", len(teacher_layers))
    teacher_checkpoints = [
        (l, t) for l in teacher_layers for t in token_positions
    ]
    teacher_imprints_raw = extract_causal_imprints(
        teacher_model, clean_inputs, corrupt_inputs, teacher_checkpoints,
    )

    # 按层聚合：对每层拼接所有 token 位置的痕迹
    teacher_by_layer = _aggregate_by_layer(
        teacher_imprints_raw, teacher_layers, token_positions,
    )

    # ── 提取学生痕迹 ──────────────────────────────────────────────────
    logger.info("Extracting student imprints for %d layers...", len(student_layers))
    student_checkpoints = [
        (l, t) for l in student_layers for t in token_positions
    ]
    student_imprints_raw = extract_causal_imprints(
        student_model, clean_inputs, corrupt_inputs, student_checkpoints,
    )

    student_by_layer = _aggregate_by_layer(
        student_imprints_raw, student_layers, token_positions,
    )

    # ── CKA 搜索 ─────────────────────────────────────────────────────
    return find_best_layer_mapping(teacher_by_layer, student_by_layer)


# ======================================================================
# 内部工具
# ======================================================================

def _aggregate_by_layer(
    imprints: dict[tuple[int, int], torch.Tensor],  # {(l, t): (N, d)}
    layers: list[int],
    token_positions: list[int],
) -> dict[int, torch.Tensor]:  # {l: (N, d * n_positions)}
    """将同一层、不同 token 位置的痕迹拼接为单个张量。

    对每层 l，将 [imprints[(l, t1)], imprints[(l, t2)], ...]
    在特征维度上 concat，得到 (N, d * n_positions)。

    Args:
        imprints: {(layer, token_pos): (N, d_model)} 的痕迹字典。
        layers: 层索引列表。
        token_positions: token 位置列表。

    Returns:
        {layer: (N, d_model * n_positions)} 的聚合字典。
    """
    result: dict[int, torch.Tensor] = {}

    for layer in layers:
        parts: list[torch.Tensor] = []
        for t in token_positions:
            key = (layer, t)
            assert key in imprints, (
                f"Missing imprint for checkpoint ({layer}, {t})"
            )
            parts.append(imprints[key])  # (N, d_model)

        # 在特征维度上拼接
        concatenated = torch.cat(parts, dim=-1)  # (N, d_model * n_positions)
        result[layer] = concatenated

    return result


def _log_cka_matrix(
    cka_matrix: np.ndarray,           # (n_T, n_S)
    t_layers: list[int],
    s_layers: list[int],
    mapping: dict[int, int],
) -> None:
    """打印完整的 CKA 矩阵和最终层映射结果。"""
    n_t, n_s = cka_matrix.shape

    logger.info("=" * 70)
    logger.info("CKA Layer Matching Results")
    logger.info("=" * 70)

    # 表头
    header = "Teacher \\ Student |"
    for s_l in s_layers:
        header += f"  S-L{s_l:>2} "
    header += "| Best"
    logger.info(header)
    logger.info("-" * len(header))

    # 矩阵行
    for i, t_layer in enumerate(t_layers):
        row = f"    T-L{t_layer:>2}          |"
        best_j = int(np.argmax(cka_matrix[i]))
        for j in range(n_s):
            marker = " *" if j == best_j else "  "
            row += f" {cka_matrix[i, j]:.3f}{marker}"
        row += f"| S-L{mapping[t_layer]}"
        logger.info(row)

    logger.info("-" * 70)

    # 最终映射汇总
    logger.info("Final mapping:")
    for t_layer in t_layers:
        s_layer = mapping[t_layer]
        i = t_layers.index(t_layer)
        j = s_layers.index(s_layer)
        score = cka_matrix[i, j]
        logger.info(
            "  Teacher L%d → Student L%d  (CKA = %.4f)",
            t_layer, s_layer, score,
        )
    logger.info("=" * 70)
