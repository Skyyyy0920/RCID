"""因果干预一致性评估，验证学生是否保留教师的回路行为。

核心思想：
对教师和学生施加 *相同* 的 activation patching 干预
（在检查点位置将 clean 残差流替换为 corrupt 残差流），
观察两者输出变化方向是否一致。

指标：
    consistency = 样本中输出变化方向一致的比例
    "一致" 定义为：两者在目标 token 上的 logit 差值同号
    （即 patching 后教师 logit 下降 ↔ 学生 logit 也下降）

统一接口：evaluate(model, dataset, **kwargs) -> dict[str, float]
"""

from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def evaluate(
    model: nn.Module,
    dataset: DataLoader[dict[str, torch.Tensor]],
    *,
    teacher_model: nn.Module,
    checkpoints: list[tuple[int, int]],
    layer_mapping: dict[int, int],
    device: str | None = None,
) -> dict[str, float]:
    """评估学生与教师的因果干预一致性。

    Args:
        model: 学生 GPT-2 风格模型。
        dataset: DataLoader，产出 IOI/GT batch 字典
            （需含 clean_ids, corrupt_ids, answer_token_id 或等效键）。
        teacher_model: 教师 GPT-2 风格模型。
        checkpoints: 因果检查点列表 [(teacher_layer, token_pos), ...]。
        layer_mapping: 教师层→学生层映射 {teacher_layer: student_layer}。
        device: 目标设备；None 则自动检测。

    Returns:
        {"consistency": float,            # 整体一致性 [0, 1]
         "consistency_per_cp": float,     # 各检查点平均一致性
         "n_checkpoints": float}          # 检查点数量
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()
    teacher_model.eval()

    # 每个检查点独立统计一致性
    cp_correct: dict[tuple[int, int], int] = defaultdict(int)
    cp_count: dict[tuple[int, int], int] = defaultdict(int)

    with torch.no_grad():
        for batch in dataset:
            clean_ids = batch["clean_ids"].to(device)      # (B, seq_len)
            corrupt_ids = batch["corrupt_ids"].to(device)   # (B, seq_len)

            # 确定 answer token 位置（IOI 用 answer_token_id）
            answer_ids = batch.get("answer_token_id")       # (B,) or None
            if answer_ids is not None:
                answer_ids = answer_ids.to(device)

            last_pos = clean_ids.shape[1] - 1

            # ── 基线 logits（不干预）──────────────────
            teacher_base = teacher_model(clean_ids).logits[:, last_pos, :]  # (B, V)
            student_base = model(clean_ids).logits[:, last_pos, :]          # (B, V)

            for t_layer, t_pos in checkpoints:
                s_layer = layer_mapping[t_layer]

                # ── 教师：patching 后的 logits ────────
                teacher_patched = _patched_logits(
                    teacher_model, clean_ids, corrupt_ids,
                    layer=t_layer, token_pos=t_pos,
                )[:, last_pos, :]  # (B, V)

                # ── 学生：对应位置 patching ────────────
                student_patched = _patched_logits(
                    model, clean_ids, corrupt_ids,
                    layer=s_layer, token_pos=t_pos,
                )[:, last_pos, :]  # (B, V)

                # ── 计算一致性 ────────────────────────
                if answer_ids is not None:
                    # IOI: 只看 answer token 的 logit 变化方向
                    t_delta = _gather_logit_delta(
                        teacher_base, teacher_patched, answer_ids,
                    )  # (B,)
                    s_delta = _gather_logit_delta(
                        student_base, student_patched, answer_ids,
                    )  # (B,)
                else:
                    # 通用: 看 argmax token 的 logit 变化
                    top_tokens = teacher_base.argmax(dim=-1)  # (B,)
                    t_delta = _gather_logit_delta(
                        teacher_base, teacher_patched, top_tokens,
                    )
                    s_delta = _gather_logit_delta(
                        student_base, student_patched, top_tokens,
                    )

                # 同号 = 一致（包括都为 0 的情况）
                consistent = ((t_delta * s_delta) >= 0).sum().item()
                cp_correct[(t_layer, t_pos)] += int(consistent)
                cp_count[(t_layer, t_pos)] += clean_ids.shape[0]

    # ── 汇总 ──────────────────────────────────────────
    total_correct = sum(cp_correct.values())
    total_count = sum(cp_count.values())
    overall = total_correct / max(total_count, 1)

    per_cp_scores: list[float] = []
    for cp in checkpoints:
        if cp_count[cp] > 0:
            score = cp_correct[cp] / cp_count[cp]
            per_cp_scores.append(score)
            logger.info(
                "Checkpoint (%d, %d): consistency=%.4f (%d/%d)",
                cp[0], cp[1], score, cp_correct[cp], cp_count[cp],
            )

    avg_per_cp = (
        sum(per_cp_scores) / len(per_cp_scores) if per_cp_scores else 0.0
    )

    logger.info(
        "Causal consistency: overall=%.4f, per_cp_avg=%.4f, n_cp=%d",
        overall, avg_per_cp, len(checkpoints),
    )

    return {
        "consistency": overall,
        "consistency_per_cp": avg_per_cp,
        "n_checkpoints": float(len(checkpoints)),
    }


# ======================================================================
# Activation Patching — 在指定位置注入 corrupt 残差流
# ======================================================================

@contextmanager
def _patch_hook(
    model: nn.Module,
    layer: int,
    token_pos: int,
    corrupt_residual: torch.Tensor,
) -> Iterator[None]:
    """注册 forward hook，将指定层、位置的残差流替换为 corrupt 版本。

    Args:
        model: GPT-2 风格模型。
        layer: 要 patch 的层索引。
        token_pos: 要 patch 的 token 位置。
        corrupt_residual: corrupt 残差流 (B, seq_len, d_model)。
    """
    def hook(
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden = output[0].clone()  # (B, seq_len, d_model)
        hidden[:, token_pos, :] = corrupt_residual[:, token_pos, :]
        return (hidden,) + output[1:]

    handle = model.transformer.h[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _get_layer_residual(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """前向传播并提取指定层的残差流。

    Returns:
        residual: (B, seq_len, d_model)
    """
    storage: dict[str, torch.Tensor] = {}

    def hook(
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, ...],
    ) -> None:
        storage["r"] = output[0]  # (B, seq_len, d_model)

    handle = model.transformer.h[layer].register_forward_hook(hook)
    try:
        model(input_ids)
    finally:
        handle.remove()

    return storage["r"]  # (B, seq_len, d_model)


def _patched_logits(
    model: nn.Module,
    clean_ids: torch.Tensor,       # (B, seq_len)
    corrupt_ids: torch.Tensor,     # (B, seq_len)
    layer: int,
    token_pos: int,
) -> torch.Tensor:
    """对模型做 activation patching 并返回 patched logits。

    流程：
    1. 用 corrupt_ids 前向传播，提取 layer 的残差流
    2. 用 clean_ids 前向传播，但在 layer 的 token_pos 位置
       注入 corrupt 残差流
    3. 返回 patched logits

    Returns:
        logits: (B, seq_len, vocab_size)
    """
    # Step 1: 获取 corrupt 残差流
    corrupt_residual = _get_layer_residual(
        model, corrupt_ids, layer,
    )  # (B, seq_len, d_model)

    # Step 2: clean 前向 + patch
    with _patch_hook(model, layer, token_pos, corrupt_residual):
        outputs = model(clean_ids)

    return outputs.logits  # (B, seq_len, vocab_size)


def _gather_logit_delta(
    base_logits: torch.Tensor,       # (B, vocab_size)
    patched_logits: torch.Tensor,    # (B, vocab_size)
    token_ids: torch.Tensor,         # (B,)
) -> torch.Tensor:
    """计算指定 token 的 logit 变化量。

    Returns:
        delta: (B,) — patched - base，正值表示 logit 增大。
    """
    base_vals = base_logits.gather(
        1, token_ids.unsqueeze(1),
    ).squeeze(1)  # (B,)
    patched_vals = patched_logits.gather(
        1, token_ids.unsqueeze(1),
    ).squeeze(1)  # (B,)
    return patched_vals - base_vals  # (B,)
