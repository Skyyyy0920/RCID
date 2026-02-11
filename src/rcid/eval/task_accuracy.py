"""特定任务准确率评估（IOI、Greater-than 等）。

统一接口：evaluate(model, dataset, **kwargs) -> dict[str, float]

IOI 准确率：模型在最后位置预测的 top-1 token 是否为间接宾语名字。
Greater-Than 准确率：模型预测的数字是否大于阈值。
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def evaluate(
    model: nn.Module,
    dataset: DataLoader[dict[str, torch.Tensor]],
    *,
    task: str = "ioi",
    device: str | None = None,
) -> dict[str, float]:
    """评估模型在特定任务上的准确率。

    Args:
        model: GPT-2 风格模型（需具备 model.transformer.h 属性）。
        dataset: DataLoader，产出 batch 字典。
            IOI batch: {clean_ids, corrupt_ids, io_token_pos,
                        s2_token_pos, answer_token_id}
            GT batch:  {clean_ids, corrupt_ids, year_token_pos,
                        clean_threshold, corrupt_threshold}
        task: 任务类型，"ioi" 或 "greater_than"。
        device: 目标设备；None 则自动检测。

    Returns:
        {"accuracy": float} — 准确率，范围 [0, 1]。
    """
    assert task in ("ioi", "greater_than"), f"Unknown task: {task!r}"

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch in dataset:
            if task == "ioi":
                correct, count = _eval_ioi_batch(model, batch, device)
            else:
                correct, count = _eval_gt_batch(model, batch, device)
            total_correct += correct
            total_count += count

    accuracy = total_correct / max(total_count, 1)

    logger.info(
        "Task=%s | accuracy=%.4f (%d/%d)",
        task, accuracy, total_correct, total_count,
    )
    return {"accuracy": accuracy}


# ======================================================================
# IOI 评估
# ======================================================================

def _eval_ioi_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[int, int]:
    """单个 batch 的 IOI 准确率计算。

    对每个样本，取 clean_ids 的最后一个 token 位置的 logits，
    判断 argmax 是否等于 answer_token_id。

    Returns:
        (correct_count, batch_size)
    """
    clean_ids = batch["clean_ids"].to(device)          # (B, seq_len)
    answer_ids = batch["answer_token_id"].to(device)   # (B,)

    logits = model(clean_ids).logits  # (B, seq_len, vocab_size)

    # 最后一个非 pad 位置的 logits — IOI 任务在句尾预测
    # 由于同一 batch 内序列等长（collate 已 pad），取 seq_len-1
    last_pos = clean_ids.shape[1] - 1
    last_logits = logits[:, last_pos, :]  # (B, vocab_size)
    predictions = last_logits.argmax(dim=-1)  # (B,)

    correct = (predictions == answer_ids).sum().item()
    return int(correct), clean_ids.shape[0]


# ======================================================================
# Greater-Than 评估
# ======================================================================

# GPT-2 tokenizer 中两位数字 token: " 00" ~ " 99"
# 预构建 token_id -> 数值 映射
_DIGIT_TOKEN_CACHE: dict[int, int] | None = None


def _get_digit_tokens(model: nn.Module) -> dict[int, int]:
    """构建两位数字 token_id -> int 值的映射。

    GPT-2 对 " 00" ~ " 99" 的编码为单个 token。
    缓存结果避免重复计算。
    """
    global _DIGIT_TOKEN_CACHE
    if _DIGIT_TOKEN_CACHE is not None:
        return _DIGIT_TOKEN_CACHE

    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    mapping: dict[int, int] = {}
    for val in range(100):
        text = f" {val:02d}"  # " 00", " 01", ..., " 99"
        ids = tokenizer.encode(text)
        if len(ids) == 1:
            mapping[ids[0]] = val

    assert len(mapping) >= 90, (
        f"Expected >=90 single-token two-digit numbers, got {len(mapping)}"
    )

    _DIGIT_TOKEN_CACHE = mapping
    return mapping


def _eval_gt_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[int, int]:
    """单个 batch 的 Greater-Than 准确率计算。

    对每个样本，取 clean_ids 最后位置的 logits，
    在所有两位数字 token 中找 argmax，
    判断其数值是否 > clean_threshold。

    Returns:
        (correct_count, batch_size)
    """
    clean_ids = batch["clean_ids"].to(device)              # (B, seq_len)
    thresholds = batch["clean_threshold"].to(device)       # (B,)

    logits = model(clean_ids).logits  # (B, seq_len, vocab_size)
    last_pos = clean_ids.shape[1] - 1
    last_logits = logits[:, last_pos, :]  # (B, vocab_size)

    digit_map = _get_digit_tokens(model)
    digit_ids = sorted(digit_map.keys())
    digit_vals = torch.tensor(
        [digit_map[tid] for tid in digit_ids],
        device=device,
    )  # (n_digits,)
    digit_ids_t = torch.tensor(digit_ids, device=device)  # (n_digits,)

    # 只看两位数字 token 的 logits
    digit_logits = last_logits[:, digit_ids_t]  # (B, n_digits)
    best_idx = digit_logits.argmax(dim=-1)  # (B,)
    predicted_vals = digit_vals[best_idx]  # (B,)

    correct = (predicted_vals > thresholds).sum().item()
    return int(correct), clean_ids.shape[0]
