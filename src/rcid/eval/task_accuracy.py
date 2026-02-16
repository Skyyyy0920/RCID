"""特定任务准确率评估（IOI、Greater-than、Induction、SVA、Docstring）。

统一接口：evaluate(model, dataset, **kwargs) -> dict[str, float]

IOI 准确率：模型在最后位置预测的 top-1 token 是否为间接宾语名字。
Greater-Than 准确率：模型预测的数字是否大于阈值。
Induction 准确率：模型在 target_pos 位置 top-1 预测是否为答案 token。
SVA 准确率：单数动词 logit 之和 > 复数动词 logit 之和（clean = 单数主语）。
Docstring 准确率：模型在末尾位置 top-1 预测是否为正确参数名。
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

VALID_TASKS = ("ioi", "greater_than", "induction", "sva", "docstring")

_DISPATCH: dict[str, type] = {}  # filled after function defs


def evaluate(
    model: nn.Module,
    dataset: DataLoader[dict[str, torch.Tensor]],
    *,
    task: str = "ioi",
    device: str | None = None,
) -> dict[str, float]:
    """评估模型在特定任务上的准确率。

    Args:
        model: GPT-2 风格模型。
        dataset: DataLoader，产出 batch 字典。
        task: "ioi" | "greater_than" | "induction" | "sva" | "docstring"。
        device: 目标设备；None 则自动检测。

    Returns:
        {"accuracy": float} — 准确率，范围 [0, 1]。
    """
    assert task in VALID_TASKS, (
        f"Unknown task: {task!r}, valid: {VALID_TASKS}"
    )

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()
    total_correct = 0
    total_count = 0

    eval_fn = _DISPATCH[task]

    with torch.no_grad():
        for batch in dataset:
            correct, count = eval_fn(model, batch, device)
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
    """IOI: argmax at last position == answer_token_id。"""
    clean_ids = batch["clean_ids"].to(device)          # (B, seq_len)
    answer_ids = batch["answer_token_id"].to(device)   # (B,)

    logits = model(clean_ids).logits  # (B, seq_len, vocab_size)
    last_pos = clean_ids.shape[1] - 1
    predictions = logits[:, last_pos, :].argmax(dim=-1)  # (B,)

    correct = (predictions == answer_ids).sum().item()
    return int(correct), clean_ids.shape[0]


# ======================================================================
# Greater-Than 评估
# ======================================================================

_DIGIT_TOKEN_CACHE: dict[int, int] | None = None


def _get_digit_tokens(model: nn.Module) -> dict[int, int]:
    """构建两位数字 token_id -> int 值的映射。缓存结果。"""
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
    """Greater-Than: 在两位数字 token 中预测值 > clean_threshold。"""
    clean_ids = batch["clean_ids"].to(device)              # (B, seq_len)
    thresholds = batch["clean_threshold"].to(device)       # (B,)

    logits = model(clean_ids).logits  # (B, seq_len, vocab_size)
    last_pos = clean_ids.shape[1] - 1
    last_logits = logits[:, last_pos, :]  # (B, vocab_size)

    digit_map = _get_digit_tokens(model)
    digit_ids = sorted(digit_map.keys())
    digit_vals = torch.tensor(
        [digit_map[tid] for tid in digit_ids], device=device,
    )  # (n_digits,)
    digit_ids_t = torch.tensor(digit_ids, device=device)  # (n_digits,)

    digit_logits = last_logits[:, digit_ids_t]  # (B, n_digits)
    best_idx = digit_logits.argmax(dim=-1)  # (B,)
    predicted_vals = digit_vals[best_idx]  # (B,)

    correct = (predicted_vals > thresholds).sum().item()
    return int(correct), clean_ids.shape[0]


# ======================================================================
# Induction 评估
# ======================================================================

def _eval_induction_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[int, int]:
    """Induction: argmax at target_pos == answer_token_id。"""
    clean_ids = batch["clean_ids"].to(device)              # (B, seq_len)
    target_pos = batch["target_pos"].to(device)            # (B,)
    answer_ids = batch["answer_token_id"].to(device)       # (B,)

    logits = model(clean_ids).logits  # (B, seq_len, vocab_size)
    B = clean_ids.shape[0]
    # 逐样本取 target_pos 处的 logits
    tgt_logits = logits[torch.arange(B, device=device), target_pos, :]  # (B, V)
    predictions = tgt_logits.argmax(dim=-1)  # (B,)

    correct = (predictions == answer_ids).sum().item()
    return int(correct), B


# ======================================================================
# SVA 评估
# ======================================================================

def _eval_sva_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[int, int]:
    """SVA: 单数动词 logit 之和 > 复数动词 logit 之和（clean = 单数主语）。"""
    clean_ids = batch["clean_ids"].to(device)              # (B, seq_len)
    verb_pos = batch["verb_pos"].to(device)                # (B,)
    sg_ids = batch["singular_verb_ids"].to(device)         # (n_verbs,)
    pl_ids = batch["plural_verb_ids"].to(device)           # (n_verbs,)

    logits = model(clean_ids).logits  # (B, seq_len, vocab_size)
    B = clean_ids.shape[0]
    v_logits = logits[torch.arange(B, device=device), verb_pos, :]  # (B, V)

    sg_score = v_logits[:, sg_ids].sum(dim=-1)  # (B,)
    pl_score = v_logits[:, pl_ids].sum(dim=-1)  # (B,)
    # Clean 是单数主语 → 应偏好单数动词
    correct = (sg_score > pl_score).sum().item()
    return int(correct), B


# ======================================================================
# Docstring 评估
# ======================================================================

def _eval_docstring_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[int, int]:
    """Docstring: argmax at target_pos == answer_token_id。"""
    clean_ids = batch["clean_ids"].to(device)              # (B, seq_len)
    target_pos = batch["target_pos"].to(device)            # (B,)
    answer_ids = batch["answer_token_id"].to(device)       # (B,)

    logits = model(clean_ids).logits  # (B, seq_len, vocab_size)
    B = clean_ids.shape[0]
    tgt_logits = logits[torch.arange(B, device=device), target_pos, :]  # (B, V)
    predictions = tgt_logits.argmax(dim=-1)  # (B,)

    correct = (predictions == answer_ids).sum().item()
    return int(correct), B


# ======================================================================
# 分派表
# ======================================================================

_DISPATCH = {
    "ioi": _eval_ioi_batch,
    "greater_than": _eval_gt_batch,
    "induction": _eval_induction_batch,
    "sva": _eval_sva_batch,
    "docstring": _eval_docstring_batch,
}
