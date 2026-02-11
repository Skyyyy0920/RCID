"""WikiText 困惑度评估，衡量通用语言建模能力。

在 WikiText-103 验证集上计算困惑度，用于检测蒸馏是否破坏了通用能力。
困惑度 = exp(average_cross_entropy_loss)。

统一接口：evaluate(model, dataset, **kwargs) -> dict[str, float]
"""

from __future__ import annotations

import logging
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def evaluate(
    model: nn.Module,
    dataset: DataLoader[dict[str, torch.Tensor]] | None = None,
    *,
    max_samples: int = 0,
    stride: int = 512,
    max_length: int = 1024,
    device: str | None = None,
) -> dict[str, float]:
    """计算模型在 WikiText-103 验证集上的困惑度。

    Args:
        model: GPT-2 风格模型。
        dataset: 可选 DataLoader；若为 None 则自动加载 WikiText-103。
        max_samples: 最大评估 token 数（0=全量）。
        stride: 滑动窗口步长（默认 512，即半窗重叠）。
        max_length: 单窗最大 token 数（默认 1024，GPT-2 上限）。
        device: 目标设备；None 则自动检测。

    Returns:
        {"perplexity": float,        # 困惑度 (越低越好)
         "avg_loss": float,          # 平均 cross-entropy loss
         "n_tokens": float}          # 评估的总 token 数
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()

    if dataset is not None:
        return _eval_from_dataloader(model, dataset, device)

    # 自动加载 WikiText-103 验证集
    input_ids = _load_wikitext103(device)  # (1, total_tokens)
    total_len = input_ids.shape[1]

    if max_samples > 0:
        total_len = min(total_len, max_samples)
        input_ids = input_ids[:, :total_len]

    logger.info(
        "Evaluating perplexity on %d tokens (stride=%d, max_length=%d)",
        total_len, stride, max_length,
    )

    # 滑动窗口计算 — 避免 OOM
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for begin in range(0, total_len, stride):
            end = min(begin + max_length, total_len)
            chunk = input_ids[:, begin:end]  # (1, chunk_len)

            logits = model(chunk).logits  # (1, chunk_len, vocab_size)

            # 只计算 [stride_start:] 位置的 loss，避免窗口重叠区域重复计数
            # 第一个窗口: 从位置 1 开始（因为位置 0 没有 target）
            # 后续窗口: 从位置 overlap_start 开始
            if begin == 0:
                loss_start = 1
            else:
                loss_start = max_length - stride

            # shift: logits[:-1] 预测 labels[1:]
            shift_logits = logits[:, loss_start - 1:end - begin - 1, :]  # (1, L, V)
            shift_labels = chunk[:, loss_start:]  # (1, L)

            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="sum",
            )

            n_tokens = shift_labels.numel()
            total_nll += loss.item()
            total_tokens += n_tokens

            if end >= total_len:
                break

    avg_loss = total_nll / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)

    logger.info(
        "Perplexity=%.2f (avg_loss=%.4f, n_tokens=%d)",
        perplexity, avg_loss, total_tokens,
    )

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "n_tokens": float(total_tokens),
    }


# ======================================================================
# WikiText-103 加载
# ======================================================================

def _load_wikitext103(device: torch.device) -> torch.Tensor:
    """加载 WikiText-103 验证集并 tokenize 为连续 token 序列。

    Returns:
        input_ids: (1, total_tokens)
    """
    from datasets import load_dataset
    from transformers import GPT2Tokenizer

    logger.info("Loading WikiText-103 validation set...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 拼接所有非空行为连续文本
    texts: list[str] = [
        t for t in ds["text"] if t.strip()  # type: ignore[union-attr]
    ]
    full_text = "\n".join(texts)

    token_ids = tokenizer.encode(full_text)
    logger.info("WikiText-103 validation: %d tokens", len(token_ids))

    return torch.tensor(
        [token_ids], dtype=torch.long, device=device,
    )  # (1, total_tokens)


# ======================================================================
# DataLoader-based 评估（用于自定义数据集）
# ======================================================================

class TextChunkDataset(Dataset[dict[str, torch.Tensor]]):
    """将长 token 序列切分为固定长度 chunk 的辅助数据集。"""

    def __init__(
        self,
        token_ids: list[int],
        chunk_size: int = 1024,
    ) -> None:
        self.chunks: list[torch.Tensor] = []
        for i in range(0, len(token_ids) - 1, chunk_size):
            chunk = token_ids[i : i + chunk_size + 1]
            if len(chunk) > 1:
                self.chunks.append(
                    torch.tensor(chunk, dtype=torch.long)
                )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        return {
            "input_ids": chunk[:-1],   # (chunk_size,)
            "labels": chunk[1:],       # (chunk_size,)
        }


def _eval_from_dataloader(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    """从 DataLoader 计算困惑度。

    batch 需含 input_ids 和 labels（或 clean_ids 作为 fallback）。
    """
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            if "input_ids" in batch:
                ids = batch["input_ids"].to(device)      # (B, seq_len)
                labels = batch["labels"].to(device)      # (B, seq_len)
            elif "clean_ids" in batch:
                ids = batch["clean_ids"].to(device)      # (B, seq_len)
                labels = ids[:, 1:]                       # (B, seq_len-1)
                ids = ids[:, :-1]                         # (B, seq_len-1)
            else:
                raise KeyError("Batch must contain 'input_ids' or 'clean_ids'")

            logits = model(ids).logits  # (B, seq_len, vocab_size)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            )
            total_nll += loss.item()
            total_tokens += labels.numel()

    avg_loss = total_nll / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100.0))  # clamp 防止溢出

    logger.info(
        "Perplexity=%.2f (avg_loss=%.4f, n_tokens=%d)",
        perplexity, avg_loss, total_tokens,
    )

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "n_tokens": float(total_tokens),
    }
