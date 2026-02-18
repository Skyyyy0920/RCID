"""Perplexity evaluation on a text corpus."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def evaluate_perplexity(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    texts: list[str],
    max_length: int = 512,
    stride: int = 256,
) -> float:
    """Compute perplexity on a list of texts using sliding window.

    Args:
        model: Language model.
        tokenizer: Tokenizer for encoding texts.
        texts: List of text strings.
        max_length: Maximum sequence length per window.
        stride: Step size for sliding window.

    Returns:
        Perplexity (lower = better).
    """
    device = next(model.parameters()).device
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(input_ids) < 2:
            continue

        for start in range(0, len(input_ids) - 1, stride):
            end = min(start + max_length, len(input_ids))
            chunk = torch.tensor([input_ids[start:end]], device=device)

            with torch.no_grad():
                logits = model(chunk).logits  # (1, seq, vocab)

            # NLL: compare logits[:-1] with targets[1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:].contiguous()

            nll = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            n_tokens = shift_labels.numel()
            total_nll += nll.item()
            total_tokens += n_tokens

            if end == len(input_ids):
                break

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_nll / total_tokens
    return math.exp(avg_nll)
