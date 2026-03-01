"""Dolly-15K dataset utilities for RCID.

Provides fixed train/val/test splits and prompt formatting.
All splits are derived from a single ``split_dolly_dataset(seed=42)`` call
so that every consumer (instruction dataset, contrastive pair generator,
evaluation) uses identical partitions.

Dolly format maps to Alpaca-style prompts:
    Dolly ``instruction`` → Alpaca ``### Instruction``
    Dolly ``context``     → Alpaca ``### Input``
    Dolly ``response``    → Alpaca ``### Response``
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DOLLY_CACHE: dict[str, dict] = {}


def format_dolly_prompt(instruction: str, context: str = "") -> str:
    """Format a Dolly sample into Alpaca-style prompt.

    Dolly's 'context' maps to Alpaca's 'input' field.
    The prompt ends with ``### Response:\\n`` — no response content.
    """
    if context and context.strip():
        return (
            "Below is an instruction that describes a task, "
            "paired with further context.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{context}\n\n"
            "### Response:\n"
        )
    else:
        return (
            "Below is an instruction that describes a task.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:\n"
        )


def split_dolly_dataset(seed: int = 42) -> dict[str, Any]:
    """Load Dolly-15K and split into fixed train/val/test.

    Result is cached to avoid redundant loading.

    Returns
    -------
    dict
        ``{"train": Dataset(~14011), "val": Dataset(500), "test": Dataset(500)}``
    """
    cache_key = f"dolly_{seed}"
    if cache_key in _DOLLY_CACHE:
        return _DOLLY_CACHE[cache_key]

    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    ds = ds.shuffle(seed=seed)

    splits = {
        "test": ds.select(range(500)),
        "val": ds.select(range(500, 1000)),
        "train": ds.select(range(1000, len(ds))),
    }

    logger.info(
        "Dolly splits: train=%d, val=%d, test=%d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    _DOLLY_CACHE[cache_key] = splits
    return splits


def get_dolly_prompts(
    split: str = "train",
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Return (prompts, gt_responses) for a Dolly split.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    seed : int
        Shuffle seed (must match across all consumers).

    Returns
    -------
    prompts : list[str]
        Formatted instruction prompts (ending with ``### Response:\\n``).
    responses : list[str]
        Ground truth response strings.
    """
    splits = split_dolly_dataset(seed=seed)
    ds = splits[split]

    prompts, responses = [], []
    for item in ds:
        prompts.append(
            format_dolly_prompt(item["instruction"], item.get("context", ""))
        )
        responses.append(item["response"])
    return prompts, responses


def get_dolly_texts_for_contrastive(
    seed: int = 42,
) -> list[str]:
    """Return prompt-only texts from Dolly train split for contrastive pair generation.

    These are the texts fed into EntitySwapGenerator / NumberPerturbGenerator /
    LLMGenerator.  They include instruction + context but NOT the response, so
    that:
    - The 'last position' predicts the first response token.
    - Entity/number perturbations in instruction/context are causally relevant.

    Returns
    -------
    list[str]
        Formatted prompt strings (~14k).
    """
    prompts, _ = get_dolly_prompts(split="train", seed=seed)
    return prompts
