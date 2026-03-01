"""Instruction dataset for large-scale sequence-level KL distillation.

Loads instruction-tuning data (Alpaca, Dolly, SlimOrca, etc.) from
HuggingFace, tokenises each sample into a single sequence, and provides
masks that distinguish prompt tokens from response tokens so that the KL
loss can be applied selectively.

Dolly-15K receives special handling: its train/val/test splits are
obtained from :func:`rcid.data.dolly_utils.split_dolly_dataset` to
guarantee identical partitions everywhere.

Usage::

    from rcid.data.instruction_dataset import InstructionDataset

    ds = InstructionDataset(
        dataset_name="databricks/databricks-dolly-15k",
        tokenizer=tokenizer,
        max_seq_len=512,
        split="train",
    )
    loader = DataLoader(ds, batch_size=8, collate_fn=ds.collate_fn)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ── Prompt templates ─────────────────────────────────────────────────────

_ALPACA_NO_INPUT = (
    "Below is an instruction that describes a task.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)

_ALPACA_WITH_INPUT = (
    "Below is an instruction that describes a task, "
    "paired with further context.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

_ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

_ALPACA_PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, "
    "paired with further context.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


# ── Format helpers ───────────────────────────────────────────────────────


def _format_alpaca(sample: dict[str, Any]) -> tuple[str, str]:
    """Return (prompt, full_text) for an Alpaca-format sample."""
    instruction = sample.get("instruction", "")
    inp = sample.get("input", "")
    output = sample.get("output", "")

    if inp:
        prompt = _ALPACA_PROMPT_WITH_INPUT.format(instruction=instruction, input=inp)
        full = _ALPACA_WITH_INPUT.format(instruction=instruction, input=inp, output=output)
    else:
        prompt = _ALPACA_PROMPT_NO_INPUT.format(instruction=instruction)
        full = _ALPACA_NO_INPUT.format(instruction=instruction, output=output)
    return prompt, full


def _format_chat(sample: dict[str, Any]) -> tuple[str, str]:
    """Return (prompt, full_text) for a conversations-list format.

    Expects ``sample["conversations"]`` to be a list of dicts with
    ``role`` and ``content`` keys.
    """
    convs = sample.get("conversations", [])
    prompt_parts: list[str] = []
    full_parts: list[str] = []

    for turn in convs:
        role = turn.get("role", turn.get("from", ""))
        content = turn.get("content", turn.get("value", ""))
        line = f"{role}: {content}"
        full_parts.append(line)
        if role in ("user", "human", "system"):
            prompt_parts.append(line)

    prompt = "\n".join(prompt_parts) + "\nassistant: " if prompt_parts else ""
    full = "\n".join(full_parts)
    return prompt, full


def _format_dolly(sample: dict[str, Any]) -> tuple[str, str]:
    """Return (prompt, full_text) for a Dolly-format sample."""
    from rcid.data.dolly_utils import format_dolly_prompt

    instruction = sample.get("instruction", "")
    context = sample.get("context", "")
    response = sample.get("response", "")

    prompt = format_dolly_prompt(instruction, context)
    full = prompt + response
    return prompt, full


def _detect_and_format(sample: dict[str, Any]) -> tuple[str, str]:
    """Auto-detect format and return (prompt, full_text)."""
    if "instruction" in sample and "output" in sample:
        return _format_alpaca(sample)       # Alpaca: has 'output' field
    if "instruction" in sample and "response" in sample:
        return _format_dolly(sample)        # Dolly: has 'response' field
    if "instruction" in sample:
        return _format_alpaca(sample)       # Fallback for instruction-only
    if "conversations" in sample:
        return _format_chat(sample)
    # Fallback: use any available text field as both prompt and full
    for key in ("text", "content", "prompt"):
        if key in sample and sample[key]:
            text = str(sample[key])
            # No way to split prompt/response, treat all as response
            return "", text
    return "", ""


# ── Dataset class ────────────────────────────────────────────────────────


class InstructionDataset(Dataset):
    """HuggingFace instruction dataset tokenised for sequence-level KL.

    Each item provides:
    - ``input_ids``      (seq_len,) — full tokenised sequence
    - ``attention_mask``  (seq_len,) — 1 for real tokens, 0 for padding
    - ``labels_mask``     (seq_len,) — 1 for response tokens, 0 for prompt/pad
    """

    def __init__(
        self,
        dataset_name: str = "tatsu-lab/alpaca",
        tokenizer: Any = None,
        max_seq_len: int = 512,
        max_samples: int | None = None,
        split: str = "train",
    ) -> None:
        if tokenizer is None:
            raise ValueError("tokenizer is required")

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.pad_id = pad_id

        # Load from HuggingFace — Dolly uses fixed splits
        from datasets import load_dataset
        if "dolly" in dataset_name.lower():
            from rcid.data.dolly_utils import split_dolly_dataset
            all_splits = split_dolly_dataset(seed=42)
            raw = all_splits.get(split, all_splits["train"])
            logger.info(
                "Using Dolly fixed split: %s (%d samples)", split, len(raw),
            )
        else:
            raw = load_dataset(dataset_name, split=split)
        if max_samples is not None and len(raw) > max_samples:
            raw = raw.select(range(max_samples))
        logger.info("Loaded %d samples from %s/%s", len(raw), dataset_name, split)

        # Pre-tokenise everything
        self.input_ids: list[torch.Tensor] = []
        self.attention_masks: list[torch.Tensor] = []
        self.labels_masks: list[torch.Tensor] = []

        skipped = 0
        for sample in raw:
            prompt_text, full_text = _detect_and_format(sample)
            if not full_text:
                skipped += 1
                continue

            # Tokenise full sequence
            full_enc = tokenizer(
                full_text, truncation=True, max_length=max_seq_len,
                return_tensors="pt",
            )
            ids = full_enc.input_ids[0]         # (L,)
            attn = full_enc.attention_mask[0]    # (L,)

            # Tokenise prompt-only to find boundary
            if prompt_text:
                prompt_enc = tokenizer(
                    prompt_text, truncation=True, max_length=max_seq_len,
                    return_tensors="pt",
                )
                prompt_len = prompt_enc.input_ids.shape[1]
            else:
                prompt_len = 0

            # labels_mask: 1 for response tokens, 0 for prompt / padding
            labels_mask = torch.zeros_like(ids, dtype=torch.long)
            labels_mask[prompt_len:] = 1
            # Zero out any padding positions
            labels_mask = labels_mask * attn

            self.input_ids.append(ids)
            self.attention_masks.append(attn)
            self.labels_masks.append(labels_mask)

        if skipped:
            logger.info("Skipped %d empty samples", skipped)
        logger.info(
            "InstructionDataset ready: %d samples, max_seq_len=%d",
            len(self.input_ids), max_seq_len,
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],            # (seq_len,)
            "attention_mask": self.attention_masks[idx],  # (seq_len,)
            "labels_mask": self.labels_masks[idx],        # (seq_len,)
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Dynamically pad to the longest sequence in the batch."""
        max_len = max(b["input_ids"].shape[0] for b in batch)
        bs = len(batch)

        # Infer pad_id from the first sample (position 0 is never padding
        # in a well-formed sequence, so we can't rely on it; use 0 as safe default)
        pad_id = 0

        input_ids = torch.full((bs, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(bs, max_len, dtype=torch.long)
        labels_mask = torch.zeros(bs, max_len, dtype=torch.long)

        for i, b in enumerate(batch):
            L = b["input_ids"].shape[0]
            input_ids[i, :L] = b["input_ids"]
            attention_mask[i, :L] = b["attention_mask"]
            labels_mask[i, :L] = b["labels_mask"]

        return {
            "input_ids": input_ids,            # (batch, max_len)
            "attention_mask": attention_mask,   # (batch, max_len)
            "labels_mask": labels_mask,         # (batch, max_len)
        }
