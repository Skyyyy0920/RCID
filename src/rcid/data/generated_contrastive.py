"""Load auto-generated contrastive pairs from JSON into a ContrastiveDataset.

Bridge between ``scripts/generate_contrastive_pairs.py`` (which produces
JSON files) and the RCID training pipeline (which consumes
``ContrastiveDataset`` objects).

Usage::

    from rcid.data.generated_contrastive import GeneratedContrastiveDataset

    dataset = GeneratedContrastiveDataset(
        json_path="data/contrastive_pairs.json",
        tokenizer=tokenizer,
        teacher=teacher,
        max_seq_len=256,
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from rcid.circuit.contrastive import ContrastiveDataset

logger = logging.getLogger(__name__)


class GeneratedContrastiveDataset(ContrastiveDataset):
    """ContrastiveDataset built from a JSON file of auto-generated pairs.

    The JSON file is expected to contain a list of dicts, each with at least
    ``clean`` and ``corrupt`` text fields (as produced by
    ``generate_contrastive_pairs.py``).

    Parameters
    ----------
    json_path : str | Path
        Path to the JSON file.
    tokenizer : Any
        HuggingFace tokenizer (must support ``__call__`` with
        ``return_tensors="pt"``).
    teacher : nn.Module | None
        Teacher model used to infer ``correct_token_id`` / ``wrong_token_id``
        via argmax at ``answer_pos``.  If None, these fields are set to 0.
    max_seq_len : int
        Maximum token length; sequences are truncated / padded to this.
    source_filter : str | list[str] | None
        Keep only pairs whose ``source`` field matches.  None = keep all.
    device : str | torch.device
        Device for teacher inference (only used when *teacher* is provided).
    model_family : str
        Value stored in ``self.model_family`` (e.g. ``"qwen3"``).
    """

    def __init__(
        self,
        json_path: str | Path,
        tokenizer: Any,
        teacher: nn.Module | None = None,
        max_seq_len: int = 256,
        source_filter: str | list[str] | None = None,
        device: str | torch.device = "cpu",
        model_family: str = "auto",
    ) -> None:
        # ── Load & filter JSON ───────────────────────────────────────
        with open(json_path, "r", encoding="utf-8") as f:
            raw: list[dict[str, Any]] = json.load(f)
        logger.info("Loaded %d raw pairs from %s", len(raw), json_path)

        if source_filter is not None:
            allowed = {source_filter} if isinstance(source_filter, str) else set(source_filter)
            raw = [r for r in raw if r.get("source") in allowed]
            logger.info("After source filter (%s): %d pairs", allowed, len(raw))

        if not raw:
            raise ValueError(f"No pairs remaining after filtering {json_path}")

        # ── Tokenize ─────────────────────────────────────────────────
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        clean_ids_list: list[torch.Tensor] = []
        corrupt_ids_list: list[torch.Tensor] = []
        modified_positions_list: list[list[int]] = []
        sources: list[str] = []

        for entry in raw:
            c_enc = tokenizer(
                entry["clean"], truncation=True, max_length=max_seq_len,
                return_tensors="pt",
            ).input_ids[0]  # (seq_c,)
            x_enc = tokenizer(
                entry["corrupt"], truncation=True, max_length=max_seq_len,
                return_tensors="pt",
            ).input_ids[0]  # (seq_x,)

            # Align lengths: truncate longer to shorter, then pad both to max_seq_len
            min_len = min(c_enc.shape[0], x_enc.shape[0])
            c_enc = c_enc[:min_len]
            x_enc = x_enc[:min_len]

            clean_ids_list.append(c_enc)
            corrupt_ids_list.append(x_enc)
            modified_positions_list.append(entry.get("modified_positions", []))
            sources.append(entry.get("source", "unknown"))

        # Pad to uniform length
        max_len = max(t.shape[0] for t in clean_ids_list)
        max_len = min(max_len, max_seq_len)

        n = len(clean_ids_list)
        clean_ids = torch.full((n, max_len), pad_id, dtype=torch.long)   # (N, L)
        corrupt_ids = torch.full((n, max_len), pad_id, dtype=torch.long)  # (N, L)

        for i, (c, x) in enumerate(zip(clean_ids_list, corrupt_ids_list)):
            L = min(c.shape[0], max_len)
            clean_ids[i, :L] = c[:L]
            corrupt_ids[i, :L] = x[:L]

        # ── answer_pos: last non-pad token ───────────────────────────
        # mask: 1 where token != pad_id
        non_pad = (clean_ids != pad_id).long()  # (N, L)
        # Count valid tokens per sample, clamp to at least 1
        lengths = non_pad.sum(dim=1).clamp(min=1)  # (N,)
        answer_pos = lengths - 1  # (N,) — 0-indexed last valid position

        # ── correct / wrong token ids ────────────────────────────────
        if teacher is not None:
            correct_ids, wrong_ids = self._infer_target_tokens(
                teacher, clean_ids, corrupt_ids, answer_pos, device,
            )
        else:
            correct_ids = torch.zeros(n, dtype=torch.long)
            wrong_ids = torch.zeros(n, dtype=torch.long)

        # ── key_positions & is_modified ──────────────────────────────
        # Re-derive modified positions from actual token ids (more reliable
        # than the JSON field, which may refer to a different tokenizer run).
        first_mod = torch.zeros(n, dtype=torch.long)
        has_mod = torch.zeros(n, dtype=torch.bool)
        for i in range(n):
            diff_mask = clean_ids[i] != corrupt_ids[i]
            positions = diff_mask.nonzero(as_tuple=False).squeeze(-1)
            if positions.numel() > 0:
                first_mod[i] = positions[0].item()
                has_mod[i] = True

        key_positions: dict[str, torch.Tensor] = {
            "modified": first_mod,  # (N,) first differing position
        }
        is_modified: dict[str, bool] = {
            "modified": True,
        }

        # ── Infer model_family ───────────────────────────────────────
        if model_family == "auto":
            # Best-effort: check tokenizer class name
            tok_name = type(tokenizer).__name__.lower()
            if "qwen" in tok_name:
                model_family = "qwen3"
            elif "llama" in tok_name:
                model_family = "llama3"
            else:
                model_family = "unknown"

        # ── Init base class ──────────────────────────────────────────
        super().__init__(
            clean_ids=clean_ids,
            corrupt_ids=corrupt_ids,
            answer_pos=answer_pos,
            correct_token_id=correct_ids,
            wrong_token_id=wrong_ids,
            key_positions=key_positions,
            is_modified=is_modified,
            model_family=model_family,
        )

        # Extra metadata (not in base class but useful for debugging)
        self.sources = sources
        logger.info(
            "GeneratedContrastiveDataset: N=%d, seq_len=%d, family=%s",
            n, max_len, model_family,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_target_tokens(
        teacher: nn.Module,
        clean_ids: torch.Tensor,    # (N, L)
        corrupt_ids: torch.Tensor,  # (N, L)
        answer_pos: torch.Tensor,   # (N,)
        device: str | torch.device,
        batch_size: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run teacher on clean/corrupt to get argmax token at answer_pos.

        Returns (correct_token_id, wrong_token_id), each shape (N,).
        """
        n = clean_ids.shape[0]
        correct = torch.zeros(n, dtype=torch.long)
        wrong = torch.zeros(n, dtype=torch.long)

        teacher.eval()
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            c_batch = clean_ids[start:end].to(device)       # (B, L)
            x_batch = corrupt_ids[start:end].to(device)     # (B, L)
            ans_batch = answer_pos[start:end]                # (B,)
            bs = c_batch.shape[0]
            idx = torch.arange(bs, device=device)

            with torch.no_grad():
                c_logits = teacher(c_batch).logits            # (B, L, V)
                x_logits = teacher(x_batch).logits            # (B, L, V)

            c_at_ans = c_logits[idx, ans_batch.to(device)]    # (B, V)
            x_at_ans = x_logits[idx, ans_batch.to(device)]    # (B, V)

            correct[start:end] = c_at_ans.argmax(dim=-1).cpu()
            wrong[start:end] = x_at_ans.argmax(dim=-1).cpu()

        return correct, wrong
