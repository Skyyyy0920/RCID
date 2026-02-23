"""Load auto-generated contrastive pairs from JSON into a ContrastiveDataset.

Bridge between ``scripts/generate_contrastive_pairs.py`` (which produces
JSON files) and the RCID training pipeline (which consumes
``ContrastiveDataset`` objects).

Supports two JSON formats:

1. **Legacy (bare list)**: ``[{"clean": ..., "corrupt": ...}, ...]``
2. **Per-task envelope**: ``{"task_type": ..., "pairs": [...], ...}``

Usage::

    # Single file (legacy or envelope)
    ds = GeneratedContrastiveDataset(
        json_path="data/contrastive_pairs.json",
        tokenizer=tokenizer, teacher=teacher,
    )

    # Directory of per-task JSONs
    ds = GeneratedContrastiveDataset.from_directory(
        "data/contrastive_pairs/",
        tokenizer=tokenizer, teacher=teacher,
        task_types=["entity_swap", "number_perturb"],
    )

    # Multiple explicit files
    ds = GeneratedContrastiveDataset.from_multiple_files(
        ["data/entity_swap.json", "data/number_perturb.json"],
        tokenizer=tokenizer, teacher=teacher,
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


def _load_pairs_from_json(path: Path) -> list[dict[str, Any]]:
    """Load pairs from a JSON file, handling both envelope and bare-list."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        # Legacy bare-list format
        logger.info("Loaded %d pairs from %s (bare-list format)", len(data), path)
        return data
    elif isinstance(data, dict) and "pairs" in data:
        # Per-task envelope format
        pairs = data["pairs"]
        task_type = data.get("task_type", "unknown")
        logger.info(
            "Loaded %d pairs from %s (envelope: task_type=%s)",
            len(pairs), path, task_type,
        )
        return pairs
    else:
        raise ValueError(
            f"Unrecognised JSON format in {path}. "
            "Expected a list of dicts or a dict with a 'pairs' key."
        )


class GeneratedContrastiveDataset(ContrastiveDataset):
    """ContrastiveDataset built from auto-generated contrastive pairs.

    Accepts a single JSON file (legacy bare-list or per-task envelope), or
    use the ``from_directory`` / ``from_multiple_files`` classmethods to
    merge multiple per-task files.

    Parameters
    ----------
    json_path : str | Path
        Path to a single JSON file.
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
        raw = _load_pairs_from_json(Path(json_path))

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

    # ------------------------------------------------------------------
    # Classmethods for multi-file loading
    # ------------------------------------------------------------------

    @classmethod
    def _from_pairs(
        cls,
        pairs: list[dict[str, Any]],
        tokenizer: Any,
        teacher: nn.Module | None = None,
        max_seq_len: int = 256,
        source_filter: str | list[str] | None = None,
        device: str | torch.device = "cpu",
        model_family: str = "auto",
    ) -> "GeneratedContrastiveDataset":
        """Build a dataset from an already-loaded list of pair dicts.

        Writes a temporary JSON and delegates to ``__init__``.  This avoids
        duplicating the tokenisation / teacher-inference logic while keeping
        the constructor signature simple.
        """
        import tempfile, os

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(pairs, f, ensure_ascii=False)
            return cls(
                json_path=tmp_path,
                tokenizer=tokenizer,
                teacher=teacher,
                max_seq_len=max_seq_len,
                source_filter=source_filter,
                device=device,
                model_family=model_family,
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    @classmethod
    def from_directory(
        cls,
        dir_path: str | Path,
        tokenizer: Any,
        teacher: nn.Module | None = None,
        max_seq_len: int = 256,
        task_types: list[str] | None = None,
        source_filter: str | list[str] | None = None,
        device: str | torch.device = "cpu",
        model_family: str = "auto",
    ) -> "GeneratedContrastiveDataset":
        """Load and merge per-task JSONs from a directory.

        Parameters
        ----------
        dir_path : str | Path
            Directory containing per-task JSON files (e.g.
            ``entity_swap.json``, ``number_perturb.json``).
        task_types : list[str] | None
            If given, only load files whose stem matches one of these names.
            E.g. ``["entity_swap", "number_perturb"]`` loads only those two.
            If ``None``, all ``*.json`` files in the directory are loaded.

        Other parameters are passed through to the constructor.
        """
        dp = Path(dir_path)
        if not dp.is_dir():
            raise NotADirectoryError(f"Not a directory: {dp}")

        json_files = sorted(dp.glob("*.json"))
        if task_types is not None:
            allowed = set(task_types)
            json_files = [f for f in json_files if f.stem in allowed]

        if not json_files:
            raise FileNotFoundError(
                f"No matching JSON files in {dp} "
                f"(task_types filter={task_types})"
            )

        merged: list[dict[str, Any]] = []
        for jf in json_files:
            merged.extend(_load_pairs_from_json(jf))

        logger.info(
            "from_directory: merged %d pairs from %d files in %s",
            len(merged), len(json_files), dp,
        )

        return cls._from_pairs(
            merged,
            tokenizer=tokenizer,
            teacher=teacher,
            max_seq_len=max_seq_len,
            source_filter=source_filter,
            device=device,
            model_family=model_family,
        )

    @classmethod
    def from_multiple_files(
        cls,
        paths: list[str | Path],
        tokenizer: Any,
        teacher: nn.Module | None = None,
        max_seq_len: int = 256,
        source_filter: str | list[str] | None = None,
        device: str | torch.device = "cpu",
        model_family: str = "auto",
    ) -> "GeneratedContrastiveDataset":
        """Load and merge multiple JSON files (paths given explicitly).

        Each file may be in legacy bare-list or per-task envelope format.
        """
        merged: list[dict[str, Any]] = []
        for p in paths:
            merged.extend(_load_pairs_from_json(Path(p)))

        logger.info(
            "from_multiple_files: merged %d pairs from %d files",
            len(merged), len(paths),
        )

        return cls._from_pairs(
            merged,
            tokenizer=tokenizer,
            teacher=teacher,
            max_seq_len=max_seq_len,
            source_filter=source_filter,
            device=device,
            model_family=model_family,
        )
