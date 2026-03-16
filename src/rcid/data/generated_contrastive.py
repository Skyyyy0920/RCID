"""Load auto-generated contrastive pairs from JSON for RCID training.

Wraps the output of ``scripts/generate_contrastive_pairs.py`` into a
torch Dataset compatible with the RCID training loop.

Each JSON record contains ``{"clean": str, "corrupt": str, "task_type": str}``.
The dataset tokenises pairs, pads to equal length, and provides batched
access.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class GeneratedContrastiveDataset(Dataset):
    """Dataset of auto-generated (clean, corrupt) contrastive pairs.

    Attributes
    ----------
    clean_ids : (N, seq_len) long tensor
    corrupt_ids : (N, seq_len) long tensor
    seq_len : int
    is_modified : dict[str, bool]
        Position metadata (simplified for generated pairs).
    key_positions : dict[str, torch.Tensor]
        Named positions — for generated pairs this contains
        ``"modified"`` with the first differing token index.
    """

    def __init__(
        self,
        json_path: str | Path,
        tokenizer: Any,
        max_seq_len: int = 512,
        **kwargs: Any,
    ) -> None:
        path = Path(json_path)
        assert path.exists(), f"Contrastive pairs file not found: {path}"

        records = self._load_json(path)
        logger.info("Loaded %d raw records from %s", len(records), path)

        self._build(records, tokenizer, max_seq_len)

    @classmethod
    def from_directory(
        cls,
        dir_path: str | Path,
        tokenizer: Any,
        max_seq_len: int = 512,
        task_types: list[str] | None = None,
        **kwargs: Any,
    ) -> "GeneratedContrastiveDataset":
        """Load from a directory of per-task JSON files.

        Parameters
        ----------
        dir_path : path
            Directory containing ``entity_swap.json``, ``number_perturb.json``, etc.
        task_types : list[str] | None
            If specified, only load these task types. ``None`` loads all.
        """
        dir_p = Path(dir_path)
        assert dir_p.is_dir(), f"Not a directory: {dir_p}"

        all_records: list = []
        for json_file in sorted(dir_p.glob("*.json")):
            if json_file.name == "generation_summary.json":
                continue
            task_name = json_file.stem
            if task_types is not None and task_name not in task_types:
                continue
            records = cls._load_json(json_file)
            logger.info("Loaded %d records from %s", len(records), json_file.name)
            all_records.extend(records)

        logger.info("Total: %d contrastive pairs from directory %s", len(all_records), dir_p)
        instance = cls.__new__(cls)
        instance._build(all_records, tokenizer, max_seq_len)
        return instance

    @staticmethod
    def _load_json(path: Path) -> list:
        """Load records from a JSON or JSONL file, handling multiple formats.

        Supported formats:
        - JSON array: ``[{...}, {...}, ...]``
        - JSON dict with a wrapping key: ``{"pairs": [...]}`` or ``{"data": [...]}``
        - JSONL: one JSON object per line
        """
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning("Empty file: %s", path)
            return []

        # Try standard JSON first
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try JSONL (one JSON object per line)
            records: list = []
            for line_no, line in enumerate(text.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("JSONL parse failed at line %d in %s", line_no, path)
            if records:
                logger.info("Parsed %s as JSONL (%d records)", path.name, len(records))
            return records

        # JSON parsed successfully — normalise to a list of records
        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            # Check for wrapping keys: "pairs", "data", "records", "samples"
            for key in ("pairs", "data", "records", "samples", "examples"):
                if key in data and isinstance(data[key], list):
                    logger.info("Unwrapped JSON key %r (%d records) from %s",
                                key, len(data[key]), path.name)
                    return data[key]
            # Last resort: find the first list-valued key
            for key, val in data.items():
                if isinstance(val, list) and len(val) > 0:
                    logger.info("Unwrapped JSON key %r (%d records) from %s",
                                key, len(val), path.name)
                    return val
            # Dict itself might be a single record
            logger.info("Treating top-level dict as a single record from %s", path.name)
            return [data]

        logger.warning("Unexpected JSON type %s in %s", type(data).__name__, path)
        return [data]

    @staticmethod
    def _parse_record(rec: Any) -> tuple[str, str] | None:
        """Extract (clean, corrupt) strings from a record.

        Supports multiple JSON formats:
        - dict with ``"clean"``/``"corrupt"`` keys (canonical)
        - dict with ``"original"``/``"perturbed"`` keys
        - dict with ``"text_a"``/``"text_b"`` keys
        - list/tuple of two strings ``[clean, corrupt]``
        """
        if isinstance(rec, dict):
            clean = rec.get("clean") or rec.get("original") or rec.get("text_a")
            corrupt = rec.get("corrupt") or rec.get("perturbed") or rec.get("text_b")
            if isinstance(clean, str) and isinstance(corrupt, str):
                return clean, corrupt
        elif isinstance(rec, (list, tuple)) and len(rec) >= 2:
            if isinstance(rec[0], str) and isinstance(rec[1], str):
                return rec[0], rec[1]
        return None

    def _build(
        self,
        records: list,
        tokenizer: Any,
        max_seq_len: int,
    ) -> None:
        """Tokenise and pad all pairs."""
        clean_list: list[torch.Tensor] = []
        corrupt_list: list[torch.Tensor] = []
        mod_positions: list[int] = []
        skipped = 0

        for rec in records:
            parsed = self._parse_record(rec)
            if parsed is None:
                skipped += 1
                continue
            clean_text, corrupt_text = parsed

            c_enc = tokenizer(
                clean_text, truncation=True, max_length=max_seq_len,
                return_tensors="pt",
            )
            x_enc = tokenizer(
                corrupt_text, truncation=True, max_length=max_seq_len,
                return_tensors="pt",
            )
            c_ids = c_enc.input_ids[0]    # (L_c,)
            x_ids = x_enc.input_ids[0]    # (L_x,)

            # Pad to same length
            max_len = max(c_ids.shape[0], x_ids.shape[0])
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

            if c_ids.shape[0] < max_len:
                c_ids = torch.cat([
                    c_ids,
                    torch.full((max_len - c_ids.shape[0],), pad_id, dtype=torch.long),
                ])
            if x_ids.shape[0] < max_len:
                x_ids = torch.cat([
                    x_ids,
                    torch.full((max_len - x_ids.shape[0],), pad_id, dtype=torch.long),
                ])

            clean_list.append(c_ids)
            corrupt_list.append(x_ids)

            # Find first differing position
            diff = (c_ids != x_ids).nonzero(as_tuple=False)
            mod_pos = diff[0].item() if len(diff) > 0 else 0
            mod_positions.append(mod_pos)

        # Pad all to same seq_len
        max_seq = max(c.shape[0] for c in clean_list) if clean_list else 1
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        self.clean_ids = torch.zeros(len(clean_list), max_seq, dtype=torch.long)
        self.corrupt_ids = torch.zeros(len(corrupt_list), max_seq, dtype=torch.long)

        for i, (c, x) in enumerate(zip(clean_list, corrupt_list)):
            L = c.shape[0]
            self.clean_ids[i, :L] = c
            self.corrupt_ids[i, :L] = x

        self.key_positions = {
            "modified": torch.tensor(mod_positions, dtype=torch.long),
        }
        self.is_modified = {"modified": True}
        self._seq_len = max_seq

        if skipped > 0:
            logger.warning("Skipped %d unparseable records", skipped)
        logger.info(
            "GeneratedContrastiveDataset: %d pairs, seq_len=%d",
            len(self), self._seq_len,
        )

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def __len__(self) -> int:
        return self.clean_ids.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "clean_ids": self.clean_ids[idx],       # (seq_len,)
            "corrupt_ids": self.corrupt_ids[idx],   # (seq_len,)
        }

    @staticmethod
    def collate_fn(
        batch: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        return {
            "clean_ids": torch.stack([b["clean_ids"] for b in batch]),
            "corrupt_ids": torch.stack([b["corrupt_ids"] for b in batch]),
        }
