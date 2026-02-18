"""ContrastiveDataset base class for all contrastive pair datasets."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class ContrastiveDataset(Dataset):
    """Base class for contrastive pair datasets.

    All datasets provide (clean, corrupt) input pairs where the difference
    is minimal and controlled. Each sample records token-level metadata
    for causal analysis.
    """

    def __init__(
        self,
        clean_ids: torch.Tensor,          # (N, seq_len)
        corrupt_ids: torch.Tensor,        # (N, seq_len)
        answer_pos: torch.Tensor,         # (N,)
        correct_token_id: torch.Tensor,   # (N,)
        wrong_token_id: torch.Tensor,     # (N,)
        key_positions: dict[str, torch.Tensor],  # {name: (N,)}
        is_modified: dict[str, bool],     # which positions differ
        model_family: str,                # "qwen3" or "llama3"
    ) -> None:
        assert clean_ids.shape == corrupt_ids.shape, (
            f"Shape mismatch: {clean_ids.shape} vs {corrupt_ids.shape}"
        )
        n = clean_ids.shape[0]
        assert answer_pos.shape == (n,)
        assert correct_token_id.shape == (n,)
        assert wrong_token_id.shape == (n,)
        for name, pos_tensor in key_positions.items():
            assert pos_tensor.shape == (n,), f"{name}: expected ({n},), got {pos_tensor.shape}"

        self.clean_ids = clean_ids
        self.corrupt_ids = corrupt_ids
        self.answer_pos = answer_pos
        self.correct_token_id = correct_token_id
        self.wrong_token_id = wrong_token_id
        self.key_positions = key_positions
        self.is_modified = is_modified
        self.model_family = model_family

    def __len__(self) -> int:
        return self.clean_ids.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | bool]:
        item: dict[str, torch.Tensor | int | bool] = {
            "clean_ids": self.clean_ids[idx],           # (seq_len,)
            "corrupt_ids": self.corrupt_ids[idx],       # (seq_len,)
            "answer_pos": self.answer_pos[idx],         # scalar
            "correct_token_id": self.correct_token_id[idx],  # scalar
            "wrong_token_id": self.wrong_token_id[idx],      # scalar
        }
        for name, pos_tensor in self.key_positions.items():
            item[f"pos_{name}"] = pos_tensor[idx]
        return item

    @staticmethod
    def collate_fn(
        batch: list[dict[str, torch.Tensor | int | bool]],
    ) -> dict[str, torch.Tensor]:
        """Stack a list of sample dicts into batched tensors."""
        keys = batch[0].keys()
        result: dict[str, torch.Tensor] = {}
        for key in keys:
            values = [sample[key] for sample in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = torch.tensor(values)
        return result

    @property
    def seq_len(self) -> int:
        return self.clean_ids.shape[1]

    def get_modified_positions(self) -> list[str]:
        """Return names of positions that differ between clean and corrupt."""
        return [name for name, modified in self.is_modified.items() if modified]

    def get_unmodified_positions(self) -> list[str]:
        """Return names of positions that are the same in clean and corrupt."""
        return [name for name, modified in self.is_modified.items() if not modified]
