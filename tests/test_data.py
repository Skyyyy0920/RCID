"""Tests for dataset classes using a mock tokenizer."""

from __future__ import annotations

import pytest
import torch

from rcid.circuit.contrastive import ContrastiveDataset
from rcid.data.ioi import IOIDataset, build_single_token_names
from rcid.data.factual_probing import FactualProbingDataset
from rcid.data.winogrande import WinoGrandeDataset


# ---------------------------------------------------------------------------
# Mock tokenizer that encodes each word (space-delimited) as one token
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Deterministic tokenizer: each whitespace-delimited word → one token."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._next_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.name_or_path = "mock-tokenizer"
        # Pre-populate some known tokens
        self._get_or_add("<pad>")
        self._get_or_add("<eos>")

    def _get_or_add(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = self._next_id
            self._next_id += 1
        return self._vocab[word]

    def encode(
        self, text: str, add_special_tokens: bool = True
    ) -> list[int]:
        # Strip leading space for lookup but keep words
        text = text.strip()
        if not text:
            return []
        words = text.split()
        return [self._get_or_add(w) for w in words]

    def decode(self, ids: list[int]) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        return " ".join(inv.get(i, f"<unk:{i}>") for i in ids)


# ---------------------------------------------------------------------------
# ContrastiveDataset base class tests
# ---------------------------------------------------------------------------

class TestContrastiveDataset:
    def test_basic_construction(self) -> None:
        n, seq_len = 5, 10
        ds = ContrastiveDataset(
            clean_ids=torch.randint(0, 100, (n, seq_len)),
            corrupt_ids=torch.randint(0, 100, (n, seq_len)),
            answer_pos=torch.zeros(n, dtype=torch.long),
            correct_token_id=torch.ones(n, dtype=torch.long),
            wrong_token_id=torch.full((n,), 2, dtype=torch.long),
            key_positions={"pos_a": torch.zeros(n, dtype=torch.long)},
            is_modified={"pos_a": True},
            model_family="test",
        )
        assert len(ds) == n
        assert ds.seq_len == seq_len

    def test_getitem_returns_dict(self) -> None:
        n = 3
        ds = ContrastiveDataset(
            clean_ids=torch.arange(30).reshape(n, 10),
            corrupt_ids=torch.arange(30).reshape(n, 10) + 100,
            answer_pos=torch.tensor([9, 9, 9]),
            correct_token_id=torch.tensor([1, 2, 3]),
            wrong_token_id=torch.tensor([4, 5, 6]),
            key_positions={"x": torch.tensor([0, 1, 2])},
            is_modified={"x": True},
            model_family="test",
        )
        item = ds[0]
        assert "clean_ids" in item
        assert "corrupt_ids" in item
        assert "pos_x" in item

    def test_collate_fn(self) -> None:
        n = 3
        ds = ContrastiveDataset(
            clean_ids=torch.arange(30).reshape(n, 10),
            corrupt_ids=torch.arange(30).reshape(n, 10),
            answer_pos=torch.tensor([9, 9, 9]),
            correct_token_id=torch.tensor([1, 2, 3]),
            wrong_token_id=torch.tensor([4, 5, 6]),
            key_positions={},
            is_modified={},
            model_family="test",
        )
        batch = ContrastiveDataset.collate_fn([ds[0], ds[1]])
        assert batch["clean_ids"].shape == (2, 10)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(AssertionError, match="Shape mismatch"):
            ContrastiveDataset(
                clean_ids=torch.zeros(3, 10, dtype=torch.long),
                corrupt_ids=torch.zeros(3, 12, dtype=torch.long),
                answer_pos=torch.zeros(3, dtype=torch.long),
                correct_token_id=torch.zeros(3, dtype=torch.long),
                wrong_token_id=torch.zeros(3, dtype=torch.long),
                key_positions={},
                is_modified={},
                model_family="test",
            )

    def test_modified_unmodified_positions(self) -> None:
        ds = ContrastiveDataset(
            clean_ids=torch.zeros(2, 5, dtype=torch.long),
            corrupt_ids=torch.zeros(2, 5, dtype=torch.long),
            answer_pos=torch.zeros(2, dtype=torch.long),
            correct_token_id=torch.zeros(2, dtype=torch.long),
            wrong_token_id=torch.zeros(2, dtype=torch.long),
            key_positions={
                "a": torch.zeros(2, dtype=torch.long),
                "b": torch.zeros(2, dtype=torch.long),
            },
            is_modified={"a": True, "b": False},
            model_family="test",
        )
        assert ds.get_modified_positions() == ["a"]
        assert ds.get_unmodified_positions() == ["b"]


# ---------------------------------------------------------------------------
# IOI dataset tests
# ---------------------------------------------------------------------------

class TestIOIDataset:
    def test_name_pool_mock(self) -> None:
        tok = MockTokenizer()
        names = build_single_token_names(tok)
        # All names should be single-token in mock tokenizer
        assert len(names) >= 20

    def test_ioi_construction(self) -> None:
        tok = MockTokenizer()
        ioi = IOIDataset(tokenizer=tok, n_samples=10, seed=42)
        ds = ioi.dataset
        assert len(ds) == 10
        assert ds.clean_ids.shape[0] == 10

    def test_clean_corrupt_differ_at_s2(self) -> None:
        tok = MockTokenizer()
        ioi = IOIDataset(tokenizer=tok, n_samples=5, seed=42)
        ds = ioi.dataset
        for i in range(len(ds)):
            clean = ds.clean_ids[i]
            corrupt = ds.corrupt_ids[i]
            diff_mask = clean != corrupt
            # At least one position should differ (S2)
            assert diff_mask.any(), f"Sample {i}: clean == corrupt"

    def test_is_modified_flag(self) -> None:
        tok = MockTokenizer()
        ioi = IOIDataset(tokenizer=tok, n_samples=5, seed=42)
        ds = ioi.dataset
        assert ds.is_modified["s2"] is True
        assert ds.is_modified["io"] is False
        assert ds.is_modified["end"] is False

    def test_model_family_set(self) -> None:
        tok = MockTokenizer()
        ioi = IOIDataset(tokenizer=tok, n_samples=5, seed=42)
        assert ioi.dataset.model_family == "mock-tokenizer"


# ---------------------------------------------------------------------------
# Factual Probing tests
# ---------------------------------------------------------------------------

class TestFactualProbingDataset:
    def test_construction(self) -> None:
        tok = MockTokenizer()
        fp = FactualProbingDataset(tokenizer=tok, seed=42)
        ds = fp.dataset
        assert len(ds) > 0

    def test_clean_corrupt_differ_at_entity(self) -> None:
        tok = MockTokenizer()
        fp = FactualProbingDataset(tokenizer=tok, seed=42)
        ds = fp.dataset
        for i in range(len(ds)):
            clean = ds.clean_ids[i]
            corrupt = ds.corrupt_ids[i]
            assert (clean != corrupt).any()

    def test_is_modified_entity(self) -> None:
        tok = MockTokenizer()
        fp = FactualProbingDataset(tokenizer=tok, seed=42)
        assert fp.dataset.is_modified["entity"] is True


# ---------------------------------------------------------------------------
# WinoGrande tests
# ---------------------------------------------------------------------------

class TestWinoGrandeDataset:
    def test_construction(self) -> None:
        tok = MockTokenizer()
        wg = WinoGrandeDataset(tokenizer=tok, seed=42)
        ds = wg.dataset
        assert len(ds) > 0

    def test_is_modified_positions(self) -> None:
        tok = MockTokenizer()
        wg = WinoGrandeDataset(tokenizer=tok, seed=42)
        assert wg.dataset.is_modified["modified"] is True
        assert wg.dataset.is_modified["pronoun"] is False

    def test_clean_corrupt_differ(self) -> None:
        tok = MockTokenizer()
        wg = WinoGrandeDataset(tokenizer=tok, seed=42)
        ds = wg.dataset
        for i in range(len(ds)):
            assert (ds.clean_ids[i] != ds.corrupt_ids[i]).any()
