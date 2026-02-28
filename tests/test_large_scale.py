"""Tests for large-scale distillation components.

Uses TinyTransformerModel (from conftest.py) so no external models are needed.
Each test is self-contained and creates its own synthetic data.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from conftest import TinyAdapter, TinyTransformerModel
from rcid.data.contrastive_generators import EntitySwapGenerator, NumberPerturbGenerator
from rcid.data.contrastive_validator import ContrastivePairValidator


# ==================================================================
# Fixtures
# ==================================================================


class _FakeTokenizer:
    """Minimal tokenizer for tests — character-level with pad support."""

    def __init__(self, vocab_size: int = 100) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab_size = vocab_size

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        truncation: bool = False,
        max_length: int = 512,
        **kwargs: object,
    ) -> SimpleNamespace:
        ids = [min(ord(c) % (self.vocab_size - 2) + 2, self.vocab_size - 1)
               for c in text]
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        t = torch.tensor([ids], dtype=torch.long)
        mask = torch.ones_like(t)
        return SimpleNamespace(input_ids=t, attention_mask=mask)

    def decode(self, ids: torch.Tensor, **kwargs: object) -> str:
        return "".join(chr(max(i.item(), 32)) for i in ids)


class _FakeInstructionDataset(Dataset):
    """Tiny instruction dataset that mimics InstructionDataset interface."""

    def __init__(self, n: int = 8, seq_len: int = 12, vocab_size: int = 100) -> None:
        self.input_ids = [
            torch.randint(2, vocab_size, (seq_len,)) for _ in range(n)
        ]
        self.attention_masks = [torch.ones(seq_len, dtype=torch.long) for _ in range(n)]
        self.labels_masks = [torch.ones(seq_len, dtype=torch.long) for _ in range(n)]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels_mask": self.labels_masks[idx],
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(b["input_ids"].shape[0] for b in batch)
        bs = len(batch)
        input_ids = torch.zeros(bs, max_len, dtype=torch.long)
        attention_mask = torch.zeros(bs, max_len, dtype=torch.long)
        labels_mask = torch.zeros(bs, max_len, dtype=torch.long)
        for i, b in enumerate(batch):
            L = b["input_ids"].shape[0]
            input_ids[i, :L] = b["input_ids"]
            attention_mask[i, :L] = b["attention_mask"]
            labels_mask[i, :L] = b["labels_mask"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels_mask": labels_mask,
        }


@pytest.fixture
def tiny_teacher() -> TinyTransformerModel:
    torch.manual_seed(0)
    return TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)


@pytest.fixture
def tiny_student() -> TinyTransformerModel:
    torch.manual_seed(1)
    return TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)


@pytest.fixture
def tiny_adapter() -> TinyAdapter:
    return TinyAdapter()


@pytest.fixture
def fake_tokenizer() -> _FakeTokenizer:
    return _FakeTokenizer(vocab_size=100)


# ==================================================================
# 1. InstructionDataset
# ==================================================================


class TestInstructionDatasetLoading:
    """Test InstructionDataset with fake Alpaca-format data."""

    def test_tokenize_and_pad(self) -> None:
        from rcid.data.instruction_dataset import _detect_and_format

        sample = {
            "instruction": "Translate to French.",
            "input": "Hello",
            "output": "Bonjour",
        }
        prompt, full = _detect_and_format(sample)
        assert "Translate to French" in prompt
        assert "Bonjour" in full
        assert prompt in full

    def test_collate_fn_shapes(self) -> None:
        ds = _FakeInstructionDataset(n=6, seq_len=10)
        batch = [ds[i] for i in range(4)]
        collated = ds.collate_fn(batch)
        assert collated["input_ids"].shape == (4, 10)

    def test_collate_fn_variable_lengths(self) -> None:
        ds = _FakeInstructionDataset(n=4, seq_len=8)
        ds.input_ids[0] = torch.randint(2, 100, (5,))
        ds.attention_masks[0] = torch.ones(5, dtype=torch.long)
        ds.labels_masks[0] = torch.ones(5, dtype=torch.long)

        batch = [ds[i] for i in range(3)]
        collated = ds.collate_fn(batch)
        assert collated["input_ids"].shape[1] == 8
        assert collated["input_ids"][0, 5:].sum() == 0


# ==================================================================
# 2. EntitySwapGenerator
# ==================================================================


class TestEntitySwapGenerator:
    def test_detects_and_swaps_country(self, tiny_teacher, fake_tokenizer) -> None:
        gen = EntitySwapGenerator(
            teacher=tiny_teacher, tokenizer=fake_tokenizer, seed=42,
        )
        pairs = gen.generate("The capital of France is Paris.")
        assert len(pairs) > 0
        for clean, corrupt in pairs:
            assert corrupt != clean

    def test_no_entities_returns_empty(self, tiny_teacher, fake_tokenizer) -> None:
        gen = EntitySwapGenerator(
            teacher=tiny_teacher, tokenizer=fake_tokenizer, seed=42,
        )
        pairs = gen.generate("The quick brown fox jumps over the lazy dog.")
        assert pairs == []


# ==================================================================
# 3. NumberPerturbGenerator
# ==================================================================


class TestNumberPerturbGenerator:
    def test_detects_and_perturbs_number(self, tiny_teacher, fake_tokenizer) -> None:
        gen = NumberPerturbGenerator(
            teacher=tiny_teacher, tokenizer=fake_tokenizer, seed=42,
        )
        pairs = gen.generate("John has 5 apples and gives 2 away.")
        assert len(pairs) > 0

    def test_no_numbers_returns_empty(self, tiny_teacher, fake_tokenizer) -> None:
        gen = NumberPerturbGenerator(
            teacher=tiny_teacher, tokenizer=fake_tokenizer, seed=42,
        )
        pairs = gen.generate("The quick brown fox jumps over the lazy dog.")
        assert pairs == []


# ==================================================================
# 4. ContrastivePairValidator
# ==================================================================


class TestContrastiveValidator:
    def test_find_modified_positions(self) -> None:
        clean = torch.tensor([1, 2, 3, 4, 5])
        corrupt = torch.tensor([1, 2, 9, 4, 5])
        positions = ContrastivePairValidator.find_modified_positions(clean, corrupt)
        assert positions == [2]

    def test_identical_pair_fails_output_check(
        self, tiny_teacher, tiny_adapter, fake_tokenizer,
    ) -> None:
        validator = ContrastivePairValidator(
            teacher=tiny_teacher, adapter=tiny_adapter,
            tokenizer=fake_tokenizer, device="cpu",
        )
        result = validator.validate("same text here", "same text here")
        assert result["teacher_output_changed"] is False


# ==================================================================
# 5. ScalableDistillationTrainer — standard_kd
# ==================================================================


class TestScalableTrainerStandardKD:
    @pytest.fixture
    def setup(self, tiny_teacher, tiny_student, tiny_adapter, fake_tokenizer) -> dict:
        main_ds = _FakeInstructionDataset(n=8, seq_len=10, vocab_size=100)
        config = {
            "method": "standard_kd",
            "epochs": 2,
            "batch_size": 4,
            "gradient_accumulation": 1,
            "lr": 1e-3,
            "fp16": False,
            "temperature": 2.0,
            "use_wandb": False,
        }
        return {
            "teacher": tiny_teacher, "student": tiny_student,
            "adapter": tiny_adapter, "tokenizer": fake_tokenizer,
            "main_ds": main_ds, "config": config,
        }

    def test_trains_without_error(self, setup: dict) -> None:
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        trainer = ScalableDistillationTrainer(
            teacher=setup["teacher"], student=setup["student"],
            teacher_adapter=setup["adapter"], student_adapter=setup["adapter"],
            tokenizer=setup["tokenizer"],
            main_dataset=setup["main_ds"],
            config=setup["config"],
        )
        history = trainer.train()
        assert "loss" in history
        assert len(history["loss"]) == 2

    def test_student_params_updated(self, setup: dict) -> None:
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        student = setup["student"]
        params_before = {n: p.clone() for n, p in student.named_parameters()}

        trainer = ScalableDistillationTrainer(
            teacher=setup["teacher"], student=student,
            teacher_adapter=setup["adapter"], student_adapter=setup["adapter"],
            tokenizer=setup["tokenizer"],
            main_dataset=setup["main_ds"],
            config=setup["config"],
        )
        trainer.train()

        any_changed = any(
            not torch.equal(p, params_before[n])
            for n, p in student.named_parameters()
        )
        assert any_changed


# ==================================================================
# 6. ScalableDistillationTrainer — adaptive methods
# ==================================================================


class TestScalableTrainerAdaptive:
    """Test trainer with adaptive KL methods (AKL, KLR)."""

    @pytest.fixture
    def base_setup(self, tiny_teacher, tiny_student, tiny_adapter, fake_tokenizer):
        main_ds = _FakeInstructionDataset(n=8, seq_len=10, vocab_size=100)
        return {
            "teacher": tiny_teacher, "student": tiny_student,
            "adapter": tiny_adapter, "tokenizer": fake_tokenizer,
            "main_ds": main_ds,
        }

    def _make_trainer(self, setup: dict, method_cfg: dict):
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        config = {
            "epochs": 1,
            "batch_size": 4,
            "gradient_accumulation": 1,
            "lr": 1e-3,
            "fp16": False,
            "temperature": 2.0,
            "use_wandb": False,
            **method_cfg,
        }
        # Need fresh student each time
        torch.manual_seed(1)
        student = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        return ScalableDistillationTrainer(
            teacher=setup["teacher"], student=student,
            teacher_adapter=setup["adapter"], student_adapter=setup["adapter"],
            tokenizer=setup["tokenizer"],
            main_dataset=setup["main_ds"],
            config=config,
        )

    def test_klr_token(self, base_setup) -> None:
        trainer = self._make_trainer(base_setup, {
            "method": "standard_kd_klr", "klr_granularity": "token",
        })
        history = trainer.train()
        assert "alpha_mean" in history
        assert len(history["loss"]) == 1

    def test_klr_batch_ema(self, base_setup) -> None:
        trainer = self._make_trainer(base_setup, {
            "method": "standard_kd_klr", "klr_granularity": "batch",
            "klr_beta": 0.99,
        })
        history = trainer.train()
        assert "alpha_mean" in history

    def test_akl(self, base_setup) -> None:
        trainer = self._make_trainer(base_setup, {
            "method": "standard_kd_akl", "akl_mu": 0.5,
        })
        history = trainer.train()
        assert "alpha_mean" in history

    def test_reverse_kl(self, base_setup) -> None:
        trainer = self._make_trainer(base_setup, {"method": "reverse_kl"})
        history = trainer.train()
        assert "alpha_mean" in history
        # alpha should be 0.0 for reverse KL
        assert all(a == 0.0 for a in history["alpha_mean"])

    def test_jeffreys_via_fixed_alpha(self, base_setup) -> None:
        trainer = self._make_trainer(base_setup, {
            "method": "standard_kd_klr",
            "klr_granularity": "batch",
            "klr_fixed_alpha": 0.5,
        })
        history = trainer.train()
        assert "alpha_mean" in history
        # alpha should be 0.5 for Jeffreys
        assert all(abs(a - 0.5) < 1e-6 for a in history["alpha_mean"])

    def test_jsonl_output(self, base_setup) -> None:
        """Training should produce training_stats.jsonl."""
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "method": "standard_kd_klr",
                "klr_granularity": "batch",
                "klr_beta": 0.99,
                "epochs": 1,
                "batch_size": 4,
                "gradient_accumulation": 1,
                "lr": 1e-3,
                "fp16": False,
                "temperature": 2.0,
                "use_wandb": False,
                "jsonl_every": 1,  # log every step for test
            }
            torch.manual_seed(1)
            student = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
            trainer = ScalableDistillationTrainer(
                teacher=base_setup["teacher"], student=student,
                teacher_adapter=base_setup["adapter"],
                student_adapter=base_setup["adapter"],
                tokenizer=base_setup["tokenizer"],
                main_dataset=base_setup["main_ds"],
                config=config,
            )
            trainer.train(save_dir=tmpdir)

            jsonl_path = Path(tmpdir) / "training_stats.jsonl"
            assert jsonl_path.exists()
            lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) > 0
            record = json.loads(lines[0])
            assert "step" in record
            assert "loss" in record
