"""Tests for large-scale distillation components.

Uses TinyTransformerModel (from conftest.py) so no external models are needed.
Each test is self-contained and creates its own synthetic data.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from conftest import TinyAdapter, TinyTransformerModel
from rcid.circuit.contrastive import ContrastiveDataset
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
        # Deterministic: map each character to an int in [2, vocab_size)
        ids = [min(ord(c) % (self.vocab_size - 2) + 2, self.vocab_size - 1)
               for c in text]
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        t = torch.tensor([ids], dtype=torch.long)   # (1, seq_len)
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
        """Verify InstructionDataset tokenises and builds correct items."""
        from rcid.data.instruction_dataset import (
            _detect_and_format,
            InstructionDataset,
        )

        # Test format detection on Alpaca-style dicts
        sample = {
            "instruction": "Translate to French.",
            "input": "Hello",
            "output": "Bonjour",
        }
        prompt, full = _detect_and_format(sample)
        assert "Translate to French" in prompt
        assert "Bonjour" in full
        assert prompt in full  # prompt is a prefix of full text

        sample_no_input = {
            "instruction": "Say hi.",
            "output": "Hi there!",
        }
        prompt2, full2 = _detect_and_format(sample_no_input)
        assert "Say hi" in prompt2
        assert "Hi there" in full2

    def test_collate_fn_shapes(self) -> None:
        """collate_fn should produce (batch, max_len) tensors."""
        ds = _FakeInstructionDataset(n=6, seq_len=10)
        batch = [ds[i] for i in range(4)]
        collated = ds.collate_fn(batch)

        assert collated["input_ids"].shape == (4, 10)
        assert collated["attention_mask"].shape == (4, 10)
        assert collated["labels_mask"].shape == (4, 10)

    def test_collate_fn_variable_lengths(self) -> None:
        """collate_fn pads to max length in the batch."""
        ds = _FakeInstructionDataset(n=4, seq_len=8)
        # Override one sample to be shorter
        ds.input_ids[0] = torch.randint(2, 100, (5,))
        ds.attention_masks[0] = torch.ones(5, dtype=torch.long)
        ds.labels_masks[0] = torch.ones(5, dtype=torch.long)

        batch = [ds[i] for i in range(3)]
        collated = ds.collate_fn(batch)
        # Max len is 8 (from samples 1 and 2)
        assert collated["input_ids"].shape[1] == 8
        # First sample should be padded (zeros at end)
        assert collated["input_ids"][0, 5:].sum() == 0


# ==================================================================
# 2. EntitySwapGenerator
# ==================================================================


class TestEntitySwapGenerator:
    """Test entity replacement in text."""

    def test_detects_and_swaps_country(self, tiny_teacher: TinyTransformerModel,
                                        fake_tokenizer: _FakeTokenizer) -> None:
        """Text with a known country should produce swap pairs."""
        gen = EntitySwapGenerator(
            teacher=tiny_teacher, tokenizer=fake_tokenizer, seed=42,
        )
        text = "The capital of France is Paris."
        # Bypass teacher validation — just test swap logic
        pairs = gen.generate(text)

        assert len(pairs) > 0, "Should produce at least one pair"
        for clean, corrupt in pairs:
            assert clean == text
            assert corrupt != text
            # The country or city was swapped
            assert ("France" not in corrupt) or ("Paris" not in corrupt)

    def test_only_entity_changes(self, tiny_teacher: TinyTransformerModel,
                                  fake_tokenizer: _FakeTokenizer) -> None:
        """Non-entity parts of the text should be unchanged."""
        gen = EntitySwapGenerator(
            teacher=tiny_teacher, tokenizer=fake_tokenizer, seed=42,
        )
        text = "Alice went to the store and bought milk."
        pairs = gen.generate(text)

        for clean, corrupt in pairs:
            # "went to the store and bought milk" should be in corrupt
            assert "went to the store and bought milk" in corrupt

    def test_no_entities_returns_empty(self, tiny_teacher: TinyTransformerModel,
                                        fake_tokenizer: _FakeTokenizer) -> None:
        """Text without known entities should return empty list."""
        gen = EntitySwapGenerator(
            teacher=tiny_teacher, tokenizer=fake_tokenizer, seed=42,
        )
        text = "The quick brown fox jumps over the lazy dog."
        pairs = gen.generate(text)
        assert pairs == []


# ==================================================================
# 3. NumberPerturbGenerator
# ==================================================================


class TestNumberPerturbGenerator:
    """Test number perturbation in text."""

    def test_detects_and_perturbs_number(self, tiny_teacher: TinyTransformerModel,
                                          fake_tokenizer: _FakeTokenizer) -> None:
        """Text with a number should produce perturbed pairs."""
        gen = NumberPerturbGenerator(
            teacher=tiny_teacher, tokenizer=fake_tokenizer, seed=42,
        )
        text = "John has 5 apples and gives 2 away."
        pairs = gen.generate(text)

        assert len(pairs) > 0, "Should produce at least one pair"
        for clean, corrupt in pairs:
            assert clean == text
            assert corrupt != text

    def test_only_number_changes(self, tiny_teacher: TinyTransformerModel,
                                  fake_tokenizer: _FakeTokenizer) -> None:
        """Non-numeric parts should be unchanged."""
        gen = NumberPerturbGenerator(
            teacher=tiny_teacher, tokenizer=fake_tokenizer, seed=42,
            max_perturbations_per_number=1,
        )
        text = "There are 10 items in the box."
        pairs = gen.generate(text)

        for clean, corrupt in pairs:
            # Check: text differs only in the number
            clean_words = clean.split()
            corrupt_words = corrupt.split()
            diffs = [
                (a, b) for a, b in zip(clean_words, corrupt_words) if a != b
            ]
            # Only one word (the number) should differ
            assert len(diffs) == 1
            # The changed word should be a number
            assert diffs[0][1].isdigit()

    def test_no_numbers_returns_empty(self, tiny_teacher: TinyTransformerModel,
                                       fake_tokenizer: _FakeTokenizer) -> None:
        """Text without numbers should return empty list."""
        gen = NumberPerturbGenerator(
            teacher=tiny_teacher, tokenizer=fake_tokenizer, seed=42,
        )
        text = "The quick brown fox jumps over the lazy dog."
        pairs = gen.generate(text)
        assert pairs == []


# ==================================================================
# 4. ContrastivePairValidator
# ==================================================================


class TestContrastiveValidator:
    """Test quality validation of contrastive pairs."""

    def test_find_modified_positions(self) -> None:
        """find_modified_positions should return indices of differing tokens."""
        clean = torch.tensor([1, 2, 3, 4, 5])
        corrupt = torch.tensor([1, 2, 9, 4, 5])  # position 2 differs
        positions = ContrastivePairValidator.find_modified_positions(clean, corrupt)
        assert positions == [2]

    def test_find_modified_positions_multiple(self) -> None:
        """Multiple differing positions should all be found."""
        clean = torch.tensor([1, 2, 3, 4, 5])
        corrupt = torch.tensor([9, 2, 3, 8, 5])  # positions 0 and 3
        positions = ContrastivePairValidator.find_modified_positions(clean, corrupt)
        assert positions == [0, 3]

    def test_find_modified_positions_identical(self) -> None:
        """Identical sequences should return empty list."""
        ids = torch.tensor([1, 2, 3, 4, 5])
        positions = ContrastivePairValidator.find_modified_positions(ids, ids)
        assert positions == []

    def test_validate_returns_all_checks(
        self, tiny_teacher: TinyTransformerModel, tiny_adapter: TinyAdapter,
        fake_tokenizer: _FakeTokenizer,
    ) -> None:
        """validate() should return dict with all 5 check keys."""
        validator = ContrastivePairValidator(
            teacher=tiny_teacher,
            adapter=tiny_adapter,
            tokenizer=fake_tokenizer,
            causal_effect_threshold=0.0,  # low threshold for tiny model
            device="cpu",
        )
        result = validator.validate(
            "The capital of France is", "The capital of Germany is",
        )
        expected_keys = {
            "teacher_output_changed",
            "edit_distance_ok",
            "length_preserved",
            "tokens_alignable",
            "causal_effect_exists",
        }
        assert set(result.keys()) == expected_keys
        # Each value should be a bool
        for v in result.values():
            assert isinstance(v, bool)

    def test_identical_pair_fails_output_check(
        self, tiny_teacher: TinyTransformerModel, tiny_adapter: TinyAdapter,
        fake_tokenizer: _FakeTokenizer,
    ) -> None:
        """Identical clean/corrupt should fail teacher_output_changed."""
        validator = ContrastivePairValidator(
            teacher=tiny_teacher, adapter=tiny_adapter,
            tokenizer=fake_tokenizer, device="cpu",
        )
        result = validator.validate("same text here", "same text here")
        assert result["teacher_output_changed"] is False


# ==================================================================
# 5. ScalableDistillationTrainer — standard_kd only
# ==================================================================


class TestScalableTrainerStandardKD:
    """Test ScalableDistillationTrainer without RCID."""

    @pytest.fixture
    def setup(
        self, tiny_teacher: TinyTransformerModel,
        tiny_student: TinyTransformerModel, tiny_adapter: TinyAdapter,
        fake_tokenizer: _FakeTokenizer,
    ) -> dict:
        main_ds = _FakeInstructionDataset(n=8, seq_len=10, vocab_size=100)
        config = {
            "epochs": 2,
            "batch_size": 4,
            "gradient_accumulation": 1,
            "lr": 1e-3,
            "fp16": False,
            "temperature": 2.0,
            "lambda_rcid": 0.0,
            "use_wandb": False,
        }
        return {
            "teacher": tiny_teacher,
            "student": tiny_student,
            "adapter": tiny_adapter,
            "tokenizer": fake_tokenizer,
            "main_ds": main_ds,
            "config": config,
        }

    def test_trains_without_error(self, setup: dict) -> None:
        """Standard KD training should complete without errors."""
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        trainer = ScalableDistillationTrainer(
            teacher=setup["teacher"], student=setup["student"],
            teacher_adapter=setup["adapter"], student_adapter=setup["adapter"],
            tokenizer=setup["tokenizer"],
            main_dataset=setup["main_ds"],
            contrastive_dataset=None,
            config=setup["config"],
        )
        history = trainer.train()

        assert "loss" in history
        assert "kl_loss" in history
        assert len(history["loss"]) == 2  # 2 epochs

    def test_loss_is_positive(self, setup: dict) -> None:
        """KL loss should be positive."""
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        trainer = ScalableDistillationTrainer(
            teacher=setup["teacher"], student=setup["student"],
            teacher_adapter=setup["adapter"], student_adapter=setup["adapter"],
            tokenizer=setup["tokenizer"],
            main_dataset=setup["main_ds"],
            contrastive_dataset=None,
            config=setup["config"],
        )
        history = trainer.train()
        assert all(l > 0 for l in history["kl_loss"])

    def test_student_params_updated(self, setup: dict) -> None:
        """Student parameters should change after training."""
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        student = setup["student"]
        params_before = {n: p.clone() for n, p in student.named_parameters()}

        trainer = ScalableDistillationTrainer(
            teacher=setup["teacher"], student=student,
            teacher_adapter=setup["adapter"], student_adapter=setup["adapter"],
            tokenizer=setup["tokenizer"],
            main_dataset=setup["main_ds"],
            contrastive_dataset=None,
            config=setup["config"],
        )
        trainer.train()

        any_changed = any(
            not torch.equal(p, params_before[n])
            for n, p in student.named_parameters()
        )
        assert any_changed, "Student parameters should change after training"

    def test_teacher_params_frozen(self, setup: dict) -> None:
        """Teacher parameters should NOT change during training."""
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        teacher = setup["teacher"]
        params_before = {n: p.clone() for n, p in teacher.named_parameters()}

        trainer = ScalableDistillationTrainer(
            teacher=teacher, student=setup["student"],
            teacher_adapter=setup["adapter"], student_adapter=setup["adapter"],
            tokenizer=setup["tokenizer"],
            main_dataset=setup["main_ds"],
            contrastive_dataset=None,
            config=setup["config"],
        )
        trainer.train()

        for n, p in teacher.named_parameters():
            assert torch.equal(p, params_before[n]), f"Teacher param {n} changed!"


# ==================================================================
# 6. ScalableDistillationTrainer — with RCID
# ==================================================================


class TestScalableTrainerWithRCID:
    """Test ScalableDistillationTrainer with RCID regulariser."""

    @pytest.fixture
    def setup_rcid(
        self, tiny_teacher: TinyTransformerModel,
        tiny_student: TinyTransformerModel, tiny_adapter: TinyAdapter,
        fake_tokenizer: _FakeTokenizer,
    ) -> dict:
        torch.manual_seed(42)
        n, seq_len, vocab = 8, 10, 100

        main_ds = _FakeInstructionDataset(n=n, seq_len=seq_len, vocab_size=vocab)

        # Build a small contrastive dataset
        contrastive_ds = ContrastiveDataset(
            clean_ids=torch.randint(2, vocab, (n, seq_len)),
            corrupt_ids=torch.randint(2, vocab, (n, seq_len)),
            answer_pos=torch.full((n,), seq_len - 1, dtype=torch.long),
            correct_token_id=torch.randint(0, vocab, (n,)),
            wrong_token_id=torch.randint(0, vocab, (n,)),
            key_positions={"modified": torch.full((n,), 3, dtype=torch.long)},
            is_modified={"modified": True},
            model_family="test",
        )

        d_model = 32
        checkpoints = [(2, 3)]
        layer_mapping = {2: 2}
        W_matrices = {2: torch.eye(d_model, d_model)}

        config = {
            "epochs": 2,
            "batch_size": 4,
            "gradient_accumulation": 1,
            "lr": 1e-3,
            "fp16": False,
            "temperature": 2.0,
            "lambda_rcid": 0.5,
            "rcid_every_n_steps": 1,  # every step for test coverage
            "use_wandb": False,
        }
        return {
            "teacher": tiny_teacher,
            "student": tiny_student,
            "adapter": tiny_adapter,
            "tokenizer": fake_tokenizer,
            "main_ds": main_ds,
            "contrastive_ds": contrastive_ds,
            "checkpoints": checkpoints,
            "layer_mapping": layer_mapping,
            "W_matrices": W_matrices,
            "config": config,
        }

    def test_rcid_training_completes(self, setup_rcid: dict) -> None:
        """Training with RCID should complete without errors."""
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        trainer = ScalableDistillationTrainer(
            teacher=setup_rcid["teacher"],
            student=setup_rcid["student"],
            teacher_adapter=setup_rcid["adapter"],
            student_adapter=setup_rcid["adapter"],
            tokenizer=setup_rcid["tokenizer"],
            main_dataset=setup_rcid["main_ds"],
            contrastive_dataset=setup_rcid["contrastive_ds"],
            config=setup_rcid["config"],
            checkpoints=setup_rcid["checkpoints"],
            layer_mapping=setup_rcid["layer_mapping"],
            W_matrices=setup_rcid["W_matrices"],
        )
        history = trainer.train()

        assert len(history["loss"]) == 2
        assert len(history["rcid_loss"]) == 2

    def test_rcid_loss_nonzero(self, setup_rcid: dict) -> None:
        """RCID loss should be non-zero (teacher != random student)."""
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        trainer = ScalableDistillationTrainer(
            teacher=setup_rcid["teacher"],
            student=setup_rcid["student"],
            teacher_adapter=setup_rcid["adapter"],
            student_adapter=setup_rcid["adapter"],
            tokenizer=setup_rcid["tokenizer"],
            main_dataset=setup_rcid["main_ds"],
            contrastive_dataset=setup_rcid["contrastive_ds"],
            config=setup_rcid["config"],
            checkpoints=setup_rcid["checkpoints"],
            layer_mapping=setup_rcid["layer_mapping"],
            W_matrices=setup_rcid["W_matrices"],
        )
        history = trainer.train()

        # At least one epoch should have nonzero RCID loss
        assert any(r > 0 for r in history["rcid_loss"]), (
            f"Expected nonzero RCID loss, got {history['rcid_loss']}"
        )

    def test_use_rcid_flag_set(self, setup_rcid: dict) -> None:
        """Trainer should detect RCID components and set use_rcid=True."""
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        trainer = ScalableDistillationTrainer(
            teacher=setup_rcid["teacher"],
            student=setup_rcid["student"],
            teacher_adapter=setup_rcid["adapter"],
            student_adapter=setup_rcid["adapter"],
            tokenizer=setup_rcid["tokenizer"],
            main_dataset=setup_rcid["main_ds"],
            contrastive_dataset=setup_rcid["contrastive_ds"],
            config=setup_rcid["config"],
            checkpoints=setup_rcid["checkpoints"],
            layer_mapping=setup_rcid["layer_mapping"],
            W_matrices=setup_rcid["W_matrices"],
        )
        assert trainer.use_rcid is True

    def test_without_contrastive_disables_rcid(self, setup_rcid: dict) -> None:
        """Without contrastive data, use_rcid should be False."""
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        trainer = ScalableDistillationTrainer(
            teacher=setup_rcid["teacher"],
            student=setup_rcid["student"],
            teacher_adapter=setup_rcid["adapter"],
            student_adapter=setup_rcid["adapter"],
            tokenizer=setup_rcid["tokenizer"],
            main_dataset=setup_rcid["main_ds"],
            contrastive_dataset=None,
            config=setup_rcid["config"],
        )
        assert trainer.use_rcid is False
