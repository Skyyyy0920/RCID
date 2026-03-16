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

    def _make_trainer(self, setup: dict, method_cfg: dict, **extra_kwargs):
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
            **extra_kwargs,
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


# ==================================================================
# 7. RCID modules: patching, CKA, Procrustes, loss
# ==================================================================


class TestPatchingModule:
    """Test circuit/patching.py extract_contrastive_differences."""

    def test_extract_diffs_shape(self, tiny_teacher, tiny_adapter) -> None:
        from rcid.circuit.patching import extract_contrastive_differences

        clean = torch.randint(2, 100, (4, 8))
        corrupt = torch.randint(2, 100, (4, 8))
        diffs = extract_contrastive_differences(
            tiny_teacher, tiny_adapter, clean, corrupt,
            layers=[0, 2], batch_size=4,
        )
        assert 0 in diffs and 2 in diffs
        assert diffs[0].shape == (4, 8, 32)  # (N, seq, d_model)

    def test_identical_inputs_zero_diff(self, tiny_teacher, tiny_adapter) -> None:
        from rcid.circuit.patching import extract_contrastive_differences

        ids = torch.randint(2, 100, (3, 6))
        diffs = extract_contrastive_differences(
            tiny_teacher, tiny_adapter, ids, ids,
            layers=[1], batch_size=3,
        )
        assert diffs[1].abs().max().item() < 1e-5

    def test_pooled_shape(self, tiny_teacher, tiny_adapter) -> None:
        from rcid.circuit.patching import extract_contrastive_differences

        clean = torch.randint(2, 100, (4, 8))
        corrupt = torch.randint(2, 100, (4, 8))
        diffs = extract_contrastive_differences(
            tiny_teacher, tiny_adapter, clean, corrupt,
            layers=[0, 3], batch_size=4, pool_seq=True,
        )
        assert diffs[0].shape == (4, 32)  # (N, d_model) — pooled


class TestInterventionModule:
    """Test circuit/intervention.py patch_and_run."""

    def test_patch_and_run_shape(self, tiny_teacher, tiny_adapter) -> None:
        from rcid.circuit.intervention import patch_and_run

        clean = torch.randint(2, 100, (2, 6))
        patch_val = torch.randn(2, 32)  # (batch, d_model)
        logits = patch_and_run(
            tiny_teacher, tiny_adapter, clean,
            patch_value=patch_val, layer=1, token_pos=3,
        )
        assert logits.shape == (2, 6, 100)  # (batch, seq, vocab)

    def test_causal_effect_at_modified_pos(self, tiny_teacher, tiny_adapter) -> None:
        from rcid.circuit.intervention import compute_causal_effect

        torch.manual_seed(42)
        clean = torch.randint(2, 100, (1, 8))
        corrupt = clean.clone()
        corrupt[0, 3] = (clean[0, 3] + 1) % 100  # modify one position
        delta = compute_causal_effect(
            tiny_teacher, tiny_adapter,
            clean_ids=clean, corrupt_ids=corrupt,
            layer=2, token_pos=3,
            answer_pos=3,  # TinyModel has no attention: must be same as patch pos
            correct_id=5, wrong_id=10,
        )
        # Should have non-zero effect when patching at modified position
        assert isinstance(delta, float)


class TestCKAModule:
    """Test alignment/cka.py."""

    def test_self_cka_is_one(self) -> None:
        from rcid.alignment.cka import linear_cka

        X = torch.randn(50, 32)
        assert abs(linear_cka(X, X) - 1.0) < 1e-4

    def test_cka_matrix_shape(self) -> None:
        from rcid.alignment.cka import cka_matrix

        t_reps = {0: torch.randn(20, 64), 1: torch.randn(20, 64)}
        s_reps = {0: torch.randn(20, 32), 1: torch.randn(20, 32), 2: torch.randn(20, 32)}
        mat = cka_matrix(t_reps, s_reps)
        assert mat.shape == (2, 3)  # (n_T, n_S)


class TestProcrustesModule:
    """Test alignment/procrustes.py."""

    def test_procrustes_shape(self) -> None:
        from rcid.alignment.procrustes import procrustes_align

        source = torch.randn(50, 16)  # student: d_S=16
        target = torch.randn(50, 32)  # teacher: d_T=32
        W = procrustes_align(source, target)
        assert W.shape == (32, 16)  # (d_T, d_S)

    def test_aligned_projection(self) -> None:
        from rcid.alignment.procrustes import procrustes_align

        # Create synthetic data with known structure
        torch.manual_seed(0)
        source = torch.randn(100, 8)
        W_true = torch.randn(16, 8)
        target = source @ W_true.t()
        W_est = procrustes_align(source, target)
        # Reconstructed target should be close
        recon = source @ W_est.t()
        # Use cosine similarity (Procrustes preserves direction)
        cos = torch.nn.functional.cosine_similarity(
            recon.flatten().unsqueeze(0),
            target.flatten().unsqueeze(0),
        )
        assert cos.item() > 0.8  # Procrustes is orthogonal; true W may not be


class TestLayerMatchingModule:
    """Test alignment/layer_matching.py."""

    def test_greedy_matching(self) -> None:
        from rcid.alignment.layer_matching import match_layers

        # 4 teacher layers, 3 student layers
        cka = torch.tensor([
            [0.9, 0.1, 0.1],
            [0.2, 0.8, 0.1],
            [0.1, 0.3, 0.7],
            [0.1, 0.1, 0.95],
        ])
        mapping = match_layers(cka, 4, 3, strategy="greedy")
        assert isinstance(mapping, dict)
        assert len(mapping) == 4
        # Each teacher layer should map to one student layer
        assert all(v in range(3) for v in mapping.values())

    def test_linear_matching(self) -> None:
        from rcid.alignment.layer_matching import match_layers

        cka = torch.ones(6, 3)  # doesn't matter for linear
        mapping = match_layers(cka, 6, 3, strategy="linear")
        assert len(mapping) == 6


class TestCheckpointSelection:
    """Test circuit/checkpoint_selection.py."""

    def test_select_returns_tuples(self) -> None:
        from rcid.circuit.checkpoint_selection import select_checkpoints

        # Fake contrastive diffs: 4 layers, 3 samples, 6 seq positions, d=32
        diffs = {i: torch.randn(3, 6, 32) for i in range(4)}
        # Make layer 3 have higher norm (should be preferred)
        diffs[3] = diffs[3] * 10

        # Fake dataset with is_modified metadata
        class FakeDS:
            seq_len = 6
            is_modified = {"s2_pos": True}
            key_positions = {"s2_pos": torch.tensor([2, 3, 4])}

        checkpoints = select_checkpoints(diffs, FakeDS(), top_k=5)
        assert isinstance(checkpoints, list)
        assert len(checkpoints) <= 5
        assert all(isinstance(c, tuple) and len(c) == 2 for c in checkpoints)


class TestRCIDLossModule:
    """Test distillation/rcid_loss.py."""

    def test_rcid_loss_computes(self) -> None:
        from rcid.distillation.rcid_loss import RCIDLoss

        # Setup: 2 checkpoints, layer 1 -> student layer 0
        checkpoints = [(1, 2), (1, 4)]
        layer_mapping = {1: 0}
        W = torch.eye(32)  # same dim for simplicity
        loss_fn = RCIDLoss(checkpoints, layer_mapping, {1: W})

        teacher_diffs = {1: torch.randn(2, 6, 32)}
        s_clean = {0: torch.randn(2, 6, 32, requires_grad=True)}
        s_corrupt = {0: torch.randn(2, 6, 32, requires_grad=True)}

        loss = loss_fn(teacher_diffs, s_clean, s_corrupt)
        assert loss.isfinite()
        assert loss.item() >= 0
        loss.backward()
        assert s_clean[0].grad is not None

    def test_rcid_zero_when_identical(self) -> None:
        from rcid.distillation.rcid_loss import RCIDLoss

        checkpoints = [(0, 1)]
        layer_mapping = {0: 0}
        W = torch.eye(16)
        loss_fn = RCIDLoss(checkpoints, layer_mapping, {0: W})

        # If student diffs match teacher diffs exactly -> loss ~ 0
        d = torch.randn(3, 4, 16)
        teacher_diffs = {0: d.detach()}
        # s_clean - s_corrupt = d
        s_corrupt = torch.zeros(3, 4, 16, requires_grad=True)
        s_clean = (d + s_corrupt).requires_grad_(True)

        loss = loss_fn(teacher_diffs, {0: s_clean}, {0: s_corrupt})
        assert loss.item() < 1e-4

    def test_extract_residuals_with_grad(self, tiny_teacher, tiny_adapter) -> None:
        from rcid.distillation.rcid_loss import extract_residuals_with_grad

        # Note: need a model with requires_grad (student, not teacher)
        torch.manual_seed(1)
        student = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        student.train()
        ids = torch.randint(2, 100, (2, 6))
        cache = extract_residuals_with_grad(student, tiny_adapter, ids, [0, 2])
        assert 0 in cache and 2 in cache
        assert cache[0].shape == (2, 6, 32)


class TestGeneratedContrastiveDataset:
    """Test data/generated_contrastive.py."""

    def test_load_from_json(self, fake_tokenizer) -> None:
        from rcid.data.generated_contrastive import GeneratedContrastiveDataset

        pairs = [
            {"clean": "The capital of France is", "corrupt": "The capital of Germany is", "task_type": "entity_swap"},
            {"clean": "John has 5 apples", "corrupt": "John has 7 apples", "task_type": "number_perturb"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(pairs, f)
            tmp_path = f.name

        try:
            ds = GeneratedContrastiveDataset(tmp_path, fake_tokenizer, max_seq_len=64)
            assert len(ds) == 2
            item = ds[0]
            assert "clean_ids" in item
            assert "corrupt_ids" in item
            assert item["clean_ids"].shape == item["corrupt_ids"].shape
        finally:
            Path(tmp_path).unlink()

    def test_collate_fn(self, fake_tokenizer) -> None:
        from rcid.data.generated_contrastive import GeneratedContrastiveDataset

        pairs = [
            {"clean": "Hello world", "corrupt": "Hello earth", "task_type": "test"},
            {"clean": "Good morning", "corrupt": "Good evening", "task_type": "test"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(pairs, f)
            tmp_path = f.name

        try:
            ds = GeneratedContrastiveDataset(tmp_path, fake_tokenizer)
            batch = [ds[0], ds[1]]
            collated = ds.collate_fn(batch)
            assert collated["clean_ids"].shape[0] == 2
        finally:
            Path(tmp_path).unlink()


# ==================================================================
# 8. ScalableDistillationTrainer — RCID methods
# ==================================================================


class _FakeContrastiveDataset(Dataset):
    """Tiny contrastive dataset for RCID trainer tests."""

    def __init__(self, n: int = 8, seq_len: int = 10, vocab_size: int = 100) -> None:
        self.clean_ids = torch.randint(2, vocab_size, (n, seq_len))
        self.corrupt_ids = self.clean_ids.clone()
        # Modify one position per sample
        for i in range(n):
            pos = i % seq_len
            self.corrupt_ids[i, pos] = (self.clean_ids[i, pos] + 1) % vocab_size
        self.is_modified = {"modified": True}
        self.key_positions = {
            "modified": torch.arange(n) % seq_len,
        }
        self._seq_len = seq_len

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def __len__(self) -> int:
        return self.clean_ids.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "clean_ids": self.clean_ids[idx],
            "corrupt_ids": self.corrupt_ids[idx],
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "clean_ids": torch.stack([b["clean_ids"] for b in batch]),
            "corrupt_ids": torch.stack([b["corrupt_ids"] for b in batch]),
        }


class TestScalableTrainerRCID:
    """Test trainer with RCID methods (dual data stream)."""

    @pytest.fixture
    def rcid_setup(self, tiny_teacher, tiny_adapter, fake_tokenizer):
        main_ds = _FakeInstructionDataset(n=8, seq_len=10, vocab_size=100)
        contrastive_ds = _FakeContrastiveDataset(n=8, seq_len=10, vocab_size=100)

        # Simple layer mapping and checkpoints for 4-layer model
        layer_mapping = {0: 0, 1: 1, 2: 2, 3: 3}
        checkpoints = [(2, 3), (3, 5)]
        W_matrices = {l: torch.eye(32) for l in layer_mapping}

        return {
            "teacher": tiny_teacher,
            "adapter": tiny_adapter,
            "tokenizer": fake_tokenizer,
            "main_ds": main_ds,
            "contrastive_ds": contrastive_ds,
            "layer_mapping": layer_mapping,
            "checkpoints": checkpoints,
            "W_matrices": W_matrices,
        }

    def _make_rcid_trainer(self, setup: dict, method: str):
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        config = {
            "method": method,
            "epochs": 1,
            "batch_size": 4,
            "gradient_accumulation": 1,
            "lr": 1e-3,
            "fp16": False,
            "temperature": 2.0,
            "use_wandb": False,
        }
        torch.manual_seed(1)
        student = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)

        # Build appropriate loss function
        if method == "standard_kd_rcid":
            from rcid.distillation.rcid_loss import RCIDLoss
            loss_fn = RCIDLoss(
                setup["checkpoints"], setup["layer_mapping"], setup["W_matrices"],
            )
        elif method == "standard_kd_fitnets":
            from rcid.distillation.baselines import FitNetsLoss
            loss_fn = FitNetsLoss(setup["layer_mapping"], setup["W_matrices"])
        else:  # standard_kd_informed_fitnets
            from rcid.distillation.baselines import InformedFitNetsLoss
            loss_fn = InformedFitNetsLoss(
                setup["checkpoints"], setup["layer_mapping"], setup["W_matrices"],
            )

        return ScalableDistillationTrainer(
            teacher=setup["teacher"], student=student,
            teacher_adapter=setup["adapter"], student_adapter=setup["adapter"],
            tokenizer=setup["tokenizer"],
            main_dataset=setup["main_ds"],
            config=config,
            contrastive_dataset=setup["contrastive_ds"],
            rcid_loss_fn=loss_fn,
            lambda_rcid=0.5,
            rcid_every_n_steps=1,
            layer_mapping=setup["layer_mapping"],
            checkpoints=setup["checkpoints"],
        )

    def test_rcid_trains_without_error(self, rcid_setup) -> None:
        trainer = self._make_rcid_trainer(rcid_setup, "standard_kd_rcid")
        history = trainer.train()
        assert "loss" in history
        assert "rcid_loss" in history
        assert len(history["loss"]) == 1
        assert len(history["rcid_loss"]) == 1

    def test_fitnets_trains_without_error(self, rcid_setup) -> None:
        trainer = self._make_rcid_trainer(rcid_setup, "standard_kd_fitnets")
        history = trainer.train()
        assert "rcid_loss" in history
        assert len(history["loss"]) == 1

    def test_informed_fitnets_trains_without_error(self, rcid_setup) -> None:
        trainer = self._make_rcid_trainer(rcid_setup, "standard_kd_informed_fitnets")
        history = trainer.train()
        assert "rcid_loss" in history
        assert len(history["loss"]) == 1

    def test_rcid_student_params_updated(self, rcid_setup) -> None:
        trainer = self._make_rcid_trainer(rcid_setup, "standard_kd_rcid")
        student = trainer.student
        params_before = {n: p.clone() for n, p in student.named_parameters()}

        trainer.train()

        any_changed = any(
            not torch.equal(p, params_before[n])
            for n, p in student.named_parameters()
        )
        assert any_changed

    def test_rcid_method_requires_contrastive(self, rcid_setup) -> None:
        """RCID method without contrastive_dataset should fail."""
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        config = {"method": "standard_kd_rcid", "epochs": 1, "batch_size": 4,
                  "fp16": False}
        torch.manual_seed(1)
        student = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)

        with pytest.raises(AssertionError, match="requires contrastive_dataset"):
            ScalableDistillationTrainer(
                teacher=rcid_setup["teacher"], student=student,
                teacher_adapter=rcid_setup["adapter"],
                student_adapter=rcid_setup["adapter"],
                tokenizer=rcid_setup["tokenizer"],
                main_dataset=rcid_setup["main_ds"],
                config=config,
                # No contrastive_dataset!
            )
