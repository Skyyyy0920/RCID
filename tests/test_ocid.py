"""Tests for OCID loss, LogitLens monitor, adaptive lambda, and trainer integration."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from conftest import TinyAdapter, TinyTransformerModel
from rcid.distillation.ocid_loss import OCIDLoss


# ==================================================================
# 1. OCIDLoss unit tests
# ==================================================================


class TestOCIDLoss:
    def test_finite_positive_loss(self) -> None:
        loss_fn = OCIDLoss(temperature=2.0)
        B, L, V = 2, 8, 50
        t_clean = torch.randn(B, L, V)
        t_corrupt = torch.randn(B, L, V)
        s_clean = torch.randn(B, L, V)
        s_corrupt = torch.randn(B, L, V)
        loss, stats = loss_fn(t_clean, t_corrupt, s_clean, s_corrupt)
        assert loss.isfinite()
        assert loss.item() >= 0.0

    def test_identical_sensitivity_zero_loss(self) -> None:
        """When student sensitivity matches teacher, loss should be ~0."""
        loss_fn = OCIDLoss(temperature=2.0)
        B, L, V = 2, 8, 50
        t_clean = torch.randn(B, L, V)
        t_corrupt = torch.randn(B, L, V)
        # Student has same logits as teacher -> same sensitivity -> loss ~ 0
        loss, stats = loss_fn(t_clean, t_corrupt, t_clean, t_corrupt)
        assert loss.item() < 0.01
        assert stats["mean_cos_sim"] > 0.99

    def test_no_teacher_change_zero_loss(self) -> None:
        """When teacher output doesn't change, all positions are inactive."""
        loss_fn = OCIDLoss(temperature=2.0)
        B, L, V = 2, 8, 50
        t_logits = torch.randn(B, L, V)
        # Same teacher logits for clean and corrupt -> delta_T = 0
        s_clean = torch.randn(B, L, V)
        s_corrupt = torch.randn(B, L, V)
        loss, stats = loss_fn(t_logits, t_logits, s_clean, s_corrupt)
        assert loss.item() < 1e-6
        assert stats["n_active_positions"] == 0

    def test_mask_respected(self) -> None:
        loss_fn = OCIDLoss(temperature=2.0)
        B, L, V = 2, 8, 50
        t_clean = torch.randn(B, L, V)
        t_corrupt = torch.randn(B, L, V)
        s_clean = torch.randn(B, L, V)
        s_corrupt = torch.randn(B, L, V)
        # Mask out half the positions
        mask = torch.zeros(B, L)
        mask[:, :4] = 1
        loss_masked, stats_masked = loss_fn(
            t_clean, t_corrupt, s_clean, s_corrupt, mask=mask,
        )
        loss_full, stats_full = loss_fn(
            t_clean, t_corrupt, s_clean, s_corrupt,
        )
        assert stats_masked["n_active_positions"] <= stats_full["n_active_positions"]

    def test_gradient_flows_to_student(self) -> None:
        """Gradient should flow through student logits only."""
        loss_fn = OCIDLoss(temperature=2.0)
        B, L, V = 2, 8, 50
        t_clean = torch.randn(B, L, V)
        t_corrupt = torch.randn(B, L, V)
        s_clean = torch.randn(B, L, V, requires_grad=True)
        s_corrupt = torch.randn(B, L, V, requires_grad=True)
        loss, _ = loss_fn(t_clean, t_corrupt, s_clean, s_corrupt)
        loss.backward()
        assert s_clean.grad is not None
        assert s_corrupt.grad is not None

    def test_stats_keys(self) -> None:
        """Check all expected diagnostic keys are present."""
        loss_fn = OCIDLoss(temperature=2.0)
        B, L, V = 2, 6, 30
        t_clean = torch.randn(B, L, V)
        t_corrupt = torch.randn(B, L, V)
        s_clean = torch.randn(B, L, V)
        s_corrupt = torch.randn(B, L, V)
        _, stats = loss_fn(t_clean, t_corrupt, s_clean, s_corrupt)
        for key in ("ocid_loss", "mean_teacher_norm", "mean_cos_sim", "n_active_positions"):
            assert key in stats, f"Missing key: {key}"


# ==================================================================
# 2. LogitLensMonitor unit tests
# ==================================================================


class TestLogitLensMonitor:
    def test_diagnose_returns_correct_keys(self) -> None:
        from rcid.distillation.logit_lens_monitor import LogitLensMonitor

        torch.manual_seed(0)
        teacher = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        student = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        adapter = TinyAdapter()

        monitor = LogitLensMonitor(teacher, student, adapter, adapter)
        clean_ids = torch.randint(2, 100, (2, 8))
        corrupt_ids = torch.randint(2, 100, (2, 8))

        report = monitor.diagnose(clean_ids, corrupt_ids)
        assert "layer_health" in report
        assert "overall_health" in report
        assert "worst_layers" in report

    def test_layer_health_per_student_layer(self) -> None:
        from rcid.distillation.logit_lens_monitor import LogitLensMonitor

        torch.manual_seed(0)
        teacher = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        student = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        adapter = TinyAdapter()

        monitor = LogitLensMonitor(teacher, student, adapter, adapter)
        clean_ids = torch.randint(2, 100, (2, 8))
        corrupt_ids = torch.randint(2, 100, (2, 8))

        report = monitor.diagnose(clean_ids, corrupt_ids)
        # Should have one entry per student layer
        assert len(report["layer_health"]) == 4

    def test_sample_layers_respected(self) -> None:
        from rcid.distillation.logit_lens_monitor import LogitLensMonitor

        torch.manual_seed(0)
        teacher = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        student = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        adapter = TinyAdapter()

        monitor = LogitLensMonitor(
            teacher, student, adapter, adapter,
            sample_layers=[0, 2],
        )
        clean_ids = torch.randint(2, 100, (2, 8))
        corrupt_ids = torch.randint(2, 100, (2, 8))

        report = monitor.diagnose(clean_ids, corrupt_ids)
        assert set(report["layer_health"].keys()) == {0, 2}

    def test_identical_models_high_health(self) -> None:
        """Same teacher and student should give high health."""
        from rcid.distillation.logit_lens_monitor import LogitLensMonitor

        torch.manual_seed(0)
        model = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        adapter = TinyAdapter()

        # Same model as both teacher and student
        monitor = LogitLensMonitor(model, model, adapter, adapter)
        clean_ids = torch.randint(2, 100, (2, 8))
        corrupt_ids = torch.randint(2, 100, (2, 8))

        report = monitor.diagnose(clean_ids, corrupt_ids)
        # Same model -> perfect alignment
        assert report["overall_health"] > 0.99

    def test_no_grad(self) -> None:
        """Monitor should not produce gradients."""
        from rcid.distillation.logit_lens_monitor import LogitLensMonitor

        torch.manual_seed(0)
        teacher = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        student = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        adapter = TinyAdapter()

        monitor = LogitLensMonitor(teacher, student, adapter, adapter)
        clean_ids = torch.randint(2, 100, (2, 8))
        corrupt_ids = torch.randint(2, 100, (2, 8))

        report = monitor.diagnose(clean_ids, corrupt_ids)
        # No parameters should have gradients
        for p in student.parameters():
            assert p.grad is None


# ==================================================================
# 3. AdaptiveLambdaController unit tests
# ==================================================================


class TestAdaptiveLambdaController:
    def test_initial_lambda(self) -> None:
        from rcid.distillation.adaptive_lambda import AdaptiveLambdaController
        ctrl = AdaptiveLambdaController(lambda_init=0.1)
        assert ctrl.get_lambda() == pytest.approx(0.1)

    def test_low_health_increases_lambda(self) -> None:
        from rcid.distillation.adaptive_lambda import AdaptiveLambdaController
        ctrl = AdaptiveLambdaController(lambda_init=0.1, target_health=0.7)
        # Health well below target -> lambda should increase
        init_lambda = ctrl.get_lambda()
        for _ in range(5):
            ctrl.update(health=0.3)
        assert ctrl.get_lambda() > init_lambda

    def test_high_health_decreases_lambda(self) -> None:
        from rcid.distillation.adaptive_lambda import AdaptiveLambdaController
        ctrl = AdaptiveLambdaController(lambda_init=0.5, target_health=0.7)
        # Health above target -> lambda should decrease
        init_lambda = ctrl.get_lambda()
        for _ in range(5):
            ctrl.update(health=0.95)
        assert ctrl.get_lambda() < init_lambda

    def test_lambda_clamped_to_min_max(self) -> None:
        from rcid.distillation.adaptive_lambda import AdaptiveLambdaController
        ctrl = AdaptiveLambdaController(
            lambda_init=0.1, lambda_min=0.01, lambda_max=1.0,
        )
        # Push very low
        for _ in range(50):
            ctrl.update(health=1.0)
        assert ctrl.get_lambda() >= 0.01
        # Push very high
        for _ in range(50):
            ctrl.update(health=0.0)
        assert ctrl.get_lambda() <= 1.0

    def test_at_target_lambda_stable(self) -> None:
        from rcid.distillation.adaptive_lambda import AdaptiveLambdaController
        ctrl = AdaptiveLambdaController(
            lambda_init=0.1, target_health=0.7, gain=0.5, ema_beta=0.9,
        )
        # Feed exact target health -> lambda should stay near initial
        for _ in range(10):
            ctrl.update(health=0.7)
        assert ctrl.get_lambda() == pytest.approx(0.1, rel=0.05)


# ==================================================================
# 4. Fake datasets for trainer tests
# ==================================================================


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


class _FakeContrastiveDataset(Dataset):
    """Tiny contrastive dataset for OCID trainer tests."""

    def __init__(self, n: int = 8, seq_len: int = 10, vocab_size: int = 100) -> None:
        self.clean_ids = torch.randint(2, vocab_size, (n, seq_len))
        self.corrupt_ids = self.clean_ids.clone()
        for i in range(n):
            pos = i % seq_len
            self.corrupt_ids[i, pos] = (self.clean_ids[i, pos] + 1) % vocab_size

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


# ==================================================================
# 5. OCID trainer integration tests
# ==================================================================


@pytest.mark.skip(reason="OCID trainer routing removed — standalone module tests remain")
class TestScalableTrainerOCID:
    """Test trainer with standard_kd_ocid method."""

    @pytest.fixture
    def ocid_setup(self) -> dict:
        torch.manual_seed(0)
        teacher = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        adapter = TinyAdapter()
        main_ds = _FakeInstructionDataset(n=8, seq_len=10, vocab_size=100)
        contrastive_ds = _FakeContrastiveDataset(n=8, seq_len=10, vocab_size=100)
        return {
            "teacher": teacher,
            "adapter": adapter,
            "main_ds": main_ds,
            "contrastive_ds": contrastive_ds,
        }

    def test_ocid_trains_without_error(self, ocid_setup: dict) -> None:
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        config = {
            "method": "standard_kd_ocid",
            "epochs": 1,
            "batch_size": 4,
            "gradient_accumulation": 1,
            "lr": 1e-3,
            "fp16": False,
            "temperature": 2.0,
            "use_wandb": False,
            "lambda_ocid": 0.1,
            "ocid_every_n_steps": 1,
        }
        torch.manual_seed(1)
        student = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)

        trainer = ScalableDistillationTrainer(
            teacher=ocid_setup["teacher"], student=student,
            teacher_adapter=ocid_setup["adapter"],
            student_adapter=ocid_setup["adapter"],
            tokenizer=None, main_dataset=ocid_setup["main_ds"],
            config=config,
            contrastive_dataset=ocid_setup["contrastive_ds"],
        )
        history = trainer.train()
        assert "loss" in history
        assert "ocid_loss" in history
        assert len(history["ocid_loss"]) == 1

    def test_ocid_student_params_updated(self, ocid_setup: dict) -> None:
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        config = {
            "method": "standard_kd_ocid",
            "epochs": 1,
            "batch_size": 4,
            "gradient_accumulation": 1,
            "lr": 1e-3,
            "fp16": False,
            "temperature": 2.0,
            "use_wandb": False,
            "lambda_ocid": 0.5,
            "ocid_every_n_steps": 1,
        }
        torch.manual_seed(1)
        student = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        before = {n: p.clone() for n, p in student.named_parameters()}

        trainer = ScalableDistillationTrainer(
            teacher=ocid_setup["teacher"], student=student,
            teacher_adapter=ocid_setup["adapter"],
            student_adapter=ocid_setup["adapter"],
            tokenizer=None, main_dataset=ocid_setup["main_ds"],
            config=config,
            contrastive_dataset=ocid_setup["contrastive_ds"],
        )
        trainer.train()

        any_changed = any(
            not torch.equal(before[n], p)
            for n, p in student.named_parameters()
        )
        assert any_changed, "Student parameters should be updated after training"

    def test_ocid_requires_contrastive_dataset(self, ocid_setup: dict) -> None:
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        config = {
            "method": "standard_kd_ocid",
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

        with pytest.raises(AssertionError, match="contrastive_dataset"):
            ScalableDistillationTrainer(
                teacher=ocid_setup["teacher"], student=student,
                teacher_adapter=ocid_setup["adapter"],
                student_adapter=ocid_setup["adapter"],
                tokenizer=None, main_dataset=ocid_setup["main_ds"],
                config=config,
                contrastive_dataset=None,  # should fail
            )

    def test_ocid_history_has_cos_sim(self, ocid_setup: dict) -> None:
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        config = {
            "method": "standard_kd_ocid",
            "epochs": 1,
            "batch_size": 4,
            "gradient_accumulation": 1,
            "lr": 1e-3,
            "fp16": False,
            "temperature": 2.0,
            "use_wandb": False,
            "lambda_ocid": 0.1,
            "ocid_every_n_steps": 1,
        }
        torch.manual_seed(1)
        student = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)

        trainer = ScalableDistillationTrainer(
            teacher=ocid_setup["teacher"], student=student,
            teacher_adapter=ocid_setup["adapter"],
            student_adapter=ocid_setup["adapter"],
            tokenizer=None, main_dataset=ocid_setup["main_ds"],
            config=config,
            contrastive_dataset=ocid_setup["contrastive_ds"],
        )
        history = trainer.train()
        assert "ocid_mean_cos_sim" in history
        assert "ocid_n_active" in history
        assert len(history["ocid_mean_cos_sim"]) == 1


# ==================================================================
# 6. Adaptive OCID trainer integration tests
# ==================================================================


@pytest.mark.skip(reason="OCID trainer routing removed — standalone module tests remain")
class TestScalableTrainerOCIDAdaptive:
    """Test trainer with standard_kd_ocid_adaptive method."""

    @pytest.fixture
    def adaptive_setup(self) -> dict:
        torch.manual_seed(0)
        teacher = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        adapter = TinyAdapter()
        main_ds = _FakeInstructionDataset(n=8, seq_len=10, vocab_size=100)
        contrastive_ds = _FakeContrastiveDataset(n=8, seq_len=10, vocab_size=100)
        return {
            "teacher": teacher,
            "adapter": adapter,
            "main_ds": main_ds,
            "contrastive_ds": contrastive_ds,
        }

    def test_adaptive_trains_without_error(self, adaptive_setup: dict) -> None:
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        config = {
            "method": "standard_kd_ocid_adaptive",
            "epochs": 1,
            "batch_size": 4,
            "gradient_accumulation": 1,
            "lr": 1e-3,
            "fp16": False,
            "temperature": 2.0,
            "use_wandb": False,
            "lambda_ocid": 0.1,
            "ocid_every_n_steps": 1,
            "monitor_every_n_steps": 1,
            "target_health": 0.7,
        }
        torch.manual_seed(1)
        student = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)

        trainer = ScalableDistillationTrainer(
            teacher=adaptive_setup["teacher"], student=student,
            teacher_adapter=adaptive_setup["adapter"],
            student_adapter=adaptive_setup["adapter"],
            tokenizer=None, main_dataset=adaptive_setup["main_ds"],
            config=config,
            contrastive_dataset=adaptive_setup["contrastive_ds"],
        )
        history = trainer.train()
        assert "loss" in history
        assert "ocid_loss" in history
        assert "lambda_ocid" in history
        assert "overall_health" in history

    def test_adaptive_history_has_lambda_and_health(self, adaptive_setup: dict) -> None:
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        config = {
            "method": "standard_kd_ocid_adaptive",
            "epochs": 1,
            "batch_size": 4,
            "gradient_accumulation": 1,
            "lr": 1e-3,
            "fp16": False,
            "temperature": 2.0,
            "use_wandb": False,
            "lambda_ocid": 0.1,
            "ocid_every_n_steps": 1,
            "monitor_every_n_steps": 1,
            "target_health": 0.7,
        }
        torch.manual_seed(1)
        student = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)

        trainer = ScalableDistillationTrainer(
            teacher=adaptive_setup["teacher"], student=student,
            teacher_adapter=adaptive_setup["adapter"],
            student_adapter=adaptive_setup["adapter"],
            tokenizer=None, main_dataset=adaptive_setup["main_ds"],
            config=config,
            contrastive_dataset=adaptive_setup["contrastive_ds"],
        )
        history = trainer.train()
        assert len(history["lambda_ocid"]) == 1
        assert len(history["overall_health"]) == 1
        # Lambda should be a positive float
        assert history["lambda_ocid"][0] > 0
        # Health should be in [-1, 1] range (cosine similarity)
        assert -1.1 <= history["overall_health"][0] <= 1.1

    def test_adaptive_requires_contrastive_dataset(self, adaptive_setup: dict) -> None:
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        config = {
            "method": "standard_kd_ocid_adaptive",
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

        with pytest.raises(AssertionError, match="contrastive_dataset"):
            ScalableDistillationTrainer(
                teacher=adaptive_setup["teacher"], student=student,
                teacher_adapter=adaptive_setup["adapter"],
                student_adapter=adaptive_setup["adapter"],
                tokenizer=None, main_dataset=adaptive_setup["main_ds"],
                config=config,
                contrastive_dataset=None,
            )

    def test_adaptive_student_params_updated(self, adaptive_setup: dict) -> None:
        from rcid.distillation.scalable_trainer import ScalableDistillationTrainer

        config = {
            "method": "standard_kd_ocid_adaptive",
            "epochs": 1,
            "batch_size": 4,
            "gradient_accumulation": 1,
            "lr": 1e-3,
            "fp16": False,
            "temperature": 2.0,
            "use_wandb": False,
            "lambda_ocid": 0.5,
            "ocid_every_n_steps": 1,
            "monitor_every_n_steps": 1,
            "target_health": 0.7,
        }
        torch.manual_seed(1)
        student = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        before = {n: p.clone() for n, p in student.named_parameters()}

        trainer = ScalableDistillationTrainer(
            teacher=adaptive_setup["teacher"], student=student,
            teacher_adapter=adaptive_setup["adapter"],
            student_adapter=adaptive_setup["adapter"],
            tokenizer=None, main_dataset=adaptive_setup["main_ds"],
            config=config,
            contrastive_dataset=adaptive_setup["contrastive_ds"],
        )
        trainer.train()

        any_changed = any(
            not torch.equal(before[n], p)
            for n, p in student.named_parameters()
        )
        assert any_changed, "Student parameters should be updated after training"
