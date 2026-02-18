"""Tests for distillation losses and trainer."""

import pytest
import torch
import torch.nn as nn

from conftest import TinyAdapter, TinyTransformerModel
from rcid.circuit.contrastive import ContrastiveDataset
from rcid.distillation.baselines import (
    FitNetsLoss,
    InformedFitNetsLoss,
    StandardKDLoss,
)
from rcid.distillation.rcid_loss import RCIDLoss
from rcid.distillation.trainer import UnifiedTrainer


# ---------------------------------------------------------------------------
# StandardKD tests
# ---------------------------------------------------------------------------

class TestStandardKDLoss:
    def test_identical_distributions(self) -> None:
        logits = torch.randn(4, 100)
        loss = StandardKDLoss(temperature=2.0)(logits, logits)
        assert loss.item() < 1e-4

    def test_different_distributions(self) -> None:
        t_logits = torch.randn(4, 100)
        s_logits = torch.randn(4, 100)
        loss = StandardKDLoss()(t_logits, s_logits)
        assert loss.item() > 0
        assert loss.isfinite()

    def test_gradient_flows(self) -> None:
        t_logits = torch.randn(4, 100)
        s_logits = torch.randn(4, 100, requires_grad=True)
        loss = StandardKDLoss()(t_logits, s_logits)
        loss.backward()
        assert s_logits.grad is not None
        assert s_logits.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# RCID loss tests
# ---------------------------------------------------------------------------

class TestRCIDLoss:
    @pytest.fixture
    def setup(self):
        d_T, d_S = 64, 16
        checkpoints = [(0, 3), (2, 5)]
        layer_mapping = {0: 0, 2: 1}
        W = {0: torch.eye(d_T, d_S), 2: torch.eye(d_T, d_S)}
        loss_fn = RCIDLoss(checkpoints, layer_mapping, W)
        return loss_fn, d_T, d_S, checkpoints

    def test_output_is_scalar(self, setup) -> None:
        loss_fn, d_T, d_S, checkpoints = setup
        teacher_imprints = {
            (0, 3): torch.randn(4, d_T),
            (2, 5): torch.randn(4, d_T),
        }
        s_clean = {0: torch.randn(4, 10, d_S), 1: torch.randn(4, 10, d_S)}
        s_corrupt = {0: torch.randn(4, 10, d_S), 1: torch.randn(4, 10, d_S)}
        loss = loss_fn(teacher_imprints, s_clean, s_corrupt)
        assert loss.dim() == 0
        assert loss.isfinite()

    def test_gradient_only_through_student(self, setup) -> None:
        loss_fn, d_T, d_S, checkpoints = setup
        teacher_imprints = {
            (0, 3): torch.randn(4, d_T),  # detached
            (2, 5): torch.randn(4, d_T),
        }
        s_clean = {
            0: torch.randn(4, 10, d_S, requires_grad=True),
            1: torch.randn(4, 10, d_S, requires_grad=True),
        }
        s_corrupt = {
            0: torch.randn(4, 10, d_S, requires_grad=True),
            1: torch.randn(4, 10, d_S, requires_grad=True),
        }
        loss = loss_fn(teacher_imprints, s_clean, s_corrupt)
        loss.backward()
        for layer in s_clean.values():
            assert layer.grad is not None
        for layer in s_corrupt.values():
            assert layer.grad is not None

    def test_near_zero_when_aligned(self) -> None:
        """If student diffs == teacher diffs (with identity W), loss ≈ 0."""
        d = 16
        checkpoints = [(0, 2)]
        layer_mapping = {0: 0}
        W = {0: torch.eye(d, d)}
        loss_fn = RCIDLoss(checkpoints, layer_mapping, W)

        t_diff = torch.randn(4, d)
        t_diff = t_diff / t_diff.norm(dim=-1, keepdim=True)

        # Construct student residuals so that d_S = t_diff
        s_clean = {0: torch.zeros(4, 5, d)}
        s_corrupt = {0: torch.zeros(4, 5, d)}
        s_clean[0][:, 2, :] = t_diff
        # s_corrupt stays zero, so d_S = s_clean - s_corrupt = t_diff

        loss = loss_fn({(0, 2): t_diff}, s_clean, s_corrupt)
        assert loss.item() < 1e-4, f"Expected near-zero loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# InformedFitNets tests
# ---------------------------------------------------------------------------

class TestInformedFitNetsLoss:
    def test_positive_loss(self) -> None:
        d_T, d_S = 32, 16
        checkpoints = [(0, 3)]
        layer_mapping = {0: 0}
        W = {0: torch.randn(d_T, d_S)}
        loss_fn = InformedFitNetsLoss(checkpoints, layer_mapping, W)

        t_clean = {0: torch.randn(4, 10, d_T)}
        s_clean = {0: torch.randn(4, 10, d_S, requires_grad=True)}

        loss = loss_fn(t_clean, s_clean)
        assert loss.item() > 0
        assert loss.isfinite()
        loss.backward()
        assert s_clean[0].grad is not None


# ---------------------------------------------------------------------------
# FitNets tests
# ---------------------------------------------------------------------------

class TestFitNetsLoss:
    def test_positive_loss(self) -> None:
        d_T, d_S = 32, 16
        layer_mapping = {0: 0, 1: 1}
        W = {0: torch.randn(d_T, d_S), 1: torch.randn(d_T, d_S)}
        loss_fn = FitNetsLoss(layer_mapping, W)

        t_res = {0: torch.randn(4, 8, d_T), 1: torch.randn(4, 8, d_T)}
        s_res = {
            0: torch.randn(4, 8, d_S, requires_grad=True),
            1: torch.randn(4, 8, d_S, requires_grad=True),
        }
        loss = loss_fn(t_res, s_res)
        assert loss.item() > 0
        assert loss.isfinite()


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------

class TestUnifiedTrainer:
    @pytest.fixture
    def tiny_setup(self):
        torch.manual_seed(42)
        teacher = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        student = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        adapter = TinyAdapter()

        n, seq_len = 8, 10
        ds = ContrastiveDataset(
            clean_ids=torch.randint(0, 100, (n, seq_len)),
            corrupt_ids=torch.randint(0, 100, (n, seq_len)),
            answer_pos=torch.full((n,), seq_len - 1, dtype=torch.long),
            correct_token_id=torch.randint(0, 100, (n,)),
            wrong_token_id=torch.randint(0, 100, (n,)),
            key_positions={"s2": torch.full((n,), 3, dtype=torch.long)},
            is_modified={"s2": True},
            model_family="test",
        )
        config = {"epochs": 2, "batch_size": 4, "lr": 1e-3, "fp16": False}
        return teacher, student, adapter, ds, config

    def test_standard_kd_trains(self, tiny_setup) -> None:
        teacher, student, adapter, ds, config = tiny_setup
        trainer = UnifiedTrainer(
            method="standard_kd",
            teacher=teacher, student=student,
            teacher_adapter=adapter, student_adapter=adapter,
            dataset=ds, config=config,
        )
        history = trainer.train()
        assert len(history["loss"]) == 2
        assert all(l > 0 for l in history["loss"])

    def test_rcid_trains(self, tiny_setup) -> None:
        teacher, student, adapter, ds, config = tiny_setup
        checkpoints = [(2, 3)]
        layer_mapping = {2: 2}
        W = {2: torch.eye(32, 32)}
        trainer = UnifiedTrainer(
            method="rcid",
            teacher=teacher, student=student,
            teacher_adapter=adapter, student_adapter=adapter,
            dataset=ds, config=config,
            checkpoints=checkpoints,
            layer_mapping=layer_mapping,
            W_matrices=W,
        )
        history = trainer.train()
        assert len(history["loss"]) == 2

    def test_teacher_not_updated(self, tiny_setup) -> None:
        teacher, student, adapter, ds, config = tiny_setup
        config["epochs"] = 1
        t_params_before = {n: p.clone() for n, p in teacher.named_parameters()}

        trainer = UnifiedTrainer(
            method="standard_kd",
            teacher=teacher, student=student,
            teacher_adapter=adapter, student_adapter=adapter,
            dataset=ds, config=config,
        )
        trainer.train()

        for n, p in teacher.named_parameters():
            assert torch.equal(p, t_params_before[n]), f"Teacher param {n} changed!"

    def test_student_updated(self, tiny_setup) -> None:
        teacher, student, adapter, ds, config = tiny_setup
        config["epochs"] = 1
        s_params_before = {n: p.clone() for n, p in student.named_parameters()}

        trainer = UnifiedTrainer(
            method="standard_kd",
            teacher=teacher, student=student,
            teacher_adapter=adapter, student_adapter=adapter,
            dataset=ds, config=config,
        )
        trainer.train()

        any_changed = any(
            not torch.equal(p, s_params_before[n])
            for n, p in student.named_parameters()
        )
        assert any_changed, "Student parameters should change after training"
