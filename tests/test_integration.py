"""Integration tests: full pipeline from data → train → evaluate.

These tests use TinyTransformerModel (no real model downloads)
and exercise the complete flow for both simulated model families.
"""

from __future__ import annotations

import copy

import pytest
import torch

from conftest import TinyAdapter, TinyTransformerModel
from rcid import set_all_seeds
from rcid.alignment.cka import cka_matrix
from rcid.alignment.layer_matching import match_layers
from rcid.alignment.procrustes import compute_procrustes_matrices
from rcid.circuit.checkpoint_selection import select_checkpoints
from rcid.circuit.contrastive import ContrastiveDataset
from rcid.circuit.patching import extract_contrastive_differences
from rcid.distillation.trainer import UnifiedTrainer
from rcid.eval.causal_consistency import CausalConsistencyEvaluator
from rcid.eval.task_accuracy import evaluate_task_accuracy


def _make_dataset(
    n: int = 20,
    seq_len: int = 8,
    vocab: int = 100,
    seed: int = 42,
) -> ContrastiveDataset:
    """Build a small synthetic ContrastiveDataset."""
    torch.manual_seed(seed)
    # answer_pos at last position so patching there changes logit_diff
    ans_pos = seq_len - 1
    return ContrastiveDataset(
        clean_ids=torch.randint(0, vocab, (n, seq_len)),
        corrupt_ids=torch.randint(0, vocab, (n, seq_len)),
        answer_pos=torch.full((n,), ans_pos, dtype=torch.long),
        correct_token_id=torch.randint(0, vocab // 2, (n,)),
        wrong_token_id=torch.randint(vocab // 2, vocab, (n,)),
        key_positions={"s2": torch.full((n,), 3, dtype=torch.long)},
        is_modified={"s2": True},
        model_family="test",
    )


def _make_teacher_student(
    t_layers: int = 4,
    s_layers: int = 2,
    t_dim: int = 32,
    s_dim: int = 16,
    vocab: int = 100,
    seed: int = 42,
) -> tuple[TinyTransformerModel, TinyTransformerModel, TinyAdapter, TinyAdapter]:
    """Create teacher and student models."""
    torch.manual_seed(seed)
    teacher = TinyTransformerModel(n_layers=t_layers, d_model=t_dim, vocab_size=vocab)
    student = TinyTransformerModel(n_layers=s_layers, d_model=s_dim, vocab_size=vocab)
    teacher.eval()
    t_adapter = TinyAdapter()
    s_adapter = TinyAdapter()
    return teacher, student, t_adapter, s_adapter


def _run_pipeline(
    method: str,
    epochs: int = 3,
) -> dict:
    """Run full pipeline: extract → align → train → evaluate."""
    set_all_seeds(42)
    teacher, student, t_adapter, s_adapter = _make_teacher_student()
    ds = _make_dataset()

    # 1. Extract contrastive differences from teacher
    t_layers = list(range(t_adapter.get_num_layers(teacher)))
    t_diffs = extract_contrastive_differences(
        teacher, t_adapter, ds.clean_ids, ds.corrupt_ids, t_layers,
    )

    # 2. Select checkpoints
    checkpoints = select_checkpoints(t_diffs, ds, top_k=2, diversity_ratio=0.5)

    # 3. CKA → layer matching
    s_layers = list(range(s_adapter.get_num_layers(student)))
    s_diffs = extract_contrastive_differences(
        student, s_adapter, ds.clean_ids, ds.corrupt_ids, s_layers,
    )
    cka = cka_matrix(t_diffs, s_diffs)
    layer_map = match_layers(cka, strategy="greedy")

    # 4. Procrustes alignment
    W_matrices = compute_procrustes_matrices(t_diffs, s_diffs, layer_map)

    # 5. Train
    config = {
        "lr": 1e-3,
        "epochs": epochs,
        "batch_size": 10,
        "temperature": 2.0,
        "lambda_kl": 1.0,
        "lambda_rcid": 1.0,
        "fp16": False,
        "grad_clip": 1.0,
    }
    trainer = UnifiedTrainer(
        method=method,
        teacher=teacher,
        student=student,
        teacher_adapter=t_adapter,
        student_adapter=s_adapter,
        dataset=ds,
        config=config,
        checkpoints=checkpoints,
        layer_mapping=layer_map,
        W_matrices=W_matrices,
    )
    history = trainer.train(epochs=epochs, batch_size=10)

    # 6. Evaluate
    acc_result = evaluate_task_accuracy(student, s_adapter, ds)
    evaluator = CausalConsistencyEvaluator()
    # Use answer_pos as patch position (since TinyModel has no cross-pos interaction)
    eval_checkpoints = [(cp[0], ds.answer_pos[0].item()) for cp in checkpoints]
    # Map teacher layer → student layer for evaluation
    eval_map = {cp[0]: layer_map.get(cp[0], cp[0]) for cp in eval_checkpoints}
    cc_result = evaluator.evaluate(
        teacher=teacher, student=student,
        teacher_adapter=t_adapter, student_adapter=s_adapter,
        dataset=ds, checkpoints=eval_checkpoints, layer_mapping=eval_map,
    )

    return {
        "history": history,
        "accuracy": acc_result,
        "causal_consistency": cc_result,
    }


# ---------------------------------------------------------------------------
# Pipeline tests for each distillation method
# ---------------------------------------------------------------------------


class TestStandardKDPipeline:
    """End-to-end test for StandardKD method."""

    def test_full_pipeline(self) -> None:
        result = _run_pipeline("standard_kd", epochs=3)
        assert len(result["history"]["loss"]) == 3
        assert all(l > 0 for l in result["history"]["loss"])
        assert "accuracy" in result["accuracy"]

    def test_loss_decreases(self) -> None:
        result = _run_pipeline("standard_kd", epochs=10)
        losses = result["history"]["loss"]
        # Loss should generally decrease over 10 epochs
        assert losses[-1] < losses[0], (
            f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )


class TestRCIDPipeline:
    """End-to-end test for RCID method."""

    def test_full_pipeline(self) -> None:
        result = _run_pipeline("rcid", epochs=3)
        assert len(result["history"]["loss"]) == 3
        assert "accuracy" in result["accuracy"]
        assert "mean_correlation" in result["causal_consistency"]

    def test_loss_decreases(self) -> None:
        result = _run_pipeline("rcid", epochs=10)
        losses = result["history"]["loss"]
        assert losses[-1] < losses[0], (
            f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )


class TestFitNetsPipeline:
    """End-to-end test for FitNets method."""

    def test_full_pipeline(self) -> None:
        result = _run_pipeline("fitnets", epochs=3)
        assert len(result["history"]["loss"]) == 3
        assert "accuracy" in result["accuracy"]


class TestInformedFitNetsPipeline:
    """End-to-end test for InformedFitNets method."""

    def test_full_pipeline(self) -> None:
        result = _run_pipeline("informed_fitnets", epochs=3)
        assert len(result["history"]["loss"]) == 3
        assert "accuracy" in result["accuracy"]


# ---------------------------------------------------------------------------
# Cross-method consistency
# ---------------------------------------------------------------------------


class TestCrossMethodComparison:
    """Compare all methods produce valid results."""

    def test_all_methods_produce_results(self) -> None:
        for method in ("standard_kd", "fitnets", "informed_fitnets", "rcid"):
            result = _run_pipeline(method, epochs=2)
            assert len(result["history"]["loss"]) == 2, f"{method} failed"

    def test_student_params_change(self) -> None:
        """Student parameters should change after training."""
        set_all_seeds(42)
        teacher, student, t_adapter, s_adapter = _make_teacher_student()
        params_before = {
            n: p.clone() for n, p in student.named_parameters()
        }
        ds = _make_dataset()
        config = {"lr": 1e-3, "epochs": 3, "batch_size": 10, "fp16": False}
        trainer = UnifiedTrainer(
            method="standard_kd",
            teacher=teacher, student=student,
            teacher_adapter=t_adapter, student_adapter=s_adapter,
            dataset=ds, config=config,
        )
        trainer.train(epochs=3, batch_size=10)

        changed = False
        for n, p in student.named_parameters():
            if not torch.allclose(params_before[n], p):
                changed = True
                break
        assert changed, "Student params unchanged after training"

    def test_teacher_params_frozen(self) -> None:
        """Teacher parameters must not change during training."""
        set_all_seeds(42)
        teacher, student, t_adapter, s_adapter = _make_teacher_student()
        params_before = {
            n: p.clone() for n, p in teacher.named_parameters()
        }
        ds = _make_dataset()
        config = {"lr": 1e-3, "epochs": 3, "batch_size": 10, "fp16": False}
        trainer = UnifiedTrainer(
            method="rcid",
            teacher=teacher, student=student,
            teacher_adapter=t_adapter, student_adapter=s_adapter,
            dataset=ds, config=config,
            checkpoints=[(1, 7)],
            layer_mapping={1: 0},
            W_matrices={1: torch.eye(32, 16)},
        )
        trainer.train(epochs=3, batch_size=10)

        for n, p in teacher.named_parameters():
            assert torch.allclose(params_before[n], p), f"Teacher param {n} changed!"
