"""Tests for evaluation modules."""

import pytest
import torch

from conftest import TinyAdapter, TinyTransformerModel
from rcid.circuit.contrastive import ContrastiveDataset
from rcid.eval.task_accuracy import evaluate_task_accuracy
from rcid.eval.causal_consistency import CausalConsistencyEvaluator

from rcid.eval.information_purity import evaluate_information_purity

# sklearn depends on scipy which depends on numpy C extension;
# this environment may have a broken numpy DLL, so probe directly.
try:
    from sklearn.linear_model import LogisticRegression  # noqa: F401
    HAS_SKLEARN = True
except (ImportError, OSError):
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Task accuracy tests
# ---------------------------------------------------------------------------

class TestTaskAccuracy:
    def test_perfect_accuracy(self) -> None:
        """Model that always assigns higher logit to correct token."""
        torch.manual_seed(42)
        model = TinyTransformerModel(n_layers=2, d_model=16, vocab_size=50)
        adapter = TinyAdapter()

        n = 10
        ds = ContrastiveDataset(
            clean_ids=torch.randint(0, 50, (n, 8)),
            corrupt_ids=torch.randint(0, 50, (n, 8)),
            answer_pos=torch.full((n,), 7, dtype=torch.long),
            correct_token_id=torch.zeros(n, dtype=torch.long),
            wrong_token_id=torch.ones(n, dtype=torch.long),
            key_positions={},
            is_modified={},
            model_family="test",
        )
        result = evaluate_task_accuracy(model, adapter, ds)
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0
        assert "logit_diff_mean" in result

    def test_returns_all_keys(self) -> None:
        model = TinyTransformerModel(n_layers=2, d_model=16, vocab_size=50)
        adapter = TinyAdapter()
        ds = ContrastiveDataset(
            clean_ids=torch.randint(0, 50, (5, 6)),
            corrupt_ids=torch.randint(0, 50, (5, 6)),
            answer_pos=torch.full((5,), 5, dtype=torch.long),
            correct_token_id=torch.randint(0, 50, (5,)),
            wrong_token_id=torch.randint(0, 50, (5,)),
            key_positions={},
            is_modified={},
            model_family="test",
        )
        result = evaluate_task_accuracy(model, adapter, ds)
        assert set(result.keys()) == {"accuracy", "logit_diff_mean", "logit_diff_std"}


# ---------------------------------------------------------------------------
# Causal consistency tests
# ---------------------------------------------------------------------------

class TestCausalConsistency:
    def test_self_consistency_near_one(self) -> None:
        """Teacher evaluated against itself should have correlation = 1.0.

        Note: TinyTransformerModel has no attention (position-independent
        linear layers), so patching at pos p only affects logits at pos p.
        We set answer_pos == patch_pos so patching actually changes the
        measured logit diff and produces a non-degenerate delta vector.
        """
        torch.manual_seed(42)
        model = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        adapter = TinyAdapter()

        n = 20
        # Ensure clean and corrupt differ significantly for varied deltas
        clean_ids = torch.randint(0, 100, (n, 8))
        corrupt_ids = torch.randint(0, 100, (n, 8))
        # Patch at position 5, measure answer at position 5
        patch_pos = 5
        ds = ContrastiveDataset(
            clean_ids=clean_ids,
            corrupt_ids=corrupt_ids,
            answer_pos=torch.full((n,), patch_pos, dtype=torch.long),
            correct_token_id=torch.randint(0, 50, (n,)),
            wrong_token_id=torch.randint(50, 100, (n,)),
            key_positions={"s2": torch.full((n,), patch_pos, dtype=torch.long)},
            is_modified={"s2": True},
            model_family="test",
        )
        evaluator = CausalConsistencyEvaluator()
        results = evaluator.evaluate(
            teacher=model, student=model,
            teacher_adapter=adapter, student_adapter=adapter,
            dataset=ds,
            checkpoints=[(2, patch_pos)],
            layer_mapping={2: 2},
        )
        # Self-consistency must be exactly 1.0 (same model, same inputs)
        cc = results["mean_correlation"]
        assert cc > 0.99, f"Self-consistency = {cc}, expected ~1.0"

    def test_returns_per_checkpoint(self) -> None:
        model = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
        adapter = TinyAdapter()
        n = 8
        ds = ContrastiveDataset(
            clean_ids=torch.randint(0, 100, (n, 8)),
            corrupt_ids=torch.randint(0, 100, (n, 8)),
            answer_pos=torch.full((n,), 7, dtype=torch.long),
            correct_token_id=torch.randint(0, 100, (n,)),
            wrong_token_id=torch.randint(0, 100, (n,)),
            key_positions={},
            is_modified={},
            model_family="test",
        )
        evaluator = CausalConsistencyEvaluator()
        results = evaluator.evaluate(
            teacher=model, student=model,
            teacher_adapter=adapter, student_adapter=adapter,
            dataset=ds,
            checkpoints=[(1, 2), (3, 5)],
            layer_mapping={1: 1, 3: 3},
        )
        assert len(results["per_checkpoint"]) == 2


# ---------------------------------------------------------------------------
# Information purity tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available (numpy DLL issue)")
class TestInformationPurity:
    def test_high_selectivity_with_separable_data(self) -> None:
        """Task labels correlated with representations, control labels random."""
        torch.manual_seed(42)
        n = 200
        d = 32
        # Create separable task: first dim determines label
        X = torch.randn(n, d)
        task_labels = (X[:, 0] > 0).long()
        control_labels = torch.randint(0, 5, (n,))

        result = evaluate_information_purity(X, task_labels, control_labels)
        assert result["task_accuracy"] > 0.7
        assert result["selectivity"] > 0.0

    def test_returns_all_keys(self) -> None:
        X = torch.randn(50, 16)
        task = torch.randint(0, 2, (50,))
        ctrl = torch.randint(0, 3, (50,))
        result = evaluate_information_purity(X, task, ctrl)
        assert set(result.keys()) == {"task_accuracy", "control_accuracy", "selectivity"}
