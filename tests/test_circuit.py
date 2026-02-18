"""Tests for circuit analysis: patching, intervention, checkpoint selection."""

import pytest
import torch

from conftest import TinyAdapter, TinyTransformerModel
from rcid.circuit.contrastive import ContrastiveDataset
from rcid.circuit.patching import (
    extract_contrastive_differences,
    extract_residual_at_layers,
)
from rcid.circuit.intervention import compute_causal_effect, patch_and_run
from rcid.circuit.checkpoint_selection import select_checkpoints


@pytest.fixture
def model_and_adapter():
    torch.manual_seed(42)
    model = TinyTransformerModel(n_layers=4, d_model=32, vocab_size=100)
    adapter = TinyAdapter()
    return model, adapter


# ---------------------------------------------------------------------------
# Patching (Read) tests
# ---------------------------------------------------------------------------

class TestExtractResidual:
    def test_captures_all_layers(self, model_and_adapter) -> None:
        model, adapter = model_and_adapter
        ids = torch.randint(0, 100, (2, 8))
        cache = extract_residual_at_layers(model, adapter, ids)
        assert len(cache) == 4
        for layer_idx, tensor in cache.items():
            assert tensor.shape == (2, 8, 32)  # (batch, seq, d_model)

    def test_captures_specific_layers(self, model_and_adapter) -> None:
        model, adapter = model_and_adapter
        ids = torch.randint(0, 100, (2, 8))
        cache = extract_residual_at_layers(model, adapter, ids, layers=[0, 3])
        assert set(cache.keys()) == {0, 3}

    def test_hooks_cleaned_up(self, model_and_adapter) -> None:
        model, adapter = model_and_adapter
        n_hooks_before = len(model.model.layers[0]._forward_hooks)
        extract_residual_at_layers(model, adapter, torch.randint(0, 100, (1, 5)))
        n_hooks_after = len(model.model.layers[0]._forward_hooks)
        assert n_hooks_before == n_hooks_after


class TestContrastiveDifferences:
    def test_zero_when_identical(self, model_and_adapter) -> None:
        model, adapter = model_and_adapter
        ids = torch.randint(0, 100, (2, 8))
        diffs = extract_contrastive_differences(model, adapter, ids, ids)
        for layer_idx, d in diffs.items():
            assert d.abs().max() < 1e-6, f"Layer {layer_idx}: non-zero diff"

    def test_nonzero_when_different(self, model_and_adapter) -> None:
        model, adapter = model_and_adapter
        clean = torch.randint(0, 50, (2, 8))
        corrupt = torch.randint(50, 100, (2, 8))
        diffs = extract_contrastive_differences(model, adapter, clean, corrupt)
        any_nonzero = any(d.abs().max() > 1e-6 for d in diffs.values())
        assert any_nonzero


# ---------------------------------------------------------------------------
# Intervention (Write) tests
# ---------------------------------------------------------------------------

class TestPatchAndRun:
    def test_patching_changes_output(self, model_and_adapter) -> None:
        model, adapter = model_and_adapter
        ids = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            orig_logits = model(ids).logits

        patch_val = torch.randn(2, 32) * 10.0
        patched_logits = patch_and_run(model, adapter, ids, patch_val, layer=2, token_pos=3)
        assert not torch.allclose(orig_logits, patched_logits, atol=1e-4)

    def test_no_patch_no_change(self, model_and_adapter) -> None:
        model, adapter = model_and_adapter
        ids = torch.randint(0, 100, (2, 8))
        # Capture original value at layer 2, pos 3
        cache = extract_residual_at_layers(model, adapter, ids, layers=[2])
        orig_val = cache[2][:, 3, :]  # (batch, d_model)

        patched_logits = patch_and_run(model, adapter, ids, orig_val, layer=2, token_pos=3)
        with torch.no_grad():
            orig_logits = model(ids).logits
        assert torch.allclose(patched_logits, orig_logits, atol=1e-5)


class TestCausalEffect:
    def test_delta_zero_when_same_input(self, model_and_adapter) -> None:
        model, adapter = model_and_adapter
        ids = torch.randint(0, 100, (3, 8))
        delta = compute_causal_effect(
            model, adapter,
            clean_input=ids, corrupt_input=ids,
            layer=2, token_pos=3, answer_pos=7,
            correct_token_id=torch.tensor([10, 20, 30]),
            wrong_token_id=torch.tensor([40, 50, 60]),
        )
        assert delta.shape == (3,)
        assert delta.abs().max() < 1e-4

    def test_delta_shape(self, model_and_adapter) -> None:
        model, adapter = model_and_adapter
        clean = torch.randint(0, 50, (4, 8))
        corrupt = torch.randint(50, 100, (4, 8))
        delta = compute_causal_effect(
            model, adapter,
            clean_input=clean, corrupt_input=corrupt,
            layer=2, token_pos=3, answer_pos=7,
            correct_token_id=torch.randint(0, 100, (4,)),
            wrong_token_id=torch.randint(0, 100, (4,)),
        )
        assert delta.shape == (4,)


# ---------------------------------------------------------------------------
# Checkpoint selection tests
# ---------------------------------------------------------------------------

class TestCheckpointSelection:
    def test_returns_top_k(self) -> None:
        diffs = {
            0: torch.randn(2, 10, 32),
            1: torch.randn(2, 10, 32) * 2,
            2: torch.randn(2, 10, 32) * 5,
            3: torch.randn(2, 10, 32) * 3,
        }
        ds = ContrastiveDataset(
            clean_ids=torch.zeros(2, 10, dtype=torch.long),
            corrupt_ids=torch.zeros(2, 10, dtype=torch.long),
            answer_pos=torch.zeros(2, dtype=torch.long),
            correct_token_id=torch.zeros(2, dtype=torch.long),
            wrong_token_id=torch.zeros(2, dtype=torch.long),
            key_positions={"s2": torch.tensor([3, 3])},
            is_modified={"s2": True},
            model_family="test",
        )
        checkpoints = select_checkpoints(diffs, ds, top_k=5)
        assert len(checkpoints) <= 5
        assert all(isinstance(cp, tuple) and len(cp) == 2 for cp in checkpoints)

    def test_diversity_includes_unmodified(self) -> None:
        diffs = {
            i: torch.randn(2, 10, 32) * (i + 1) for i in range(4)
        }
        ds = ContrastiveDataset(
            clean_ids=torch.zeros(2, 10, dtype=torch.long),
            corrupt_ids=torch.zeros(2, 10, dtype=torch.long),
            answer_pos=torch.zeros(2, dtype=torch.long),
            correct_token_id=torch.zeros(2, dtype=torch.long),
            wrong_token_id=torch.zeros(2, dtype=torch.long),
            key_positions={
                "mod_pos": torch.tensor([5, 5]),
                "unmod_pos": torch.tensor([2, 2]),
            },
            is_modified={"mod_pos": True, "unmod_pos": False},
            model_family="test",
        )
        checkpoints = select_checkpoints(diffs, ds, top_k=10, diversity_ratio=0.5)
        # Should include positions other than 5
        positions = {cp[1] for cp in checkpoints}
        assert len(positions) > 1, "Should include diverse positions"
