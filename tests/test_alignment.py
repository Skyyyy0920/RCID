"""Tests for alignment: CKA, layer matching, Procrustes."""

import pytest
import torch

from rcid.alignment.cka import cka_matrix, linear_cka
from rcid.alignment.layer_matching import (
    match_layers,
    proportional_layer_mapping,
)
from rcid.alignment.procrustes import compute_procrustes_matrices, procrustes_align


# ---------------------------------------------------------------------------
# CKA tests
# ---------------------------------------------------------------------------

class TestLinearCKA:
    def test_identical_representations(self) -> None:
        X = torch.randn(50, 32)
        val = linear_cka(X, X)
        assert abs(val - 1.0) < 1e-4, f"CKA(X,X) = {val}, expected 1.0"

    def test_orthogonal_representations(self) -> None:
        torch.manual_seed(42)
        X = torch.randn(100, 16)
        # Create Y orthogonal to X by projecting out X's column space
        Y = torch.randn(100, 16)
        # For large random matrices, CKA should be near 0
        # (not exactly 0, but small)
        val = linear_cka(X, Y)
        assert val < 0.5, f"CKA of random matrices should be low: {val}"

    def test_rotation_invariance(self) -> None:
        torch.manual_seed(42)
        X = torch.randn(50, 16)
        Q, _ = torch.linalg.qr(torch.randn(16, 16))
        Y = X @ Q  # rotated version
        val = linear_cka(X, Y)
        assert abs(val - 1.0) < 1e-3, f"CKA should be rotation invariant: {val}"

    def test_different_dimensions(self) -> None:
        X = torch.randn(50, 16)
        Y = torch.randn(50, 32)
        val = linear_cka(X, Y)
        assert 0.0 <= val <= 1.0 + 1e-6


class TestCKAMatrix:
    def test_shape(self) -> None:
        t_reps = {0: torch.randn(20, 64), 1: torch.randn(20, 64)}
        s_reps = {0: torch.randn(20, 32), 1: torch.randn(20, 32), 2: torch.randn(20, 32)}
        mat = cka_matrix(t_reps, s_reps)
        assert mat.shape == (2, 3)

    def test_diagonal_high_with_same_data(self) -> None:
        reps = {i: torch.randn(30, 16) for i in range(3)}
        mat = cka_matrix(reps, reps)
        for i in range(3):
            assert mat[i, i] > 0.99


# ---------------------------------------------------------------------------
# Layer matching tests
# ---------------------------------------------------------------------------

class TestLayerMatching:
    def test_greedy_picks_best(self) -> None:
        scores = torch.tensor([
            [0.1, 0.9, 0.2],
            [0.8, 0.3, 0.1],
            [0.2, 0.1, 0.7],
        ])
        mapping = match_layers(cka_scores=scores, strategy="greedy")
        assert mapping == {0: 1, 1: 0, 2: 2}

    def test_proportional_mapping(self) -> None:
        mapping = proportional_layer_mapping(36, 28)
        assert mapping[0] == 0
        assert mapping[35] == 27
        assert len(mapping) == 36

    def test_proportional_small(self) -> None:
        mapping = proportional_layer_mapping(4, 2)
        assert mapping[0] == 0
        assert mapping[3] == 1

    def test_match_layers_dispatch(self) -> None:
        scores = torch.eye(3)
        m = match_layers(cka_scores=scores, strategy="greedy")
        assert m == {0: 0, 1: 1, 2: 2}

        m2 = match_layers(n_teacher=4, n_student=2, strategy="proportional")
        assert len(m2) == 4


# ---------------------------------------------------------------------------
# Procrustes tests
# ---------------------------------------------------------------------------

class TestProcrustes:
    def test_shape(self) -> None:
        source = torch.randn(50, 16)  # student dim
        target = torch.randn(50, 64)  # teacher dim
        W = procrustes_align(source, target)
        assert W.shape == (64, 16)

    def test_orthogonality_square(self) -> None:
        """For square matrices, W should be approximately orthogonal."""
        torch.manual_seed(42)
        source = torch.randn(100, 32)
        target = torch.randn(100, 32)
        W = procrustes_align(source, target)
        # W @ W^T should be close to identity
        eye = torch.eye(32)
        diff = (W @ W.T - eye).abs().max()
        assert diff < 0.1, f"W not orthogonal: max deviation {diff}"

    def test_synthetic_recovery(self) -> None:
        """If target = source @ W_true^T, procrustes should recover W_true."""
        torch.manual_seed(42)
        d = 16
        # Create a known orthogonal W
        Q, _ = torch.linalg.qr(torch.randn(d, d))
        source = torch.randn(200, d)
        target = source @ Q.T  # target = source @ W_true^T

        W_recovered = procrustes_align(source, target)
        # W_recovered should be close to Q
        reconstruction = source @ W_recovered.T
        # Normalize for comparison (procrustes normalizes internally)
        target_n = target / target.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        recon_n = reconstruction / reconstruction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos_sim = (target_n * recon_n).sum(dim=-1).mean()
        assert cos_sim > 0.95, f"Poor recovery: cos_sim = {cos_sim}"


class TestComputeProcrustesMatrices:
    def test_returns_per_layer(self) -> None:
        t_diffs = {0: torch.randn(20, 64), 2: torch.randn(20, 64)}
        s_diffs = {0: torch.randn(20, 16), 1: torch.randn(20, 16)}
        mapping = {0: 0, 2: 1}
        Ws = compute_procrustes_matrices(t_diffs, s_diffs, mapping)
        assert 0 in Ws and 2 in Ws
        assert Ws[0].shape == (64, 16)
        assert Ws[2].shape == (64, 16)
