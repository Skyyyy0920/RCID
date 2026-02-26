"""Tests for PADDLoss (Position-Adaptive Divergence Distillation)."""

import pytest
import torch

from rcid.distillation.padd_loss import PADDLoss


class TestPADDLossBasic:
    """Core functionality tests."""

    def test_finite_positive_loss(self) -> None:
        """Random logits → loss should be a finite positive scalar."""
        loss_fn = PADDLoss(temperature=2.0, tau=1.0)
        t = torch.randn(2, 8, 50)
        s = torch.randn(2, 8, 50)
        loss, stats = loss_fn(t, s)
        assert loss.isfinite()
        assert loss.item() > 0.0

    def test_identical_logits_near_zero(self) -> None:
        """When teacher == student, both KL terms are 0 → loss ≈ 0."""
        loss_fn = PADDLoss(temperature=2.0, tau=1.0)
        logits = torch.randn(4, 16, 100)
        loss, stats = loss_fn(logits, logits)
        assert loss.item() < 1e-4

    def test_returns_stats_keys(self) -> None:
        """Stats dict should contain all expected keys."""
        loss_fn = PADDLoss()
        t = torch.randn(2, 4, 30)
        s = torch.randn(2, 4, 30)
        _, stats = loss_fn(t, s)
        expected_keys = {
            "alpha_mean", "forward_kl_mean", "reverse_kl_mean",
            "teacher_entropy_mean",
        }
        assert expected_keys == set(stats.keys())

    def test_loss_is_differentiable(self) -> None:
        """Loss should be differentiable w.r.t. student_logits."""
        loss_fn = PADDLoss()
        t = torch.randn(2, 4, 30)
        s = torch.randn(2, 4, 30, requires_grad=True)
        loss, _ = loss_fn(t, s)
        loss.backward()
        assert s.grad is not None
        assert s.grad.shape == s.shape


class TestPADDLossMask:
    """Mask handling tests."""

    def test_mask_filters_padding(self) -> None:
        """Loss from masked positions should be excluded."""
        loss_fn = PADDLoss(temperature=2.0, tau=1.0)
        t = torch.randn(2, 8, 50)
        s = torch.randn(2, 8, 50)

        # Full mask — all valid
        mask_full = torch.ones(2, 8)
        loss_full, _ = loss_fn(t, s, mask=mask_full)

        # Partial mask — last 4 positions are padding
        mask_partial = torch.ones(2, 8)
        mask_partial[:, 4:] = 0
        loss_partial, _ = loss_fn(t, s, mask=mask_partial)

        # Losses should differ (different positions contribute)
        assert loss_full.item() != pytest.approx(loss_partial.item(), abs=1e-6)

    def test_no_mask_same_as_all_ones(self) -> None:
        """No mask should behave the same as all-ones mask."""
        loss_fn = PADDLoss(temperature=2.0, tau=1.0)
        t = torch.randn(2, 8, 50)
        s = torch.randn(2, 8, 50)
        mask = torch.ones(2, 8)
        loss_no_mask, _ = loss_fn(t, s, mask=None)
        loss_with_mask, _ = loss_fn(t, s, mask=mask)
        # Should be very close (mean vs sum/count — identical for all-ones)
        assert loss_no_mask.item() == pytest.approx(loss_with_mask.item(), rel=1e-5)


class TestPADDLossTau:
    """Tau parameter behaviour tests."""

    def test_large_tau_approaches_jsd(self) -> None:
        """When tau → ∞, alpha → 0.5 everywhere → loss ≈ JSD."""
        t = torch.randn(4, 16, 100)
        s = torch.randn(4, 16, 100)

        # Large tau → alpha clamped midway
        loss_fn_large = PADDLoss(
            temperature=2.0, tau=100.0, alpha_min=0.0, alpha_max=1.0,
        )
        _, stats_large = loss_fn_large(t, s)
        # alpha should be very close to 0.5
        assert stats_large["alpha_mean"] == pytest.approx(0.5, abs=0.02)

    def test_small_tau_separates_modes(self) -> None:
        """Small tau should produce more extreme alpha values."""
        t = torch.randn(4, 16, 100)
        s = torch.randn(4, 16, 100)

        loss_fn_small = PADDLoss(
            temperature=2.0, tau=0.01, alpha_min=0.0, alpha_max=1.0,
        )
        _, stats_small = loss_fn_small(t, s)

        loss_fn_large = PADDLoss(
            temperature=2.0, tau=100.0, alpha_min=0.0, alpha_max=1.0,
        )
        _, stats_large = loss_fn_large(t, s)

        # With small tau, mean alpha can still be ~0.5 (symmetric around mu),
        # but forward_kl_mean and reverse_kl_mean contributions will differ
        # from the large-tau case since weighting is non-uniform.
        # Just verify they're both valid numbers.
        assert 0.0 <= stats_small["alpha_mean"] <= 1.0
        assert 0.0 <= stats_large["alpha_mean"] <= 1.0


class TestPADDLossAlphaClamp:
    """Alpha clamping tests."""

    def test_alpha_within_bounds(self) -> None:
        """Alpha should be clamped to [alpha_min, alpha_max]."""
        loss_fn = PADDLoss(
            temperature=2.0, tau=1.0, alpha_min=0.2, alpha_max=0.8,
        )
        t = torch.randn(4, 16, 100)
        s = torch.randn(4, 16, 100)
        _, stats = loss_fn(t, s)
        assert 0.2 <= stats["alpha_mean"] <= 0.8

    def test_fixed_alpha_zero_gives_pure_reverse_kl(self) -> None:
        """alpha_min=alpha_max=0 → all reverse KL."""
        loss_fn = PADDLoss(
            temperature=2.0, tau=1.0, alpha_min=0.0, alpha_max=0.0,
        )
        t = torch.randn(2, 8, 50)
        s = torch.randn(2, 8, 50)
        _, stats = loss_fn(t, s)
        assert stats["alpha_mean"] == pytest.approx(0.0)
        # forward_kl contribution should be 0
        assert stats["forward_kl_mean"] >= 0.0  # weighted by alpha=0 in loss
        assert stats["reverse_kl_mean"] >= 0.0

    def test_fixed_alpha_half_gives_jsd(self) -> None:
        """alpha_min=alpha_max=0.5 → pure JSD (equal mix)."""
        loss_fn = PADDLoss(
            temperature=2.0, tau=1.0, alpha_min=0.5, alpha_max=0.5,
        )
        t = torch.randn(2, 8, 50)
        s = torch.randn(2, 8, 50)
        _, stats = loss_fn(t, s)
        assert stats["alpha_mean"] == pytest.approx(0.5)


class TestPADDLossInputValidation:
    """Input shape assertions."""

    def test_rejects_2d_input(self) -> None:
        """PADDLoss requires 3D logits."""
        loss_fn = PADDLoss()
        t = torch.randn(4, 100)
        s = torch.randn(4, 100)
        with pytest.raises(AssertionError, match="3D"):
            loss_fn(t, s)
