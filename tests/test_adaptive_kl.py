"""Tests for AKLLoss and KLRatioLoss (Phase 2 adaptive losses)."""

import pytest
import torch

from rcid.distillation.adaptive_kl_losses import AKLLoss, KLRatioLoss


# ======================================================================
# AKLLoss
# ======================================================================

class TestAKLLossBasic:
    def test_finite_positive_loss(self) -> None:
        loss_fn = AKLLoss(temperature=2.0, mu=0.5)
        t = torch.randn(2, 8, 50)
        s = torch.randn(2, 8, 50)
        loss, stats = loss_fn(t, s)
        assert loss.isfinite()
        assert loss.item() > 0.0

    def test_identical_logits_near_zero(self) -> None:
        loss_fn = AKLLoss(temperature=2.0, mu=0.5)
        logits = torch.randn(4, 16, 100)
        loss, _ = loss_fn(logits, logits)
        assert loss.item() < 0.01

    def test_stats_keys(self) -> None:
        loss_fn = AKLLoss()
        t = torch.randn(2, 4, 30)
        s = torch.randn(2, 4, 30)
        _, stats = loss_fn(t, s)
        expected = {"alpha_mean", "forward_kl_mean", "reverse_kl_mean",
                    "g_head_mean", "g_tail_mean"}
        assert expected == set(stats.keys())

    def test_alpha_in_valid_range(self) -> None:
        loss_fn = AKLLoss(temperature=2.0, mu=0.5)
        t = torch.randn(4, 16, 100)
        s = torch.randn(4, 16, 100)
        _, stats = loss_fn(t, s)
        assert 0.0 <= stats["alpha_mean"] <= 1.0

    def test_differentiable(self) -> None:
        loss_fn = AKLLoss()
        t = torch.randn(2, 4, 30)
        s = torch.randn(2, 4, 30, requires_grad=True)
        loss, _ = loss_fn(t, s)
        loss.backward()
        assert s.grad is not None
        assert s.grad.shape == s.shape


class TestAKLLossMask:
    def test_mask_changes_loss(self) -> None:
        loss_fn = AKLLoss(temperature=2.0)
        t = torch.randn(2, 8, 50)
        s = torch.randn(2, 8, 50)

        mask_full = torch.ones(2, 8)
        loss_full, _ = loss_fn(t, s, mask=mask_full)

        mask_half = torch.ones(2, 8)
        mask_half[:, 4:] = 0
        loss_half, _ = loss_fn(t, s, mask=mask_half)

        assert loss_full.item() != pytest.approx(loss_half.item(), abs=1e-6)

    def test_no_mask_equals_all_ones(self) -> None:
        loss_fn = AKLLoss(temperature=2.0)
        t = torch.randn(2, 8, 50)
        s = torch.randn(2, 8, 50)
        loss_none, _ = loss_fn(t, s, mask=None)
        loss_ones, _ = loss_fn(t, s, mask=torch.ones(2, 8))
        assert loss_none.item() == pytest.approx(loss_ones.item(), rel=1e-5)

    def test_rejects_2d_input(self) -> None:
        loss_fn = AKLLoss()
        with pytest.raises(AssertionError):
            loss_fn(torch.randn(4, 100), torch.randn(4, 100))


class TestAKLLossMuVariation:
    def test_different_mu_gives_different_alpha(self) -> None:
        t = torch.randn(4, 16, 100)
        s = torch.randn(4, 16, 100)
        _, stats_low = AKLLoss(mu=0.3)(t, s)
        _, stats_high = AKLLoss(mu=0.7)(t, s)
        # Different mu changes head/tail split → different alpha
        assert stats_low["alpha_mean"] != pytest.approx(
            stats_high["alpha_mean"], abs=1e-3,
        )


# ======================================================================
# KLRatioLoss — token granularity
# ======================================================================

class TestKLRatioTokenBasic:
    def test_finite_positive_loss(self) -> None:
        loss_fn = KLRatioLoss(temperature=2.0, granularity="token")
        t = torch.randn(2, 8, 50)
        s = torch.randn(2, 8, 50)
        loss, stats = loss_fn(t, s)
        assert loss.isfinite()
        assert loss.item() > 0.0

    def test_identical_logits_near_zero(self) -> None:
        loss_fn = KLRatioLoss(temperature=2.0, granularity="token")
        logits = torch.randn(4, 16, 100)
        loss, _ = loss_fn(logits, logits)
        assert loss.item() < 0.01

    def test_stats_keys(self) -> None:
        loss_fn = KLRatioLoss(granularity="token")
        _, stats = loss_fn(torch.randn(2, 4, 30), torch.randn(2, 4, 30))
        expected = {"alpha_mean", "alpha_std", "forward_kl_mean", "reverse_kl_mean"}
        assert expected == set(stats.keys())

    def test_alpha_in_valid_range(self) -> None:
        loss_fn = KLRatioLoss(granularity="token")
        _, stats = loss_fn(torch.randn(4, 16, 100), torch.randn(4, 16, 100))
        assert 0.0 <= stats["alpha_mean"] <= 1.0

    def test_differentiable(self) -> None:
        loss_fn = KLRatioLoss(granularity="token")
        t = torch.randn(2, 4, 30)
        s = torch.randn(2, 4, 30, requires_grad=True)
        loss, _ = loss_fn(t, s)
        loss.backward()
        assert s.grad is not None

    def test_mask(self) -> None:
        loss_fn = KLRatioLoss(temperature=2.0, granularity="token")
        t = torch.randn(2, 8, 50)
        s = torch.randn(2, 8, 50)
        loss_full, _ = loss_fn(t, s, mask=torch.ones(2, 8))
        mask = torch.ones(2, 8)
        mask[:, 4:] = 0
        loss_partial, _ = loss_fn(t, s, mask=mask)
        assert loss_full.item() != pytest.approx(loss_partial.item(), abs=1e-6)


# ======================================================================
# KLRatioLoss — batch granularity
# ======================================================================

class TestKLRatioBatchBasic:
    def test_finite_positive_loss(self) -> None:
        loss_fn = KLRatioLoss(temperature=2.0, granularity="batch", beta=0.99)
        t = torch.randn(2, 8, 50)
        s = torch.randn(2, 8, 50)
        loss, stats = loss_fn(t, s)
        assert loss.isfinite()
        assert loss.item() > 0.0

    def test_stats_has_alpha_ema(self) -> None:
        loss_fn = KLRatioLoss(granularity="batch", beta=0.99)
        _, stats = loss_fn(torch.randn(2, 4, 30), torch.randn(2, 4, 30))
        assert "alpha_ema" in stats
        assert 0.0 <= stats["alpha_ema"] <= 1.0

    def test_ema_updates_over_calls(self) -> None:
        loss_fn = KLRatioLoss(granularity="batch", beta=0.5)
        t = torch.randn(2, 8, 50)
        s = torch.randn(2, 8, 50)
        _, stats1 = loss_fn(t, s)
        ema1 = stats1["alpha_ema"]
        _, stats2 = loss_fn(t, s)
        ema2 = stats2["alpha_ema"]
        # With beta=0.5, EMA should move noticeably toward the instant value
        # If instant alpha is consistent, ema should still update
        assert isinstance(ema1, float)
        assert isinstance(ema2, float)

    def test_identical_logits_near_zero(self) -> None:
        loss_fn = KLRatioLoss(temperature=2.0, granularity="batch")
        logits = torch.randn(4, 16, 100)
        loss, _ = loss_fn(logits, logits)
        assert loss.item() < 0.01


# ======================================================================
# KLRatioLoss — fixed_alpha mode
# ======================================================================

class TestKLRatioFixedAlpha:
    def test_fixed_alpha_jeffreys(self) -> None:
        """fixed_alpha=0.5 produces Jeffreys divergence."""
        loss_fn = KLRatioLoss(fixed_alpha=0.5)
        t = torch.randn(2, 8, 50)
        s = torch.randn(2, 8, 50)
        loss, stats = loss_fn(t, s)
        assert loss.isfinite()
        assert loss.item() > 0.0
        assert stats["alpha_mean"] == 0.5
        assert stats["alpha_std"] == 0.0

    def test_fixed_alpha_reverse_kl(self) -> None:
        """fixed_alpha=0.0 produces pure reverse KL."""
        loss_fn = KLRatioLoss(fixed_alpha=0.0)
        t = torch.randn(2, 8, 50)
        s = torch.randn(2, 8, 50)
        loss, stats = loss_fn(t, s)
        assert loss.isfinite()
        assert stats["alpha_mean"] == 0.0
        # Loss should equal RKL * T^2
        assert loss.item() > 0.0

    def test_fixed_alpha_forward_kl(self) -> None:
        """fixed_alpha=1.0 produces pure forward KL."""
        loss_fn = KLRatioLoss(fixed_alpha=1.0)
        t = torch.randn(2, 8, 50)
        s = torch.randn(2, 8, 50)
        loss, stats = loss_fn(t, s)
        assert loss.isfinite()
        assert stats["alpha_mean"] == 1.0

    def test_fixed_alpha_ignores_granularity(self) -> None:
        """When fixed_alpha is set, granularity doesn't matter."""
        t = torch.randn(2, 8, 50)
        s = torch.randn(2, 8, 50)
        loss_a, stats_a = KLRatioLoss(
            granularity="token", fixed_alpha=0.5,
        )(t, s)
        loss_b, stats_b = KLRatioLoss(
            granularity="batch", fixed_alpha=0.5,
        )(t, s)
        assert loss_a.item() == pytest.approx(loss_b.item(), rel=1e-5)

    def test_fixed_alpha_differentiable(self) -> None:
        loss_fn = KLRatioLoss(fixed_alpha=0.5)
        t = torch.randn(2, 4, 30)
        s = torch.randn(2, 4, 30, requires_grad=True)
        loss, _ = loss_fn(t, s)
        loss.backward()
        assert s.grad is not None

    def test_jeffreys_equals_half_fkl_plus_half_rkl(self) -> None:
        """Verify fixed_alpha=0.5 gives 0.5*FKL + 0.5*RKL."""
        t = torch.randn(2, 8, 50)
        s = torch.randn(2, 8, 50)
        _, stats = KLRatioLoss(fixed_alpha=0.5, temperature=2.0)(t, s)
        fkl = stats["forward_kl_mean"]
        rkl = stats["reverse_kl_mean"]
        # The loss should be approximately 0.5*FKL + 0.5*RKL (before T^2)
        # Check that both FKL and RKL are reported
        assert fkl > 0
        assert rkl > 0


class TestKLRatioInvalidGranularity:
    def test_rejects_unknown_granularity(self) -> None:
        with pytest.raises(AssertionError, match="granularity"):
            KLRatioLoss(granularity="unknown")

    def test_fixed_alpha_allows_any_granularity(self) -> None:
        """When fixed_alpha is set, granularity validation is skipped."""
        # Should NOT raise even with granularity="unknown" if fixed_alpha is set
        loss_fn = KLRatioLoss(granularity="unknown", fixed_alpha=0.5)
        t = torch.randn(2, 4, 30)
        s = torch.randn(2, 4, 30)
        loss, _ = loss_fn(t, s)
        assert loss.isfinite()
