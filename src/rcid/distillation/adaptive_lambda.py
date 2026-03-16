"""Adaptive lambda controller driven by causal sensitivity monitoring.

Increases OCID regularisation when monitoring detects poor alignment,
decreases it when alignment is satisfactory.

Usage::

    controller = AdaptiveLambdaController(lambda_init=0.1)
    # After each monitoring step:
    new_lambda = controller.update(health=0.45)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class AdaptiveLambdaController:
    """PID-style adaptive lambda based on causal sensitivity health.

    target_health: desired overall cosine similarity (e.g. 0.7).
    When health < target -> increase lambda (need more correction).
    When health > target -> decrease lambda (alignment is fine).

    Uses EMA smoothing to avoid oscillation.

    Args:
        lambda_init: Starting lambda value.
        lambda_min: Floor.
        lambda_max: Ceiling.
        target_health: Target overall_health from monitor.
        gain: How aggressively lambda responds to health gap.
        ema_beta: Smoothing for lambda updates.
    """

    def __init__(
        self,
        lambda_init: float = 0.1,
        lambda_min: float = 0.01,
        lambda_max: float = 1.0,
        target_health: float = 0.7,
        gain: float = 0.5,
        ema_beta: float = 0.9,
    ) -> None:
        self.lambda_val = lambda_init
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.target = target_health
        self.gain = gain
        self.beta = ema_beta

    def update(self, health: float) -> float:
        """Update lambda based on latest health score.

        Returns new lambda value.
        """
        # Error: positive means we're below target (need more regularisation)
        error = self.target - health

        # Proportional adjustment
        adjustment = 1.0 + self.gain * error
        # error > 0 (health below target) -> adjustment > 1 -> increase lambda
        # error < 0 (health above target) -> adjustment < 1 -> decrease lambda

        raw = self.lambda_val * adjustment

        # EMA smooth
        self.lambda_val = self.beta * self.lambda_val + (1 - self.beta) * raw

        # Clamp
        self.lambda_val = max(self.lambda_min, min(self.lambda_max, self.lambda_val))

        return self.lambda_val

    def get_lambda(self) -> float:
        """Return the current lambda value."""
        return self.lambda_val
