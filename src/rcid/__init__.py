"""RCID: Residual Causal Imprint Distillation."""

__version__ = "0.1.0"

import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
