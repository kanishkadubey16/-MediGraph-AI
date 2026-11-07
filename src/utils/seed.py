"""
src/utils/seed.py
-----------------
Reproducibility helpers for MediGraph AI.

Usage:
    from src.utils.seed import set_seed
    set_seed(42)
"""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Parameters
    ----------
    seed : int
        The random seed. Use the value from config.yaml ``project.seed``.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Deterministic ops (may reduce performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed — skip

