"""
tests/test_seed.py
------------------
Unit tests for the reproducibility seed utility.
"""

import random
import numpy as np
from src.utils.seed import set_seed


def test_python_random_reproducible():
    set_seed(42)
    a = random.random()
    set_seed(42)
    b = random.random()
    assert a == b


def test_numpy_random_reproducible():
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    assert np.allclose(a, b)

