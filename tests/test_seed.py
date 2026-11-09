"""
tests/test_seed.py
------------------
Unit tests for the reproducibility seed utility.
Compatible with both pytest and python -m unittest.
"""

import random
import unittest
import numpy as np
from src.utils.seed import set_seed


class TestSeed(unittest.TestCase):
    def test_python_random_reproducible(self):
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        self.assertEqual(a, b)

    def test_numpy_random_reproducible(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_allclose(a, b)


if __name__ == "__main__":
    unittest.main()
