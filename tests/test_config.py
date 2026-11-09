"""
tests/test_config.py
--------------------
Unit tests for the configuration loader.
Compatible with both pytest and python -m unittest.
"""

from pathlib import Path
import unittest
from src.utils.config import load_config, get_project_root


class TestConfig(unittest.TestCase):
    def test_load_config_returns_dict(self):
        cfg = load_config()
        self.assertIsInstance(cfg, dict)

    def test_config_has_required_top_level_keys(self):
        cfg = load_config()
        for key in ("project", "paths", "dataset", "model", "training", "inference", "app"):
            self.assertIn(key, cfg, f"Missing top-level key '{key}' in config.yaml")

    def test_config_paths_resolved(self):
        cfg = load_config()
        for key, value in cfg["paths"].items():
            self.assertTrue(Path(value).is_absolute(), f"Path '{key}' should be absolute after resolution")

    def test_config_seed_is_int(self):
        cfg = load_config()
        self.assertIsInstance(cfg["project"]["seed"], int)

    def test_config_model_channels_positive(self):
        cfg = load_config()
        for channel_key in ("in_channels", "hidden_channels", "out_channels"):
            self.assertGreater(cfg["model"][channel_key], 0)

    def test_missing_config_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_config("nonexistent_config.yaml")


if __name__ == "__main__":
    unittest.main()
