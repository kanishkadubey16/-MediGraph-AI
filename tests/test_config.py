"""
tests/test_config.py
--------------------
Unit tests for the configuration loader.
"""

from pathlib import Path
import pytest
from src.utils.config import load_config, get_project_root


def test_load_config_returns_dict():
    cfg = load_config()
    assert isinstance(cfg, dict)


def test_config_has_required_top_level_keys():
    cfg = load_config()
    for key in ("project", "paths", "dataset", "model", "training", "inference", "app"):
        assert key in cfg, f"Missing top-level key '{key}' in config.yaml"


def test_config_paths_resolved():
    cfg = load_config()
    root = get_project_root()
    for key, value in cfg["paths"].items():
        assert Path(value).is_absolute(), f"Path '{key}' should be absolute after resolution"


def test_config_seed_is_int():
    cfg = load_config()
    assert isinstance(cfg["project"]["seed"], int)


def test_config_model_channels_positive():
    cfg = load_config()
    for channel_key in ("in_channels", "hidden_channels", "out_channels"):
        assert cfg["model"][channel_key] > 0


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")

