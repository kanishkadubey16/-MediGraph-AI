"""
src/utils/config.py
-------------------
Central configuration loader for MediGraph AI.

Usage:
    from src.utils.config import load_config
    cfg = load_config()
    hidden_dim = cfg["model"]["hidden_channels"]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# Project root is two levels above this file: src/utils/ → src/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path = _DEFAULT_CONFIG) -> dict[str, Any]:
    """
    Load and return the YAML configuration file as a nested dict.

    Parameters
    ----------
    path : str or Path
        Path to the config YAML file. Defaults to <project_root>/config.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the given path.
    yaml.YAMLError
        If the file is not valid YAML.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"[MediGraph] Configuration file not found at '{config_path}'. "
            f"Expected config.yaml in the project root: {_PROJECT_ROOT}"
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    if cfg is None:
        raise ValueError(
            f"[MediGraph] Configuration file '{config_path}' is empty."
        )

    # Resolve path values relative to the project root
    if "paths" in cfg:
        for key, value in cfg["paths"].items():
            if isinstance(value, str) and not os.path.isabs(value):
                cfg["paths"][key] = str(_PROJECT_ROOT / value)

    return cfg


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT

