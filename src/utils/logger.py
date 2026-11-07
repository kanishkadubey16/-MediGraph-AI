"""
src/utils/logger.py
-------------------
Centralised logging setup for MediGraph AI.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Loaded %d nodes", n)
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_LEVEL  = logging.INFO

# Registry so we don't add duplicate handlers on repeated calls
_configured_loggers: set[str] = set()


def get_logger(
    name: str,
    level: Optional[int] = None,
    fmt: str = _DEFAULT_FORMAT,
) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Parameters
    ----------
    name : str
        Logger name — pass ``__name__`` from the calling module.
    level : int, optional
        Logging level (e.g. ``logging.DEBUG``).
        Defaults to INFO (or the value set by ``configure_root_logger``).
    fmt : str
        Log format string.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if name not in _configured_loggers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.propagate = False
        _configured_loggers.add(name)

    if level is not None:
        logger.setLevel(level)
    elif logger.level == logging.NOTSET:
        logger.setLevel(_DEFAULT_LEVEL)

    return logger


def configure_root_logger(level: int = _DEFAULT_LEVEL, fmt: str = _DEFAULT_FORMAT) -> None:
    """
    Configure the root logger.
    Call this once in a training or inference entry-point script.
    """
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout, force=True)

