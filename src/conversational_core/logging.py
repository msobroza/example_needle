"""Logging helpers shared across the project.

A thin wrapper over the stdlib ``logging`` module so applications get a sane
default format without each entry point re-implementing it.
"""

from __future__ import annotations

import logging
from typing import Optional

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(level: str | int = "INFO", *, fmt: Optional[str] = None) -> None:
    """Configure the root logger with a consistent format.

    Parameters
    ----------
    level:
        Logging level name (``"INFO"``) or numeric level.
    fmt:
        Optional log format string; defaults to :data:`DEFAULT_FORMAT`.
    """
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    logging.basicConfig(level=level, format=fmt or DEFAULT_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger (``logging.getLogger`` shorthand)."""
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger", "DEFAULT_FORMAT"]
