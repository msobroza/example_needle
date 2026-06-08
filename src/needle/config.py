"""Configuration objects for building retrievers.

A :class:`RetrieverConfig` bundles the knobs needed to construct a retriever
and can be populated from the environment, making it easy to drive the library
from a CLI, a service, or a notebook without hard-coding values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from .constants import DEFAULT_BATCH_SIZE, DEFAULT_DPI, DEFAULT_INDEX_PATH
from .exceptions import ConfigurationError


@dataclass
class RetrieverConfig:
    """Everything required to build and run a retriever backend."""

    name: str = "colqwen2"
    index_path: str = DEFAULT_INDEX_PATH
    dpi: int = DEFAULT_DPI
    batch_size: int = DEFAULT_BATCH_SIZE
    model_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigurationError("RetrieverConfig.name must be non-empty")
        if self.dpi <= 0:
            raise ConfigurationError(f"dpi must be positive, got {self.dpi}")
        if self.batch_size <= 0:
            raise ConfigurationError(
                f"batch_size must be positive, got {self.batch_size}"
            )

    def to_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``get_retriever(name, **kwargs)``."""
        kwargs: dict[str, Any] = {
            "index_path": self.index_path,
            "dpi": self.dpi,
            "batch_size": self.batch_size,
        }
        if self.model_name:
            kwargs["model_name"] = self.model_name
        return kwargs

    @classmethod
    def from_env(cls, prefix: str = "NEEDLE_") -> RetrieverConfig:
        """Build a config from ``{PREFIX}RETRIEVER`` / ``INDEX_PATH`` / ... vars."""

        def _get(key: str, default: str) -> str:
            return os.environ.get(f"{prefix}{key}", default)

        return cls(
            name=_get("RETRIEVER", cls.name),
            index_path=_get("INDEX_PATH", cls.index_path),
            dpi=int(_get("DPI", str(cls.dpi))),
            batch_size=int(_get("BATCH_SIZE", str(cls.batch_size))),
            model_name=os.environ.get(f"{prefix}MODEL_NAME") or None,
        )


__all__ = ["RetrieverConfig"]
