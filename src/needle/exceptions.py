"""Exceptions raised by the ``needle`` infrastructure layer.

These complement :mod:`needle_core.domain.exceptions` (domain errors):
anything here is about *infrastructure* — stores, embedders, configuration.
"""

from __future__ import annotations


class NeedleError(Exception):
    """Base class for all ``needle`` infrastructure errors."""


class IndexStoreError(NeedleError):
    """Raised when an index store cannot satisfy an operation."""


class EmbeddingError(NeedleError):
    """Raised when embedding images or a query fails."""


class ConfigurationError(NeedleError):
    """Raised when a configuration object is invalid or inconsistent."""


class RenderingError(NeedleError):
    """Raised when a document cannot be rasterised to page images."""


__all__ = [
    "NeedleError",
    "IndexStoreError",
    "EmbeddingError",
    "ConfigurationError",
    "RenderingError",
]
