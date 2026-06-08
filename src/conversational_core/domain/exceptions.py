"""Domain-level exceptions raised across the retrieval pipeline."""

from __future__ import annotations


class ConversationalCoreError(Exception):
    """Base class for all domain errors."""


class UnsupportedExtensionError(ConversationalCoreError):
    """Raised when a document extension has no registered extractor."""


class EmptyIndexError(ConversationalCoreError):
    """Raised when a search is attempted against an empty index."""


class IndexPersistenceError(ConversationalCoreError):
    """Raised when an index cannot be saved to or loaded from disk."""
