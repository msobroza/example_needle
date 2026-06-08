"""The :class:`IndexStorePort` — persistence/lookup of embeddings + payloads."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class IndexStorePort(ABC):
    """Abstract store holding ``(embedding, payload)`` pairs.

    Concrete adapters live in :mod:`needle.indexing`. The contract is
    intentionally tiny: append items, enumerate them, clear, and persist.
    """

    @abstractmethod
    def add(self, embedding: np.ndarray, payload: dict[str, Any]) -> None:
        """Append one embedding and its associated payload."""

    @abstractmethod
    def embeddings(self) -> list[np.ndarray]:
        """Return all stored embeddings in insertion order."""

    @abstractmethod
    def payloads(self) -> list[dict[str, Any]]:
        """Return all stored payloads in insertion order."""

    @abstractmethod
    def clear(self) -> None:
        """Remove every stored item."""

    @abstractmethod
    def save(self, path: Any) -> None:
        """Persist the store to ``path``."""

    @abstractmethod
    def load(self, path: Any) -> IndexStorePort:
        """Load the store from ``path`` (returns ``self``)."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of stored items."""
