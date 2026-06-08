"""Volatile in-memory index store — fast, ephemeral, non-persistent."""

from __future__ import annotations

from typing import Any

from ..exceptions import IndexStoreError
from .base import BaseIndexStore


class InMemoryIndexStore(BaseIndexStore):
    """An index store that lives only for the lifetime of the process.

    Useful for tests, scratch indexing, and pipelines that rebuild their
    index on every run. It deliberately rejects persistence: there is no
    on-disk representation to ``save`` to or ``load`` from.
    """

    def save(self, path: Any) -> None:
        """Always raise — this store cannot be persisted."""
        raise IndexStoreError("InMemoryIndexStore does not support persistence")

    def load(self, path: Any) -> InMemoryIndexStore:
        """Always raise — this store cannot be loaded from disk."""
        raise IndexStoreError("InMemoryIndexStore does not support persistence")
