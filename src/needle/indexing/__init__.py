"""Index-store adapters for ``needle``.

Concrete implementations of
:class:`conversational_core.domain.ports.index_store.IndexStorePort`,
each storing ``(embedding, payload)`` pairs with a different persistence
strategy:

* :class:`InMemoryIndexStore` — volatile, no persistence;
* :class:`PickleIndexStore` — legacy-compatible pickle dict;
* :class:`NumpyIndexStore` — ``.npz`` archive, good for large arrays.

All share the in-memory backbone provided by :class:`BaseIndexStore`.
"""

from __future__ import annotations

from .base import BaseIndexStore
from .memory_store import InMemoryIndexStore
from .numpy_store import NumpyIndexStore
from .pickle_store import PickleIndexStore

__all__ = [
    "BaseIndexStore",
    "InMemoryIndexStore",
    "PickleIndexStore",
    "NumpyIndexStore",
]
