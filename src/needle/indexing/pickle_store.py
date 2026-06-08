"""Pickle-backed index store — compatible with the legacy retriever format.

The on-disk layout is the exact dict produced by the existing page
retrievers in :mod:`needle.retrieval.page_retrievers`::

    {"embeddings": [...], "payloads": [...]}

so an index written by either side can be read by the other.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from ..types import PathLike
from .base import BaseIndexStore


class PickleIndexStore(BaseIndexStore):
    """Persist the store as a single pickled ``{"embeddings", "payloads"}`` dict.

    Pickle preserves arbitrary Python objects (including ragged lists of
    multi-vector arrays and rich payload values), which keeps this adapter
    drop-in compatible with the retrievers' ``save_index`` / ``load_index``.
    The trade-off is the usual one: pickle is Python-specific and unsafe to
    load from untrusted sources.
    """

    def save(self, path: PathLike) -> None:
        """Pickle the embeddings and payloads to ``path``."""
        data = {"embeddings": self._embeddings, "payloads": self._payloads}
        with open(Path(path), "wb") as fh:
            pickle.dump(data, fh)

    def load(self, path: PathLike) -> PickleIndexStore:
        """Restore the store from a pickle written by :meth:`save`."""
        with open(Path(path), "rb") as fh:
            data: dict[str, Any] = pickle.load(fh)
        self._embeddings = list(data["embeddings"])
        self._payloads = list(data["payloads"])
        return self
