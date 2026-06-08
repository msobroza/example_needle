"""Base index store — two parallel lists of embeddings and payloads.

:class:`BaseIndexStore` implements the in-memory bookkeeping shared by every
concrete adapter: appending items, enumerating them, clearing, and reporting
length. Persistence (:meth:`save` / :meth:`load`) is intentionally left
abstract so each subclass can pick its own on-disk representation.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import Any

import numpy as np

from conversational_core.domain.ports.index_store import IndexStorePort

from ..types import Embedding, Payload


class BaseIndexStore(IndexStorePort):
    """Concrete in-memory backbone for :class:`IndexStorePort` adapters.

    Items are kept in two parallel lists, ``_embeddings`` and ``_payloads``,
    indexed positionally: the ``i``-th embedding belongs with the ``i``-th
    payload. Subclasses only need to supply :meth:`save` and :meth:`load`.
    """

    def __init__(self) -> None:
        self._embeddings: list[Embedding] = []
        self._payloads: list[Payload] = []

    # ---------- Mutation ----------

    def add(self, embedding: np.ndarray, payload: dict[str, Any]) -> None:
        """Append one embedding and its associated payload."""
        self._embeddings.append(embedding)
        self._payloads.append(payload)

    def extend(
        self,
        embeddings: Iterable[np.ndarray],
        payloads: Iterable[dict[str, Any]],
    ) -> None:
        """Append several ``(embedding, payload)`` pairs at once.

        ``embeddings`` and ``payloads`` are zipped pairwise; any surplus in
        the longer iterable is silently ignored, mirroring :func:`zip`.
        """
        for embedding, payload in zip(embeddings, payloads, strict=False):
            self.add(embedding, payload)

    def clear(self) -> None:
        """Remove every stored item."""
        self._embeddings.clear()
        self._payloads.clear()

    # ---------- Access ----------

    def embeddings(self) -> list[np.ndarray]:
        """Return all stored embeddings in insertion order."""
        return self._embeddings

    def payloads(self) -> list[dict[str, Any]]:
        """Return all stored payloads in insertion order."""
        return self._payloads

    # ---------- Persistence (subclass responsibility) ----------

    @abstractmethod
    def save(self, path: Any) -> None:
        """Persist the store to ``path``."""

    @abstractmethod
    def load(self, path: Any) -> BaseIndexStore:
        """Load the store from ``path`` (returns ``self``)."""

    # ---------- Dunders ----------

    def __len__(self) -> int:
        """Number of stored items."""
        return len(self._payloads)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} items={len(self)}>"
