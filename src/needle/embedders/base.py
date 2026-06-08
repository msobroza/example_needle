"""The :class:`BaseEmbedder` — a concrete base for embedding backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np


class BaseEmbedder(ABC):
    """Base class for embedders satisfying the domain ``Embedder`` port.

    Subclasses implement :meth:`embed_images` and :meth:`embed_query` and set
    the :attr:`multi_vector` class attribute (``True`` for late-interaction /
    MaxSim models, ``False`` for dense single-vector models).
    """

    #: Whether this embedder produces multi-vector (token-level) embeddings.
    multi_vector: bool = True

    @abstractmethod
    def embed_images(self, images: Sequence[Any]) -> list[np.ndarray]:
        """Embed a batch of page images (one array per image)."""

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query string into an embedding array."""

    @property
    def regime(self) -> str:
        """Human-readable scoring regime for this embedder."""
        return "multi-vector (MaxSim)" if self.multi_vector else "dense (cosine)"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} multi_vector={self.multi_vector}>"
