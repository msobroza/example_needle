"""The :class:`Embedder` port.

An embedder turns page images and query text into vectors. The ``multi_vector``
flag tells consumers whether to score with MaxSim (late interaction) or cosine.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Structural interface implemented by every embedding backend."""

    #: ``True`` for late-interaction (multi-vector) models, ``False`` for dense.
    multi_vector: bool

    def embed_images(self, images: Sequence[Any]) -> list[np.ndarray]:
        """Embed a batch of page images.

        Returns one array per image: shape ``(num_tokens, dim)`` when
        ``multi_vector`` is ``True``, else ``(dim,)``.
        """
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query string into a ``(num_tokens, dim)`` or ``(dim,)`` array."""
        ...
