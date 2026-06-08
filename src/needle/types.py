"""Shared type aliases used across the ``needle`` package.

Centralising these keeps signatures readable and gives a single place to
document the embedding conventions:

* a **vector** is a 1-D array ``(dim,)`` — one dense embedding;
* a **multi-vector** is a 2-D array ``(num_tokens, dim)`` — a set of token
  embeddings used by late-interaction (MaxSim) scoring.
"""

from __future__ import annotations

import os
from typing import Any, Union

import numpy as np

#: A single dense embedding, shape ``(dim,)``.
Vector = np.ndarray

#: A set of token embeddings, shape ``(num_tokens, dim)``.
MultiVector = np.ndarray

#: Either a dense vector or a multi-vector.
Embedding = np.ndarray

#: Anything acceptable as a filesystem path.
PathLike = Union[str, os.PathLike]

#: The per-page bookkeeping dict stored alongside each embedding.
Payload = dict[str, Any]

__all__ = [
    "Vector",
    "MultiVector",
    "Embedding",
    "PathLike",
    "Payload",
]
