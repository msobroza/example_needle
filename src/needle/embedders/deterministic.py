"""A deterministic, weight-free embedder.

Embeddings are derived from a hash of the input bytes, so the same image or
query always maps to the same vector — no model, no network, no GPU. Handy for
tests, examples and pipeline micro-benchmarks.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import BaseEmbedder


class DeterministicEmbedder(BaseEmbedder):
    """Reproducible embedder backed by hashed pseudo-random unit vectors."""

    def __init__(
        self,
        *,
        dim: int = 32,
        num_tokens: int = 8,
        multi_vector: bool = True,
    ) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if num_tokens < 1:
            raise ValueError("num_tokens must be >= 1")
        self.dim = dim
        self.num_tokens = num_tokens
        self.multi_vector = multi_vector

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _rng(key: bytes) -> np.random.Generator:
        digest = hashlib.sha256(key).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def _unit_vectors(self, rng: np.random.Generator, n: int) -> np.ndarray:
        vecs = rng.standard_normal((n, self.dim)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return vecs

    @staticmethod
    def _image_key(image: Any) -> bytes:
        try:
            width, height = image.size
            return f"img:{width}x{height}:".encode() + image.tobytes()[:8192]
        except Exception:  # pragma: no cover - non-PIL fallback
            return ("img:" + repr(image)).encode()

    # -- Embedder port --------------------------------------------------
    def embed_images(self, images: Sequence[Any]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for image in images:
            rng = self._rng(self._image_key(image))
            if self.multi_vector:
                out.append(self._unit_vectors(rng, self.num_tokens).astype(np.float16))
            else:
                out.append(self._unit_vectors(rng, 1)[0].astype(np.float16))
        return out

    def embed_query(self, query: str) -> np.ndarray:
        rng = self._rng(("query:" + query).encode())
        if self.multi_vector:
            return self._unit_vectors(rng, max(2, len(query.split())))
        return self._unit_vectors(rng, 1)[0]
