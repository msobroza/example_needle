"""Lightweight test/demo doubles for the retrievers.

:class:`DummyEmbedderRetriever` is a fully-working
:class:`~needle.retrieval.page_retrievers.MultimodalEmbedderRetriever` whose
"model" produces *deterministic* embeddings derived from the input bytes — no
weights, no network, no GPU. It exercises the entire index/search/persist
pipeline so the real retrieval mechanics can be tested in CI, and it powers the
runnable example in ``examples/``.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .retrieval.page_retrievers import MultimodalEmbedderRetriever


class DummyEmbedderRetriever(MultimodalEmbedderRetriever):
    """A deterministic, dependency-free embedder for tests and demos."""

    def __init__(
        self,
        *,
        dim: int = 32,
        num_tokens: int = 8,
        multi_vector: bool = True,
        **kwargs: Any,
    ) -> None:
        self.dim = dim
        self.num_tokens = num_tokens
        # Instance attribute shadows the class attribute read by ``_score``.
        self.multi_vector = multi_vector
        super().__init__(**kwargs)

    # -- model lifecycle -------------------------------------------------
    def _load_model(self) -> None:
        # No real model — but set the attributes the base class expects so
        # ``show_page`` and ``__repr__`` behave.
        self.model = None
        self.processor = None

    # -- deterministic embedding helpers --------------------------------
    @staticmethod
    def _rng(key: bytes) -> np.random.Generator:
        digest = hashlib.sha256(key).digest()
        seed = int.from_bytes(digest[:8], "little")
        return np.random.default_rng(seed)

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

    # -- the three abstract hooks ---------------------------------------
    def _embed_images(self, images: list) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for image in images:
            rng = self._rng(self._image_key(image))
            if self.multi_vector:
                out.append(self._unit_vectors(rng, self.num_tokens).astype(np.float16))
            else:
                out.append(self._unit_vectors(rng, 1)[0].astype(np.float16))
        return out

    def _embed_query(self, query: str) -> np.ndarray:
        rng = self._rng(("query:" + query).encode())
        if self.multi_vector:
            tokens = max(2, len(query.split()))
            return self._unit_vectors(rng, tokens)
        return self._unit_vectors(rng, 1)[0]
