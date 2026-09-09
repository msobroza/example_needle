"""In-process sentence-transformers embedder (the ``local`` embedder backend)."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Executor
from typing import Any

from .executor import run_blocking


class SentenceTransformerEmbedder:
    """EmbedderPort backed by a ``SentenceTransformer`` model run in a thread pool.

    The model object is injected so tests can pass a fake exposing ``encode``;
    ``from_pretrained`` builds the real one (CPU only, weights loaded once).

    Example::

        embedder = SentenceTransformerEmbedder.from_pretrained(
            "BAAI/bge-small-en-v1.5", executor, batch_size=64
        )
        vector = await embedder.embed_query("vacation policy")
    """

    def __init__(
        self,
        model: Any,
        executor: Executor,
        *,
        model_name: str,
        batch_size: int = 64,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
        self._model = model
        self._executor = executor
        self._model_name = model_name
        self._batch_size = batch_size

    @classmethod
    def from_pretrained(
        cls, model_name: str, executor: Executor, *, batch_size: int = 64
    ) -> SentenceTransformerEmbedder:
        """Load ``model_name`` from the HF cache/hub on CPU and wrap it."""
        # Imported here: sentence_transformers pulls in torch, which must not be
        # paid by ``import rag_load_test`` or by the OVMS/fake backends.
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device="cpu")
        return cls(model, executor, model_name=model_name, batch_size=batch_size)

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        batch = list(texts)
        if not batch:
            return []
        return await run_blocking(self._executor, self._encode, batch)

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    async def ready(self) -> tuple[bool, str]:
        return True, self._model_name

    def _encode(self, texts: list[str]) -> list[list[float]]:
        # Unit-norm vectors so Chroma's cosine distance equals 1 - dot product.
        matrix = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [row.tolist() for row in matrix]
