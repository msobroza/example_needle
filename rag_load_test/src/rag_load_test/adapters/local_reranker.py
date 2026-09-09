"""In-process sentence-transformers cross-encoder (the ``local`` reranker backend)."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Executor
from typing import Any

from ..contracts.models import ScoredPassage
from .executor import run_blocking


class CrossEncoderReranker:
    """RerankerPort backed by a ``CrossEncoder`` model run in a thread pool.

    The model object is injected so tests can pass a fake exposing ``predict``;
    ``from_pretrained`` builds the real one (CPU only).

    Example::

        reranker = CrossEncoderReranker.from_pretrained(
            "BAAI/bge-reranker-base", executor
        )
        best = await reranker.rerank("vacation policy", candidates, top_n=5)
    """

    def __init__(
        self,
        model: Any,
        executor: Executor,
        *,
        model_name: str,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
        self._model = model
        self._executor = executor
        self._model_name = model_name
        self._batch_size = batch_size

    @classmethod
    def from_pretrained(
        cls, model_name: str, executor: Executor, *, batch_size: int = 32
    ) -> CrossEncoderReranker:
        """Load ``model_name`` from the HF cache/hub on CPU and wrap it."""
        # Imported here: sentence_transformers pulls in torch, which must not be
        # paid by ``import rag_load_test`` or by the OVMS/fake backends.
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name, device="cpu")
        return cls(model, executor, model_name=model_name, batch_size=batch_size)

    @property
    def model_name(self) -> str:
        return self._model_name

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        candidates = list(passages)
        if not candidates:
            return []
        texts = [p.text for p in candidates]
        scores = await run_blocking(self._executor, self._score, query, texts)
        rescored = [
            p.model_copy(update={"rerank_score": s})
            for p, s in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=_rerank_score, reverse=True)
        return rescored[:top_n]

    async def ready(self) -> tuple[bool, str]:
        return True, self._model_name

    def _score(self, query: str, texts: list[str]) -> list[float]:
        raw = self._model.predict(
            [(query, text) for text in texts], batch_size=self._batch_size
        )
        # numpy scalars are not JSON serialisable; the API echoes these scores.
        return [float(score) for score in raw]


def _rerank_score(passage: ScoredPassage) -> float:
    return passage.rerank_score if passage.rerank_score is not None else 0.0
