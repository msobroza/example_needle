"""Deterministic, dependency-free doubles for EmbedderPort/RerankerPort/VectorStorePort.

``FakeEmbedder`` is a hashed bag-of-words embedder, so texts sharing words get
similar vectors and retrieval over the synthetic corpus is meaningful in tests.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from ..contracts.models import Passage, ScoredPassage

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _token_vector(token: str, dim: int) -> list[float]:
    digest = hashlib.sha256(token.encode()).digest()
    raw = [((digest[i % len(digest)] / 255.0) * 2.0 - 1.0) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b, strict=False)))


class FakeEmbedder:
    def __init__(self, dim: int = 16, name: str = "fake-embedder") -> None:
        self.dim = dim
        self._name = name

    @property
    def model_name(self) -> str:
        return self._name

    def embed_text(self, text: str) -> list[float]:
        acc = [0.0] * self.dim
        for token in tokenize(text) or ["<empty>"]:
            for i, value in enumerate(_token_vector(token, self.dim)):
                acc[i] += value
        return _normalise(acc)

    async def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

    async def ready(self) -> tuple[bool, str]:
        return True, self._name


class FakeReranker:
    """Scores by Jaccard overlap between query and passage tokens (0..1)."""

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        query_tokens = set(tokenize(query))
        scored = [
            p.model_copy(update={"rerank_score": _jaccard(query_tokens, p.text)})
            for p in passages
        ]
        scored.sort(key=lambda p: p.rerank_score or 0.0, reverse=True)
        return scored[:top_n]

    async def ready(self) -> tuple[bool, str]:
        return True, "fake-reranker"


def _jaccard(query_tokens: set[str], text: str) -> float:
    passage_tokens = set(tokenize(text))
    union = query_tokens | passage_tokens
    if not union:
        return 0.0
    return len(query_tokens & passage_tokens) / len(union)


class InMemoryVectorStore:
    def __init__(self, embedder_model: str | None = "fake-embedder") -> None:
        self._rows: dict[str, tuple[Passage, list[float]]] = {}
        self._embedder_model = embedder_model

    async def upsert(
        self, passages: Sequence[Passage], embeddings: Sequence[Sequence[float]]
    ) -> None:
        if len(passages) != len(embeddings):
            raise ValueError(
                "passages/embeddings length mismatch: "
                f"{len(passages)} != {len(embeddings)}"
            )
        for passage, embedding in zip(passages, embeddings, strict=True):
            self._rows[passage.id] = (passage, list(embedding))

    async def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[ScoredPassage]:
        scored = [
            ScoredPassage(**p.model_dump(), retrieval_score=cosine(embedding, vec))
            for p, vec in self._rows.values()
        ]
        scored.sort(key=lambda p: p.retrieval_score, reverse=True)
        return scored[:top_k]

    async def count(self) -> int:
        return len(self._rows)

    def embedder_model(self) -> str | None:
        return self._embedder_model
