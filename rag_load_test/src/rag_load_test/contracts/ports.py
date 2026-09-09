"""Hexagonal ports. Adapters implement these; the workflow depends only on them.

Every method is ``async`` so the FastAPI event loop is never blocked: CPU-bound
adapters run their sync work in a bounded thread pool, HTTP adapters await httpx.
``ready()`` returns ``(ok, detail)`` and feeds ``GET /readyz``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .models import Passage, ScoredPassage


@runtime_checkable
class EmbedderPort(Protocol):
    @property
    def model_name(self) -> str: ...

    async def embed_query(self, text: str) -> list[float]: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def ready(self) -> tuple[bool, str]: ...


@runtime_checkable
class RerankerPort(Protocol):
    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]: ...

    async def ready(self) -> tuple[bool, str]: ...


@runtime_checkable
class VectorStorePort(Protocol):
    async def upsert(
        self, passages: Sequence[Passage], embeddings: Sequence[Sequence[float]]
    ) -> None: ...

    async def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[ScoredPassage]: ...

    async def count(self) -> int: ...

    def embedder_model(self) -> str | None:
        """Name of the embedder the collection was built with, if recorded."""
        ...


@runtime_checkable
class ChatModelPort(Protocol):
    @property
    def model_name(self) -> str: ...

    async def generate(self, messages: list[dict[str, str]]) -> str: ...
