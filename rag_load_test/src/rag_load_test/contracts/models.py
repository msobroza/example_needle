"""Value objects exchanged between the workflow, the adapters and the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MetadataValue = str | int | float | bool
RagMode = Literal["query", "retrieve"]


class Passage(BaseModel):
    """A chunk of text stored in the vector store."""

    id: str
    text: str
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class ScoredPassage(Passage):
    """A passage returned by retrieval, optionally re-scored by the reranker."""

    retrieval_score: float
    rerank_score: float | None = None


class StageTimings(BaseModel):
    """Wall-clock milliseconds spent in each workflow stage."""

    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        """Return ``{"embed": ..., "retrieve": ..., ...}`` (keys without ``_ms``)."""
        return {
            "embed": self.embed_ms,
            "retrieve": self.retrieve_ms,
            "rerank": self.rerank_ms,
            "generate": self.generate_ms,
            "total": self.total_ms,
        }


class RagResult(BaseModel):
    """Outcome of one workflow run."""

    question: str
    mode: RagMode
    answer: str | None
    passages: list[ScoredPassage]
    timings: StageTimings
