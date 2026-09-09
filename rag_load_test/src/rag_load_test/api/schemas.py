"""Request/response bodies for the HTTP API (spec section 8).

Example::

    response = to_query_response(result, request_id="abc", deployment="monolith")
    response.timings_ms  # keys: embed, retrieve, rerank, generate, total
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..contracts.models import MetadataValue, RagMode, RagResult


class QueryRequest(BaseModel):
    """Body of ``POST /query`` and ``POST /retrieve``."""

    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=100)
    rerank_top_k: int | None = Field(None, ge=1, le=50)


class PassageOut(BaseModel):
    """One retrieved (and possibly reranked) passage as returned to the client."""

    id: str
    text: str
    retrieval_score: float
    rerank_score: float | None
    metadata: dict[str, MetadataValue]


class QueryResponse(BaseModel):
    """Body of a successful ``/query`` or ``/retrieve`` call."""

    request_id: str
    deployment: str
    mode: RagMode
    answer: str | None
    passages: list[PassageOut]
    timings_ms: dict[str, float]


class ReadyCheck(BaseModel):
    """One ``/readyz`` probe result."""

    name: str
    ok: bool
    detail: str


class ReadyResponse(BaseModel):
    """Body of ``GET /readyz`` (HTTP 200 when ``ready``, 503 otherwise)."""

    status: Literal["ready", "not_ready"]
    checks: list[ReadyCheck]


def to_query_response(
    result: RagResult, *, request_id: str, deployment: str
) -> QueryResponse:
    """Fold a workflow ``RagResult`` into the wire shape.

    Example::

        body = to_query_response(await run_rag(graph, "q?"), request_id="r1",
                                 deployment="monolith")
    """
    return QueryResponse(
        request_id=request_id,
        deployment=deployment,
        mode=result.mode,
        answer=result.answer,
        passages=[PassageOut(**passage.model_dump()) for passage in result.passages],
        timings_ms=result.timings.as_dict(),
    )
