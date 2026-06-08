"""Search use-case service."""

from __future__ import annotations

from conversational_core.domain.ports.retriever import RetrieverPort

from ...metrics.timing import Timer
from ..dto import SearchRequest, SearchResult
from ..errors import EmptyQueryError


class SearchService:
    """Execute a search and report how long it took."""

    def __init__(self, retriever: RetrieverPort) -> None:
        self.retriever = retriever

    def execute(self, request: SearchRequest) -> SearchResult:
        if not request.query.query_text.strip():
            raise EmptyQueryError("query_text must not be empty")
        with Timer() as timer:
            hits = self.retriever.search(request.query, top_k=request.top_k)
        return SearchResult(
            query=request.query, hits=list(hits), took_ms=timer.elapsed_ms
        )


__all__ = ["SearchService"]
