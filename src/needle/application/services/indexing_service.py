"""Indexing use-case service."""

from __future__ import annotations

from conversational_core.domain.ports.retriever import RetrieverPort

from ..dto import IndexRequest, IndexResult


class IndexingService:
    """Index documents through any :class:`RetrieverPort` implementation."""

    def __init__(self, retriever: RetrieverPort) -> None:
        self.retriever = retriever

    def execute(self, request: IndexRequest) -> IndexResult:
        self.retriever.index(list(request.documents), reinit=request.reinit)
        return IndexResult(
            num_documents=len(request.documents),
            num_pages=len(self.retriever),
        )


__all__ = ["IndexingService"]
