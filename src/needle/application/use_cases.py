"""Thin functional wrappers over the application services."""

from __future__ import annotations

from collections.abc import Sequence

from needle_core.domain.interaction.query import Query
from needle_core.domain.ports.retriever import RetrieverPort

from ..retrieval.data import InputDocument
from .dto import IndexRequest, IndexResult, SearchRequest, SearchResult
from .services.indexing_service import IndexingService
from .services.search_service import SearchService


def index_documents(
    retriever: RetrieverPort,
    documents: Sequence[InputDocument],
    *,
    reinit: bool = True,
) -> IndexResult:
    """Index ``documents`` with ``retriever`` and return a summary."""
    service = IndexingService(retriever)
    return service.execute(IndexRequest(documents=list(documents), reinit=reinit))


def search(retriever: RetrieverPort, query: Query, *, top_k: int = 10) -> SearchResult:
    """Search ``retriever`` for ``query`` and return timed results."""
    service = SearchService(retriever)
    return service.execute(SearchRequest(query=query, top_k=top_k))


__all__ = ["index_documents", "search"]
