"""Application layer: use-cases that orchestrate the retrieval components.

Services here depend on the abstract
:class:`~needle_core.domain.ports.retriever.RetrieverPort`, so they work
with either a concrete model-backed retriever or a torch-free
:class:`~needle.pipeline.RetrievalPipeline`.
"""

from __future__ import annotations

from .dto import IndexRequest, IndexResult, SearchRequest, SearchResult
from .errors import ApplicationError, EmptyQueryError
from .services.indexing_service import IndexingService
from .services.ranking_service import RankingService
from .services.search_service import SearchService
from .use_cases import index_documents, search

__all__ = [
    "IndexRequest",
    "IndexResult",
    "SearchRequest",
    "SearchResult",
    "ApplicationError",
    "EmptyQueryError",
    "IndexingService",
    "SearchService",
    "RankingService",
    "index_documents",
    "search",
]
