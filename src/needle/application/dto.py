"""Request/response data-transfer objects for the application services."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

from needle_core.domain.interaction.query import Query

from ..retrieval.data import InputDocument, PreannotationPageResult


@dataclass
class IndexRequest:
    """A request to index a batch of documents."""

    documents: Sequence[InputDocument]
    reinit: bool = True


@dataclass
class IndexResult:
    """The outcome of an indexing operation."""

    num_documents: int
    num_pages: int


@dataclass
class SearchRequest:
    """A request to search the index."""

    query: Query
    top_k: int = 10


@dataclass
class SearchResult:
    """The outcome of a search, including how long it took."""

    query: Query
    hits: list[PreannotationPageResult] = field(default_factory=list)
    took_ms: float = 0.0

    def __len__(self) -> int:
        return len(self.hits)

    @property
    def is_empty(self) -> bool:
        return not self.hits

    @property
    def best(self) -> Optional[PreannotationPageResult]:
        return self.hits[0] if self.hits else None


__all__ = ["IndexRequest", "IndexResult", "SearchRequest", "SearchResult"]
