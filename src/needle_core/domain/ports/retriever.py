"""The :class:`RetrieverPort` — the high-level index/search contract.

This mirrors the surface of
:class:`needle.retrieval.page_retrievers.BasePageRetriever` so application
services can depend on the abstraction rather than a concrete retriever.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from ..interaction.query import Query


@runtime_checkable
class RetrieverPort(Protocol):
    """Structural interface for anything that can index and search pages."""

    def index(
        self, documents: Sequence[Any], reinit: bool = True, save: bool = True
    ) -> Any:
        """Index a collection of input documents."""
        ...

    def search(self, query: Query, top_k: int = 10) -> list:
        """Return the top-``k`` page results for ``query``."""
        ...

    def __len__(self) -> int:
        """Number of indexed pages."""
        ...
