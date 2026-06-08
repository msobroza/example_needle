"""The retrieval response value object.

A :class:`RetrievalResponse` bundles the :class:`Query` that was executed
with its ranked results and the wall-clock time the retrieval took.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .query import Query


@dataclass
class RetrievalResponse:
    """The result of executing a :class:`Query` against a retriever."""

    query: Query
    results: list[Any] = field(default_factory=list)
    took_ms: float = 0.0

    def __len__(self) -> int:
        return len(self.results)

    @property
    def is_empty(self) -> bool:
        """Whether the response contains no results."""
        return not self.results

    @property
    def best(self) -> Optional[Any]:
        """The top result, or ``None`` when the response is empty."""
        return self.results[0] if self.results else None
