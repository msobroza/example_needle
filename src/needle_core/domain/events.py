"""Domain events.

Lightweight records describing things that have happened in the domain
(documents indexed, queries executed, index cleared). They carry an ISO-8601
``occurred_at`` timestamp and expose a stable ``event_type`` string for
routing/serialisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DomainEvent:
    """Base class for all domain events."""

    occurred_at: str = field(default_factory=_now_iso)

    @property
    def event_type(self) -> str:
        """A stable identifier for the concrete event class."""
        return type(self).__name__


@dataclass
class DocumentIndexed(DomainEvent):
    """A document and its pages were added to the index."""

    document_id: str = ""
    pages: int = 0


@dataclass
class QueryExecuted(DomainEvent):
    """A query was run against the index."""

    query_text: str = ""
    num_results: int = 0
    took_ms: float = 0.0


@dataclass
class IndexCleared(DomainEvent):
    """The index was cleared, removing ``num_removed`` entries."""

    num_removed: int = 0
