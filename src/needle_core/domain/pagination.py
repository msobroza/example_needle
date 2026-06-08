"""Offset/limit pagination value objects.

A :class:`PageRequest` captures the offset and limit a caller asks for; a
:class:`Page` wraps the resulting slice together with the total count so
consumers can compute navigation state (next/prev, page numbers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PageRequest:
    """A request for a slice of results: ``items[offset : offset + limit]``."""

    offset: int = 0
    limit: int = 10

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("offset must be >= 0")
        if self.limit < 1:
            raise ValueError("limit must be >= 1")


@dataclass
class Page:
    """A materialised page of ``items`` plus the ``total`` available count."""

    items: list[Any]
    total: int
    request: PageRequest

    @property
    def has_next(self) -> bool:
        """Whether there are more items beyond this page."""
        return self.request.offset + len(self.items) < self.total

    @property
    def has_prev(self) -> bool:
        """Whether there is a page before this one."""
        return self.request.offset > 0

    @property
    def num_pages(self) -> int:
        """Total number of pages given the request's limit."""
        if self.total <= 0:
            return 0
        return (self.total + self.request.limit - 1) // self.request.limit

    @property
    def page_number(self) -> int:
        """1-indexed position of this page within the full result set."""
        return self.request.offset // self.request.limit + 1
