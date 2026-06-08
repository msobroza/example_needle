"""Conversation value objects.

A :class:`Turn` pairs a user :class:`Query` with an optional response. A
:class:`Conversation` is an ordered sequence of turns, providing convenient
access to the latest turn and a flattened textual history.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Optional

from .query import Query


@dataclass
class Turn:
    """A single exchange: a user query and its (optional) response."""

    query: Query
    response: Optional[Any] = None


@dataclass
class Conversation:
    """An ordered sequence of :class:`Turn` objects."""

    turns: list[Turn] = field(default_factory=list)

    def add_turn(self, query: Query, response: Optional[Any] = None) -> Turn:
        """Append a new turn and return it."""
        turn = Turn(query=query, response=response)
        self.turns.append(turn)
        return turn

    @property
    def last(self) -> Optional[Turn]:
        """The most recent turn, or ``None`` when the conversation is empty."""
        return self.turns[-1] if self.turns else None

    def history_text(self) -> str:
        """Return the concatenated query texts, one per line, in order."""
        return "\n".join(turn.query.query_text for turn in self.turns)

    def __len__(self) -> int:
        return len(self.turns)

    def __iter__(self) -> Iterator[Turn]:
        return iter(self.turns)
