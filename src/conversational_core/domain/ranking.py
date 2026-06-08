"""Scoring and ranking value objects.

A :class:`Score` pairs a raw retriever score with an optional normalised
value in ``[0, 1]``. A :class:`RankedPage` ties a score to a concrete page
(and optionally the document/version it belongs to). A :class:`Ranking` is an
ordered collection of ranked pages, best first.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Score:
    """A retriever score, optionally min-max normalised to ``[0, 1]``."""

    raw: float
    normalized: Optional[float] = None

    @property
    def value(self) -> float:
        """The normalised score if present, otherwise the raw score."""
        return self.raw if self.normalized is None else self.normalized


@dataclass(frozen=True)
class RankedPage:
    """A single page paired with its score and optional provenance."""

    page: int
    score: Score
    document: Optional[Any] = None
    version: Optional[Any] = None


@dataclass
class Ranking:
    """An ordered list of :class:`RankedPage`, best (highest) score first."""

    pages: list[RankedPage] = field(default_factory=list)

    def top_k(self, k: int) -> Ranking:
        """Return a new ranking with at most the ``k`` best pages."""
        if k < 0:
            raise ValueError("k must be non-negative")
        return Ranking(pages=self.pages[:k])

    @property
    def best(self) -> Optional[RankedPage]:
        """The highest-scoring page, or ``None`` when the ranking is empty."""
        return self.pages[0] if self.pages else None

    def __iter__(self) -> Iterator[RankedPage]:
        return iter(self.pages)

    def __len__(self) -> int:
        return len(self.pages)

    @classmethod
    def from_scores(
        cls,
        pages: Sequence[int],
        raw_scores: Sequence[float],
    ) -> Ranking:
        """Build a ranking from parallel ``pages``/``raw_scores`` sequences.

        Scores are min-max normalised into ``[0, 1]`` (all zeros when the
        range is constant) and the result is sorted by raw score descending.
        """
        if len(pages) != len(raw_scores):
            raise ValueError("pages and raw_scores must have the same length")
        if not raw_scores:
            return cls(pages=[])

        lo = min(raw_scores)
        hi = max(raw_scores)
        span = hi - lo

        ranked: list[RankedPage] = []
        for page, raw in zip(pages, raw_scores, strict=True):
            normalized = 0.0 if span == 0 else (raw - lo) / span
            score = Score(raw=raw, normalized=normalized)
            ranked.append(RankedPage(page=page, score=score))

        ranked.sort(key=lambda rp: rp.score.raw, reverse=True)
        return cls(pages=ranked)
