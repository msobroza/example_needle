"""Pluggable scoring strategies for query/document embeddings.

The formulas here mirror :meth:`needle.retrieval.page_retrievers.
MultimodalEmbedderRetriever._score` and ``_minmax`` exactly so that the
retrievers and any standalone re-ranking stay numerically consistent:

* **MaxSim** (late interaction): ``(query @ document.T).max(axis=1).sum()``;
* **cosine** (dense): ``q · d / (||q|| ||d|| + EPS)``;
* **min-max**: rescale a score vector to ``[0, 1]`` (zeros if constant).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from ..constants import EPS


class ScoringStrategy(ABC):
    """Abstract base for query/document similarity scorers."""

    @abstractmethod
    def score(self, query: np.ndarray, document: np.ndarray) -> float:
        """Return the similarity between a single ``query`` and ``document``."""

    def score_all(
        self, query: np.ndarray, documents: Sequence[np.ndarray]
    ) -> np.ndarray:
        """Score ``query`` against every document, returning a 1-D array."""
        return np.array(
            [self.score(query, document) for document in documents],
            dtype=np.float64,
        )


class MaxSimScorer(ScoringStrategy):
    """Late-interaction MaxSim scorer for multi-vector embeddings."""

    def score(self, query: np.ndarray, document: np.ndarray) -> float:
        return float((query @ document.T).max(axis=1).sum())


class CosineScorer(ScoringStrategy):
    """Dense cosine-similarity scorer with an ``EPS`` denominator floor."""

    def score(self, query: np.ndarray, document: np.ndarray) -> float:
        denom = np.linalg.norm(query) * np.linalg.norm(document) + EPS
        return float(query @ document / denom)


def get_scorer(multi_vector: bool) -> ScoringStrategy:
    """Return the scorer matching the embedding type.

    ``multi_vector=True`` selects :class:`MaxSimScorer`; ``False`` selects
    :class:`CosineScorer`.
    """
    return MaxSimScorer() if multi_vector else CosineScorer()


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Rescale ``scores`` to ``[0, 1]``; return zeros if the range is ~0.

    Mirrors ``BasePageRetriever._minmax``.
    """
    scores = np.asarray(scores)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)
