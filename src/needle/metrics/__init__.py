"""Metrics subpackage — retrieval evaluation and timing helpers.

The :mod:`evaluation` module implements binary-relevance information-retrieval
metrics over ranked id lists; :mod:`timing` provides a :class:`Timer` context
manager and a :func:`timed` decorator for wall-clock measurements.
"""

from __future__ import annotations

from .evaluation import (
    average_precision,
    dcg_at_k,
    hit_rate_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .timing import Timer, timed

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "hit_rate_at_k",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "average_precision",
    "mean_average_precision",
    "dcg_at_k",
    "ndcg_at_k",
    "Timer",
    "timed",
]
