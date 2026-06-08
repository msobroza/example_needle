"""Scoring subpackage — pluggable query/document similarity strategies.

These mirror the formulas used by the retrievers (MaxSim for multi-vector
late interaction, cosine for dense) so scoring stays consistent whether it
runs inside a retriever or standalone.
"""

from __future__ import annotations

from .strategies import (
    CosineScorer,
    MaxSimScorer,
    ScoringStrategy,
    get_scorer,
    minmax_normalize,
)

__all__ = [
    "ScoringStrategy",
    "MaxSimScorer",
    "CosineScorer",
    "get_scorer",
    "minmax_normalize",
]
