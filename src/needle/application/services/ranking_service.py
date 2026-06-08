"""Ranking utilities: score normalisation and rank fusion.

Useful when combining results from several retrievers (e.g. a multi-vector and
a dense backend) into a single ranking via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional

import numpy as np


class RankingService:
    """Pure ranking helpers (no I/O, no model)."""

    @staticmethod
    def minmax(scores: Sequence[float]) -> list[float]:
        """Min-max normalise scores to ``[0, 1]`` (zeros if constant)."""
        arr = np.asarray(list(scores), dtype=float)
        if arr.size == 0:
            return []
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-9:
            return [0.0] * arr.size
        return ((arr - lo) / (hi - lo)).tolist()

    @staticmethod
    def reciprocal_rank_fusion(
        result_lists: Sequence[Sequence[Any]],
        key: Callable[[Any], Any],
        *,
        k: int = 60,
    ) -> list[tuple[Any, float]]:
        """Fuse several ranked lists into one ``(key, score)`` ranking via RRF."""
        if k < 1:
            raise ValueError("k must be >= 1")
        scores: dict[Any, float] = {}
        for results in result_lists:
            for rank, item in enumerate(results, start=1):
                identity = key(item)
                scores[identity] = scores.get(identity, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    @classmethod
    def fuse(
        cls,
        result_lists: Sequence[Sequence[Any]],
        key: Callable[[Any], Any],
        *,
        k: int = 60,
        top_k: Optional[int] = None,
    ) -> list[Any]:
        """Return the fused identities (best first), optionally truncated."""
        ranked = cls.reciprocal_rank_fusion(result_lists, key, k=k)
        identities = [identity for identity, _ in ranked]
        return identities[:top_k] if top_k is not None else identities


__all__ = ["RankingService"]
