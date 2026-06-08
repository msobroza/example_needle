"""Information-retrieval metrics over ranked lists.

Each metric takes a ``ranked`` sequence of item ids (most relevant first) and
a ``relevant`` collection of the ids considered relevant for the query. The
relevance model is binary: an id is either relevant or it is not.

The helpers fall into three families:

* cut-off metrics (``recall_at_k``, ``precision_at_k``, ``hit_rate_at_k``,
  ``dcg_at_k``, ``ndcg_at_k``) — evaluated over the top ``k`` results;
* rank metrics (``reciprocal_rank``, ``average_precision``) — evaluated over
  the full ranking;
* their means over several queries (``mean_reciprocal_rank``,
  ``mean_average_precision``).
"""

from __future__ import annotations

import math
from collections.abc import Collection, Hashable, Sequence

ItemId = Hashable


def _check_k(k: int) -> None:
    """Validate that a cut-off ``k`` is a positive integer."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def recall_at_k(
    ranked: Sequence[ItemId],
    relevant: Collection[ItemId],
    k: int,
) -> float:
    """Fraction of relevant items retrieved within the top ``k``.

    Returns ``0.0`` when there are no relevant items.
    """
    _check_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(relevant_set)


def precision_at_k(
    ranked: Sequence[ItemId],
    relevant: Collection[ItemId],
    k: int,
) -> float:
    """Fraction of the top ``k`` results that are relevant.

    The denominator is ``k`` (not the number of items actually retrieved),
    matching the standard precision@k definition.
    """
    _check_k(k)
    relevant_set = set(relevant)
    top_k = ranked[:k]
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def hit_rate_at_k(
    ranked: Sequence[ItemId],
    relevant: Collection[ItemId],
    k: int,
) -> float:
    """``1.0`` if any relevant item appears in the top ``k``, else ``0.0``."""
    _check_k(k)
    relevant_set = set(relevant)
    return 1.0 if any(item in relevant_set for item in ranked[:k]) else 0.0


def reciprocal_rank(
    ranked: Sequence[ItemId],
    relevant: Collection[ItemId],
) -> float:
    """Reciprocal of the 1-based rank of the first relevant item.

    Returns ``0.0`` when no relevant item is present in ``ranked``.
    """
    relevant_set = set(relevant)
    for index, item in enumerate(ranked, start=1):
        if item in relevant_set:
            return 1.0 / index
    return 0.0


def mean_reciprocal_rank(
    rankeds: Sequence[Sequence[ItemId]],
    relevants: Sequence[Collection[ItemId]],
) -> float:
    """Mean of :func:`reciprocal_rank` over several queries.

    Returns ``0.0`` for an empty set of queries.
    """
    if len(rankeds) != len(relevants):
        raise ValueError(
            f"rankeds and relevants length mismatch: "
            f"{len(rankeds)} != {len(relevants)}"
        )
    if not rankeds:
        return 0.0
    return sum(
        reciprocal_rank(ranked, relevant)
        for ranked, relevant in zip(rankeds, relevants, strict=True)
    ) / len(rankeds)


def average_precision(
    ranked: Sequence[ItemId],
    relevant: Collection[ItemId],
) -> float:
    """Average precision (AP) for a single ranking.

    AP is the mean of the precision values computed at each rank where a
    relevant item is found, averaged over the total number of relevant items.
    Returns ``0.0`` when there are no relevant items.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for index, item in enumerate(ranked, start=1):
        if item in relevant_set:
            hits += 1
            precision_sum += hits / index
    return precision_sum / len(relevant_set)


def mean_average_precision(
    rankeds: Sequence[Sequence[ItemId]],
    relevants: Sequence[Collection[ItemId]],
) -> float:
    """Mean of :func:`average_precision` over several queries (MAP).

    Returns ``0.0`` for an empty set of queries.
    """
    if len(rankeds) != len(relevants):
        raise ValueError(
            f"rankeds and relevants length mismatch: "
            f"{len(rankeds)} != {len(relevants)}"
        )
    if not rankeds:
        return 0.0
    return sum(
        average_precision(ranked, relevant)
        for ranked, relevant in zip(rankeds, relevants, strict=True)
    ) / len(rankeds)


def dcg_at_k(
    ranked: Sequence[ItemId],
    relevant: Collection[ItemId],
    k: int,
) -> float:
    """Discounted cumulative gain at ``k`` with binary relevance.

    Uses the standard ``gain / log2(rank + 1)`` discount where gain is ``1``
    for a relevant item and ``0`` otherwise.
    """
    _check_k(k)
    relevant_set = set(relevant)
    gain = 0.0
    for index, item in enumerate(ranked[:k], start=1):
        if item in relevant_set:
            gain += 1.0 / math.log2(index + 1)
    return gain


def ndcg_at_k(
    ranked: Sequence[ItemId],
    relevant: Collection[ItemId],
    k: int,
) -> float:
    """Normalised DCG at ``k`` with binary relevance.

    The DCG of ``ranked`` is divided by the ideal DCG (IDCG), i.e. the DCG of a
    ranking that places as many relevant items as possible at the top. The
    result lies in ``[0, 1]`` and is ``1.0`` when the top ``k`` are exactly the
    relevant items. Returns ``0.0`` when there are no relevant items.
    """
    _check_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(ranked, relevant_set, k) / idcg
