"""Tests for the metrics subpackage (evaluation + timing)."""

from __future__ import annotations

import math
import time

import pytest

from needle.metrics.evaluation import (
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
from needle.metrics.timing import Timer, timed

# ---------------------------------------------------------------------------
# Worked example: ranked = [1, 2, 3, 4], relevant = {2, 4}
# ---------------------------------------------------------------------------
RANKED = [1, 2, 3, 4]
RELEVANT = {2, 4}


def test_recall_at_k():
    # Top-2 = [1, 2]; 1 of 2 relevant retrieved -> 0.5
    assert recall_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.5)
    # Full list retrieves both relevant items.
    assert recall_at_k(RANKED, RELEVANT, 4) == pytest.approx(1.0)


def test_precision_at_k():
    # Top-2 = [1, 2]; 1 relevant out of 2 -> 0.5
    assert precision_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.5)
    # Top-4 = [1, 2, 3, 4]; 2 relevant out of 4 -> 0.5
    assert precision_at_k(RANKED, RELEVANT, 4) == pytest.approx(0.5)


def test_hit_rate_at_k():
    assert hit_rate_at_k(RANKED, RELEVANT, 2) == 1.0
    # Top-1 = [1]; no relevant item -> 0.0
    assert hit_rate_at_k(RANKED, RELEVANT, 1) == 0.0


def test_reciprocal_rank():
    # First relevant item (2) is at rank 2 -> 1/2.
    assert reciprocal_rank(RANKED, RELEVANT) == pytest.approx(0.5)
    # No relevant item present -> 0.0
    assert reciprocal_rank([5, 6, 7], RELEVANT) == 0.0


def test_average_precision():
    # Relevant hits at ranks 2 (p=1/2) and 4 (p=2/4); AP = (0.5 + 0.5) / 2.
    assert average_precision(RANKED, RELEVANT) == pytest.approx(0.5)


def test_average_precision_perfect():
    # Both relevant items ranked first -> AP = 1.0
    assert average_precision([2, 4, 1, 3], RELEVANT) == pytest.approx(1.0)


def test_empty_relevant_returns_zero():
    assert recall_at_k(RANKED, set(), 2) == 0.0
    assert average_precision(RANKED, set()) == 0.0
    assert ndcg_at_k(RANKED, set(), 2) == 0.0
    assert reciprocal_rank(RANKED, set()) == 0.0


def test_dcg_at_k_known_value():
    # Relevant at ranks 2 and 4: 1/log2(3) + 1/log2(5).
    expected = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    assert dcg_at_k(RANKED, RELEVANT, 4) == pytest.approx(expected)


def test_ndcg_in_unit_interval():
    value = ndcg_at_k(RANKED, RELEVANT, 4)
    assert 0.0 <= value <= 1.0


def test_ndcg_perfect_when_top_k_are_relevant_in_order():
    # Top-2 are exactly the relevant items -> NDCG@2 == 1.0
    assert ndcg_at_k([2, 4, 1, 3], RELEVANT, 2) == pytest.approx(1.0)


def test_mean_reciprocal_rank_two_queries():
    rankeds = [[1, 2, 3, 4], [3, 1, 2, 4]]
    relevants = [{2, 4}, {1}]
    # RR: query 1 -> 1/2, query 2 (first relevant 1 at rank 2) -> 1/2.
    assert mean_reciprocal_rank(rankeds, relevants) == pytest.approx(0.5)


def test_mean_average_precision_two_queries():
    rankeds = [[1, 2, 3, 4], [1, 2, 3, 4]]
    relevants = [{2, 4}, {1}]
    # AP: query 1 -> 0.5, query 2 (relevant 1 at rank 1) -> 1.0; MAP = 0.75.
    assert mean_average_precision(rankeds, relevants) == pytest.approx(0.75)


def test_empty_query_sets_return_zero():
    assert mean_reciprocal_rank([], []) == 0.0
    assert mean_average_precision([], []) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_k_less_than_one_raises(k):
    with pytest.raises(ValueError):
        recall_at_k(RANKED, RELEVANT, k)
    with pytest.raises(ValueError):
        precision_at_k(RANKED, RELEVANT, k)
    with pytest.raises(ValueError):
        hit_rate_at_k(RANKED, RELEVANT, k)
    with pytest.raises(ValueError):
        dcg_at_k(RANKED, RELEVANT, k)
    with pytest.raises(ValueError):
        ndcg_at_k(RANKED, RELEVANT, k)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([[1, 2]], [{1}, {2}])
    with pytest.raises(ValueError):
        mean_average_precision([[1, 2]], [{1}, {2}])


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def test_timer_measures_sleep():
    with Timer() as timer:
        time.sleep(0.02)
    assert timer.elapsed > 0.0
    # elapsed_ms must track elapsed * 1000 closely.
    assert timer.elapsed_ms == pytest.approx(timer.elapsed * 1000.0, rel=1e-6)


def test_timer_before_start_raises():
    timer = Timer()
    with pytest.raises(RuntimeError):
        _ = timer.elapsed


def test_timed_decorator_records_duration():
    @timed
    def slow_add(a: int, b: int) -> int:
        time.sleep(0.01)
        return a + b

    assert slow_add.last_seconds is None
    assert slow_add(2, 3) == 5
    assert slow_add.last_seconds is not None
    assert slow_add.last_seconds > 0.0
