"""Tests for scoring and ranking value objects."""

from __future__ import annotations

import pytest

from conversational_core.domain.ranking import RankedPage, Ranking, Score


def test_score_value_prefers_normalized() -> None:
    assert Score(raw=3.0).value == 3.0
    assert Score(raw=3.0, normalized=0.5).value == 0.5


def test_from_scores_sorts_descending_by_raw() -> None:
    ranking = Ranking.from_scores([1, 2, 3], [0.1, 0.9, 0.5])
    assert [rp.page for rp in ranking] == [2, 3, 1]


def test_from_scores_min_max_normalises() -> None:
    ranking = Ranking.from_scores([1, 2, 3], [0.0, 5.0, 10.0])
    by_page = {rp.page: rp.score.normalized for rp in ranking}
    assert by_page[1] == pytest.approx(0.0)
    assert by_page[2] == pytest.approx(0.5)
    assert by_page[3] == pytest.approx(1.0)


def test_from_scores_constant_range_yields_zeros() -> None:
    ranking = Ranking.from_scores([1, 2], [4.0, 4.0])
    assert all(rp.score.normalized == 0.0 for rp in ranking)


def test_from_scores_empty() -> None:
    ranking = Ranking.from_scores([], [])
    assert len(ranking) == 0
    assert ranking.best is None


def test_from_scores_length_mismatch() -> None:
    with pytest.raises(ValueError):
        Ranking.from_scores([1, 2], [0.1])


def test_top_k_and_best() -> None:
    ranking = Ranking.from_scores([1, 2, 3, 4], [0.1, 0.9, 0.5, 0.7])
    assert ranking.best.page == 2
    top2 = ranking.top_k(2)
    assert [rp.page for rp in top2] == [2, 4]
    assert len(top2) == 2


def test_top_k_more_than_available() -> None:
    ranking = Ranking.from_scores([1], [0.5])
    assert len(ranking.top_k(10)) == 1


def test_top_k_negative_rejected() -> None:
    ranking = Ranking.from_scores([1], [0.5])
    with pytest.raises(ValueError):
        ranking.top_k(-1)


def test_ranked_page_provenance() -> None:
    rp = RankedPage(page=1, score=Score(raw=1.0), document="doc", version="ver")
    assert rp.document == "doc"
    assert rp.version == "ver"
