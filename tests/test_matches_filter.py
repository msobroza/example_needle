"""Tests for the read side of the filter contract + visualiser fallback."""

from __future__ import annotations

import pytest

from needle.retrieval.page_retriever_utils import (
    SimilarityMapVisualizer,
    matches_filter,
)


def test_none_criterion_always_matches():
    assert matches_filter("anything", None) is True
    assert matches_filter(None, None) is True


def test_scalar_equality():
    assert matches_filter("en", "en") is True
    assert matches_filter("en", "fr") is False


def test_list_membership():
    assert matches_filter("en", ["en", "fr"]) is True
    assert matches_filter("de", ["en", "fr"]) is False


@pytest.mark.parametrize(
    "criterion,value,expected",
    [
        ({"$eq": 5}, 5, True),
        ({"$ne": 5}, 6, True),
        ({"$in": [1, 2]}, 2, True),
        ({"$nin": [1, 2]}, 3, True),
        ({"$gt": 10}, 11, True),
        ({"$gt": 10}, 10, False),
        ({"$gte": 10}, 10, True),
        ({"$lt": 10}, 9, True),
        ({"$lte": 10}, 10, True),
        ({"$contains": "needle"}, "find the needle here", True),
        ({"$exists": True}, "x", True),
        ({"$exists": False}, None, True),
        ({"$regex": r"\d{4}"}, "year 2024", True),
        ({"$gte": 2020, "$lte": 2024}, 2022, True),
        ({"$gte": 2020, "$lte": 2024}, 2030, False),
    ],
)
def test_mongo_operators(criterion, value, expected):
    assert matches_filter(value, criterion) is expected


def test_comparison_with_none_actual_is_false():
    assert matches_filter(None, {"$gte": 1}) is False


def test_contains_on_non_container_is_false():
    assert matches_filter(123, {"$contains": "x"}) is False


def test_unsupported_operator_raises():
    with pytest.raises(ValueError):
        matches_filter(1, {"$bogus": 1})


def test_similarity_map_visualizer_falls_back_to_image():
    sentinel = object()
    # No matplotlib/colpali/IPython in the test env -> returns the image as-is.
    result = SimilarityMapVisualizer.highlight_image(
        image=sentinel, query_text="q", model=None, processor=None, device="cpu"
    )
    assert result is sentinel
