"""Tests for :class:`needle_core.domain.interaction.query.Query`."""

from __future__ import annotations

import pytest

from needle_core.domain.interaction.query import Query
from needle_core.domain.metadata.filter_spec import (
    FilterOperator,
    MetadataFilterSpec,
)


def test_basic_query():
    query = Query(query_text="where is the total?")
    assert query.query_text == "where is the total?"
    assert query.top_k == 10
    assert query.query_id.startswith("q_")
    assert query.get_metadata_filter_spec().is_empty()
    assert not query.has_filters()


def test_empty_text_raises():
    with pytest.raises(ValueError):
        Query(query_text="   ")


def test_query_of_with_filters():
    query = Query.of("revenue", top_k=3, filters={"year": 2024, "tags": ["a", "b"]})
    assert query.top_k == 3
    spec = query.get_metadata_filter_spec()
    assert spec.fields() == {"year", "tags"}
    assert query.has_filters()


def test_with_filter_is_immutable():
    base = Query(query_text="hello")
    extended = base.with_filter("year", 2024, FilterOperator.GTE)
    assert base.get_metadata_filter_spec().is_empty()
    assert extended.get_metadata_filter_spec().fields() == {"year"}
    # query identity is preserved across the copy
    assert extended.query_id == base.query_id


def test_dict_metadata_filters_are_coerced():
    query = Query(query_text="hi", metadata_filters={"lang": "en"})
    assert isinstance(query.metadata_filters, MetadataFilterSpec)
    assert query.metadata_filters.fields() == {"lang"}
