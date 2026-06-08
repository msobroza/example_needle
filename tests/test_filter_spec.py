"""Tests for the metadata filter specification."""

from __future__ import annotations

import pytest

from needle_core.domain.metadata.filter_spec import (
    FieldFilter,
    FilterOperator,
    MetadataFilterSpec,
)


def test_field_filter_coerces_operator_string():
    ff = FieldFilter("year", "gte", 2020)
    assert ff.operator is FilterOperator.GTE


def test_field_filter_requires_field():
    with pytest.raises(ValueError):
        FieldFilter("", FilterOperator.EQ, 1)


def test_spec_add_is_chainable():
    spec = (
        MetadataFilterSpec()
        .add("year", FilterOperator.GTE, 2020)
        .add("lang", "in", ["en", "fr"])
    )
    assert len(spec) == 2
    assert bool(spec) is True
    assert spec.fields() == {"year", "lang"}
    assert [f.field for f in spec] == ["year", "lang"]


def test_empty_spec():
    spec = MetadataFilterSpec()
    assert spec.is_empty()
    assert not spec
    assert len(spec) == 0


def test_from_dict_maps_lists_to_in():
    spec = MetadataFilterSpec.from_dict({"year": 2024, "tags": ["a", "b"]})
    by_field = {f.field: f for f in spec}
    assert by_field["year"].operator is FilterOperator.EQ
    assert by_field["tags"].operator is FilterOperator.IN
    assert by_field["tags"].value == ["a", "b"]


def test_from_dict_none():
    assert MetadataFilterSpec.from_dict(None).is_empty()


def test_add_accepts_field_filter_instance():
    spec = MetadataFilterSpec().add(FieldFilter("a", FilterOperator.NE, 1))
    assert spec.filters[0].operator is FilterOperator.NE
