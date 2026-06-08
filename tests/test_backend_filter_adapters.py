"""Tests for backend filter adapters."""

from __future__ import annotations

import pytest

from conversational_core.domain.metadata.backend_filter_adapters import (
    BackendFilterAdapter,
    MultiFieldFilterAdapter,
)
from conversational_core.domain.metadata.filter_spec import (
    FilterOperator,
    MetadataFilterSpec,
)


def test_to_backend_none_is_empty():
    assert MultiFieldFilterAdapter().to_backend(None) == {}
    assert MultiFieldFilterAdapter().to_backend() == {}


def test_to_backend_basic_operators():
    spec = (
        MetadataFilterSpec()
        .add("lang", FilterOperator.EQ, "en")
        .add("tags", FilterOperator.IN, ["a", "b"])
    )
    backend = MultiFieldFilterAdapter().to_backend(spec=spec)
    assert backend == {"lang": {"$eq": "en"}, "tags": {"$in": ["a", "b"]}}


def test_to_backend_merges_range_on_same_field():
    spec = (
        MetadataFilterSpec()
        .add("year", FilterOperator.GTE, 2020)
        .add("year", FilterOperator.LTE, 2024)
    )
    backend = MultiFieldFilterAdapter().to_backend(spec=spec)
    assert backend == {"year": {"$gte": 2020, "$lte": 2024}}


def test_to_backend_accepts_plain_dict():
    backend = MultiFieldFilterAdapter().to_backend({"lang": "en"})
    assert backend == {"lang": {"$eq": "en"}}


def test_backend_filter_adapter_is_abstract():
    with pytest.raises(TypeError):
        BackendFilterAdapter()  # type: ignore[abstract]
