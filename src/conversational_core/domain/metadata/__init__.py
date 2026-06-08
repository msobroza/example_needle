"""Metadata filtering: specs and backend adapters."""

from __future__ import annotations

from .backend_filter_adapters import BackendFilterAdapter, MultiFieldFilterAdapter
from .filter_spec import FieldFilter, FilterOperator, MetadataFilterSpec

__all__ = [
    "FieldFilter",
    "FilterOperator",
    "MetadataFilterSpec",
    "BackendFilterAdapter",
    "MultiFieldFilterAdapter",
]
