"""Domain layer: documents, interactions and metadata filtering."""

from __future__ import annotations

from .document.document import Document, DocumentPage, DocumentVersion
from .interaction.query import Query
from .metadata.backend_filter_adapters import (
    BackendFilterAdapter,
    MultiFieldFilterAdapter,
)
from .metadata.filter_spec import FieldFilter, FilterOperator, MetadataFilterSpec
from .types import DocumentExtension

__all__ = [
    "Document",
    "DocumentVersion",
    "DocumentPage",
    "Query",
    "DocumentExtension",
    "FieldFilter",
    "FilterOperator",
    "MetadataFilterSpec",
    "BackendFilterAdapter",
    "MultiFieldFilterAdapter",
]
