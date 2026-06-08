"""The :class:`Query` value object.

A query carries the natural-language text the user typed plus an optional
metadata filter (e.g. ``language == "en"`` and ``year >= 2020``). Retrievers
read ``query_text`` for embedding and call
:meth:`Query.get_metadata_filter_spec` to obtain the structured filter, which
is then translated to a backend representation by an adapter.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from ..metadata.filter_spec import FilterOperator, MetadataFilterSpec


@dataclass
class Query:
    """A user query plus optional structured metadata filters."""

    query_text: str
    metadata_filters: MetadataFilterSpec = field(default_factory=MetadataFilterSpec)
    top_k: int = 10
    query_id: str = field(default_factory=lambda: f"q_{uuid.uuid4().hex[:12]}")

    def __post_init__(self) -> None:
        if not isinstance(self.query_text, str) or not self.query_text.strip():
            raise ValueError("query_text must be a non-empty string")
        if isinstance(self.metadata_filters, dict):
            self.metadata_filters = MetadataFilterSpec.from_dict(self.metadata_filters)
        if self.metadata_filters is None:
            self.metadata_filters = MetadataFilterSpec()

    def get_metadata_filter_spec(self) -> MetadataFilterSpec:
        """Return the structured metadata filter attached to this query."""
        return self.metadata_filters

    def has_filters(self) -> bool:
        return not self.metadata_filters.is_empty()

    def with_filter(
        self,
        field_name: str,
        value: Any,
        operator: Union[FilterOperator, str] = FilterOperator.EQ,
    ) -> Query:
        """Return a copy of this query with an extra metadata filter."""
        spec = MetadataFilterSpec(filters=list(self.metadata_filters.filters))
        spec.add(field_name, operator, value)
        return Query(
            query_text=self.query_text,
            metadata_filters=spec,
            top_k=self.top_k,
            query_id=self.query_id,
        )

    @classmethod
    def of(
        cls,
        query_text: str,
        *,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> Query:
        """Convenience constructor from text + a plain ``{field: value}`` dict."""
        spec = MetadataFilterSpec.from_dict(filters or {})
        return cls(query_text=query_text, metadata_filters=spec, top_k=top_k)
