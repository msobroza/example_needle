"""Adapters that translate a :class:`MetadataFilterSpec` into a backend form.

The in-memory page retriever consumes the output of
:class:`MultiFieldFilterAdapter` as a mapping of ``field -> criterion`` where
each *criterion* is a Mongo-style operator dict (``{"$gte": 2020}``). The same
representation is understood by
:func:`needle.retrieval.page_retriever_utils.matches_filter`, keeping the
producer and consumer in lock-step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Union

from .filter_spec import FilterOperator, MetadataFilterSpec

#: Stable mapping from domain operators to Mongo-style operator keys.
OPERATOR_TO_BACKEND: dict[FilterOperator, str] = {
    FilterOperator.EQ: "$eq",
    FilterOperator.NE: "$ne",
    FilterOperator.IN: "$in",
    FilterOperator.NIN: "$nin",
    FilterOperator.GT: "$gt",
    FilterOperator.GTE: "$gte",
    FilterOperator.LT: "$lt",
    FilterOperator.LTE: "$lte",
    FilterOperator.CONTAINS: "$contains",
    FilterOperator.EXISTS: "$exists",
    FilterOperator.REGEX: "$regex",
}


class BackendFilterAdapter(ABC):
    """Strategy that converts a spec into a backend-specific representation."""

    @abstractmethod
    def to_backend(self, spec: Union[MetadataFilterSpec, dict[str, Any], None]) -> Any:
        """Convert ``spec`` into the backend representation."""


class MultiFieldFilterAdapter(BackendFilterAdapter):
    """Translate a spec into ``{field: {"$op": value, ...}}``.

    Multiple predicates on the same field are merged into one criterion dict,
    which naturally expresses ranges (``year >= 2000`` *and* ``year <= 2010``).
    """

    def to_backend(
        self, spec: Union[MetadataFilterSpec, dict[str, Any], None] = None
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        if spec is None:
            return result
        if isinstance(spec, dict):
            spec = MetadataFilterSpec.from_dict(spec)

        for field_filter in spec:
            op_key = OPERATOR_TO_BACKEND[field_filter.operator]
            result.setdefault(field_filter.field, {})[op_key] = field_filter.value
        return result
