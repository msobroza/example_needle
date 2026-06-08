"""Backend-agnostic metadata filter specification.

A :class:`MetadataFilterSpec` is an ordered conjunction (logical AND) of
:class:`FieldFilter` predicates. It is deliberately decoupled from any
storage backend; :mod:`conversational_core.domain.metadata.backend_filter_adapters`
translates a spec into a concrete backend representation.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union


class FilterOperator(str, Enum):
    """Comparison operators supported by a :class:`FieldFilter`."""

    EQ = "eq"
    NE = "ne"
    IN = "in"
    NIN = "nin"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    CONTAINS = "contains"
    EXISTS = "exists"
    REGEX = "regex"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    @classmethod
    def coerce(cls, value: Union[FilterOperator, str]) -> FilterOperator:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


@dataclass(frozen=True)
class FieldFilter:
    """A single ``field <operator> value`` predicate."""

    field: str
    operator: FilterOperator = FilterOperator.EQ
    value: Any = None

    def __post_init__(self) -> None:
        if not self.field or not isinstance(self.field, str):
            raise ValueError("FieldFilter.field must be a non-empty string")
        # ``frozen=True`` blocks normal assignment; use object.__setattr__.
        object.__setattr__(self, "operator", FilterOperator.coerce(self.operator))


@dataclass
class MetadataFilterSpec:
    """An AND-conjunction of :class:`FieldFilter` predicates."""

    filters: list[FieldFilter] = field(default_factory=list)

    def add(
        self,
        field_name: Union[str, FieldFilter],
        operator: Union[FilterOperator, str] = FilterOperator.EQ,
        value: Any = None,
    ) -> MetadataFilterSpec:
        """Append a predicate. Returns ``self`` for chaining."""
        if isinstance(field_name, FieldFilter):
            self.filters.append(field_name)
        else:
            self.filters.append(
                FieldFilter(field_name, FilterOperator.coerce(operator), value)
            )
        return self

    def is_empty(self) -> bool:
        return not self.filters

    def fields(self) -> set[str]:
        return {f.field for f in self.filters}

    def __iter__(self) -> Iterator[FieldFilter]:
        return iter(self.filters)

    def __len__(self) -> int:
        return len(self.filters)

    def __bool__(self) -> bool:
        return bool(self.filters)

    @classmethod
    def from_dict(cls, mapping: Union[dict[str, Any], None]) -> MetadataFilterSpec:
        """Build a spec from a plain ``{field: value}`` mapping.

        List/tuple/set values become an ``IN`` filter; everything else
        becomes an equality filter.
        """
        spec = cls()
        for key, value in (mapping or {}).items():
            if isinstance(value, (list, tuple, set)):
                spec.add(key, FilterOperator.IN, list(value))
            else:
                spec.add(key, FilterOperator.EQ, value)
        return spec
