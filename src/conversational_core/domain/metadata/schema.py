"""Declarative metadata schema validation.

A :class:`MetadataSchema` is a list of :class:`MetadataField` declarations
describing the expected shape of a document's metadata. It validates a plain
``dict`` and returns human-readable error messages for missing required
fields, type mismatches and disallowed values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MetadataField:
    """A single declared metadata field."""

    name: str
    type: type = str
    required: bool = False
    choices: Optional[tuple] = None


@dataclass
class MetadataSchema:
    """A collection of :class:`MetadataField` declarations."""

    fields: list[MetadataField]

    def validate(self, metadata: dict[str, Any]) -> list[str]:
        """Return a list of human-readable validation errors (empty if valid)."""
        errors: list[str] = []
        for spec in self.fields:
            if spec.name not in metadata:
                if spec.required:
                    errors.append(f"missing required field: {spec.name!r}")
                continue

            value = metadata[spec.name]
            # ``bool`` is a subclass of ``int``; reject the surprising match
            # unless the field explicitly expects a bool.
            if spec.type is not bool and isinstance(value, bool):
                if not isinstance(value, spec.type) or spec.type is int:
                    errors.append(
                        f"field {spec.name!r} expected type "
                        f"{spec.type.__name__}, got bool"
                    )
                    continue
            elif not isinstance(value, spec.type):
                errors.append(
                    f"field {spec.name!r} expected type "
                    f"{spec.type.__name__}, got {type(value).__name__}"
                )
                continue

            if spec.choices is not None and value not in spec.choices:
                errors.append(
                    f"field {spec.name!r} value {value!r} not in "
                    f"allowed choices {list(spec.choices)}"
                )
        return errors

    def is_valid(self, metadata: dict[str, Any]) -> bool:
        """Whether ``metadata`` satisfies the schema."""
        return not self.validate(metadata)
