"""Typed identifier value objects for the conversational retrieval domain.

These thin wrappers around a ``str`` give documents, versions and queries
distinct, self-describing identifier types while remaining trivially
serialisable (they stringify to the raw value). Identifiers are generated
with a short uuid-based suffix and a type-appropriate prefix (e.g.
``doc_1a2b3c4d5e6f``).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass


def _generate(prefix: str, *, hex_length: int = 12) -> str:
    """Return a new identifier of the form ``<prefix>_<hex>``."""
    return f"{prefix}_{uuid.uuid4().hex[:hex_length]}"


@dataclass(frozen=True)
class DocumentId:
    """Stable identifier for a logical :class:`Document`."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("DocumentId.value must be a non-empty string")

    def __str__(self) -> str:
        return self.value

    @classmethod
    def new(cls) -> DocumentId:
        """Generate a fresh ``doc_<hex12>`` identifier."""
        return cls(_generate("doc"))


@dataclass(frozen=True)
class VersionId:
    """Stable identifier for a :class:`DocumentVersion`."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("VersionId.value must be a non-empty string")

    def __str__(self) -> str:
        return self.value

    @classmethod
    def new(cls) -> VersionId:
        """Generate a fresh ``ver_<hex12>`` identifier."""
        return cls(_generate("ver"))


@dataclass(frozen=True)
class QueryId:
    """Stable identifier for a :class:`Query`."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("QueryId.value must be a non-empty string")

    def __str__(self) -> str:
        return self.value

    @classmethod
    def new(cls) -> QueryId:
        """Generate a fresh ``q_<hex12>`` identifier."""
        return cls(_generate("q"))
