"""The document aggregate.

A :class:`Document` is the *logical* entity ("the Q3 earnings report"). A
:class:`DocumentVersion` is a concrete materialisation of that document on
disk (a particular file at a particular path, with metadata). A
:class:`DocumentPage` is a single page within a version.

These are intentionally plain dataclasses: the domain layer stays free of
ORM / pydantic / ML dependencies so it can be imported anywhere.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from ..types import DocumentExtension


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Document:
    """A logical document, independent of any particular file on disk."""

    document_ext: DocumentExtension
    document_id: str = field(default_factory=lambda: _new_id("doc"))
    title: Optional[str] = None
    source: Optional[str] = None

    def __post_init__(self) -> None:
        # Be forgiving about how the extension is supplied.
        if not isinstance(self.document_ext, DocumentExtension):
            self.document_ext = DocumentExtension.from_string(self.document_ext)

    @classmethod
    def from_path(
        cls,
        path: Union[str, Path],
        *,
        title: Optional[str] = None,
        source: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Document:
        path = Path(path)
        return cls(
            document_ext=DocumentExtension.from_path(path),
            document_id=document_id or _new_id("doc"),
            title=title if title is not None else path.stem,
            source=source,
        )


@dataclass
class DocumentVersion:
    """A concrete on-disk materialisation of a :class:`Document`."""

    document_path: str
    version_id: str = field(default_factory=lambda: _new_id("ver"))
    checksum: Optional[str] = None
    document_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.document_path = str(self.document_path)
        if self.document_metadata is None:
            self.document_metadata = {}

    @property
    def path(self) -> Path:
        return Path(self.document_path)

    @property
    def filename(self) -> str:
        return self.path.name

    def with_metadata(self, **metadata: Any) -> DocumentVersion:
        """Return a copy with additional metadata merged in."""
        merged = {**self.document_metadata, **metadata}
        return DocumentVersion(
            document_path=self.document_path,
            version_id=self.version_id,
            checksum=self.checksum,
            document_metadata=merged,
        )


@dataclass
class DocumentPage:
    """A single page within a document version."""

    page_number: int
    text: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError(
                f"page_number is 1-indexed and must be >= 1, got {self.page_number}"
            )
