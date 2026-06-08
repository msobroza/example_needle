"""Data carriers exchanged with the page retrievers.

These types glue the :mod:`needle_core` domain (documents, queries)
to the retriever I/O:

* :class:`InputDocument` — what you hand to ``retriever.index(...)``.
* :class:`PreannotationPageResult` — what ``retriever.search(...)`` returns.
* :class:`PageAnnotationResult` — the ``(version, page, annotation)`` triple
  consumed by ``retriever.show(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union

from needle_core.domain.document.document import (
    Document,
    DocumentPage,
    DocumentVersion,
)
from needle_core.domain.interaction.query import Query
from needle_core.domain.types import DocumentExtension


@dataclass
class InputDocument:
    """A document paired with the concrete version to be indexed."""

    document: Document
    document_version: DocumentVersion

    @classmethod
    def from_path(
        cls,
        path: Union[str, Path],
        *,
        metadata: Optional[dict[str, Any]] = None,
        title: Optional[str] = None,
        source: Optional[str] = None,
    ) -> InputDocument:
        """Build an :class:`InputDocument` directly from a file path."""
        path = Path(path)
        document = Document.from_path(path, title=title, source=source)
        version = DocumentVersion(
            document_path=str(path),
            document_metadata=dict(metadata or {}),
        )
        return cls(document=document, document_version=version)

    @property
    def extension(self) -> DocumentExtension:
        return self.document.document_ext


@dataclass
class PageAnnotation:
    """Scoring / annotation payload attached to a single page."""

    score: Optional[float] = None
    normalized_score: Optional[float] = None
    annotations: dict[str, Any] = field(default_factory=dict)


class PageAnnotationResult(NamedTuple):
    """A ``(version, page, annotation)`` triple — consumed by ``show()``.

    Tuple-shaped on purpose so callers can unpack it::

        for version, page, annotation in results:
            ...
    """

    document_version: DocumentVersion
    document_page: DocumentPage
    page_annotation: PageAnnotation


@dataclass
class PreannotationPageResult:
    """A single page hit returned by ``retriever.search()``."""

    query: Query
    document: Document
    document_version: DocumentVersion
    page: int
    score: float
    normalized_score: float = 0.0

    def to_annotation_result(self) -> PageAnnotationResult:
        """Adapt to the ``(version, page, annotation)`` triple used by ``show``."""
        return PageAnnotationResult(
            document_version=self.document_version,
            document_page=DocumentPage(page_number=self.page),
            page_annotation=PageAnnotation(
                score=self.score, normalized_score=self.normalized_score
            ),
        )
