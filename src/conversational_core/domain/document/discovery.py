"""Filesystem discovery of indexable documents.

Walks a directory tree and turns recognised files into
``(Document, DocumentVersion)`` pairs ready for indexing. A file is kept when
its extension is a known :class:`DocumentExtension` (or appears in an
explicit ``suffixes`` allow-list).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

from ..types import DocumentExtension
from .checksum import sha256_file
from .document import Document, DocumentVersion


def _normalize_suffixes(suffixes: Optional[Iterable[str]]) -> Optional[set[str]]:
    """Normalise an explicit suffix allow-list to lowercase, dot-less form."""
    if suffixes is None:
        return None
    return {str(s).strip().lower().lstrip(".") for s in suffixes}


def _extension_of(path: Path) -> Optional[DocumentExtension]:
    """Return the known extension for ``path``, or ``None`` if unrecognised."""
    try:
        return DocumentExtension.from_path(path)
    except ValueError:
        return None


def discover_documents(
    root: Union[str, Path],
    suffixes: Optional[Iterable[str]] = None,
    *,
    with_checksum: bool = False,
) -> list[tuple[Document, DocumentVersion]]:
    """Discover indexable documents under ``root``.

    Files are kept when their extension resolves to a known
    :class:`DocumentExtension`. When ``suffixes`` is given, only those
    (case-insensitive, dot-optional) extensions are kept. Results are sorted
    by path. When ``with_checksum`` is true each version's ``checksum`` is
    filled from :func:`sha256_file`.
    """
    root_path = Path(root)
    allow = _normalize_suffixes(suffixes)

    discovered: list[tuple[Document, DocumentVersion]] = []
    for path in sorted(p for p in root_path.rglob("*") if p.is_file()):
        ext = _extension_of(path)
        if ext is None:
            continue
        if allow is not None and ext.value not in allow:
            continue

        document = Document.from_path(path)
        version = DocumentVersion(
            document_path=str(path),
            checksum=sha256_file(path) if with_checksum else None,
            document_metadata={"format": ext.value},
        )
        discovered.append((document, version))

    return discovered


def count_by_extension(root: Union[str, Path]) -> dict[str, int]:
    """Return a ``{extension_value: count}`` map of recognised files."""
    counts: dict[str, int] = {}
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        ext = _extension_of(path)
        if ext is None:
            continue
        counts[ext.value] = counts.get(ext.value, 0) + 1
    return counts
