"""Index manifests describing the documents held in an index.

A :class:`ManifestEntry` records one indexed document (its file, format, page
count and arbitrary metadata); an :class:`IndexManifest` is an ordered,
JSON-round-trippable collection of such entries plus a creation timestamp.

Formats are stored as their plain string value (e.g. ``"pdf"``), matching
:class:`conversational_core.domain.types.DocumentExtension`, which subclasses
``str`` and serialises to its value without custom plumbing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ManifestEntry:
    """A single indexed document.

    ``format`` holds the document extension as a plain string value, e.g.
    ``"pdf"`` (see :class:`~conversational_core.domain.types.DocumentExtension`).
    """

    file: str
    format: str
    pages: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this entry."""
        return {
            "file": self.file,
            "format": str(self.format),
            "pages": self.pages,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ManifestEntry:
        """Rebuild a :class:`ManifestEntry` from a mapping."""
        return cls(
            file=d["file"],
            format=str(d["format"]),
            pages=int(d["pages"]),
            metadata=dict(d.get("metadata") or {}),
        )


@dataclass
class IndexManifest:
    """An ordered collection of :class:`ManifestEntry` with a creation time."""

    entries: list[ManifestEntry] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)

    def add(self, entry: ManifestEntry) -> None:
        """Append ``entry`` to the manifest."""
        self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def total_pages(self) -> int:
        """Total number of pages across all entries."""
        return sum(entry.pages for entry in self.entries)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the manifest."""
        return {
            "created_at": self.created_at,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IndexManifest:
        """Rebuild an :class:`IndexManifest` from a mapping."""
        entries = [ManifestEntry.from_dict(e) for e in d.get("entries", [])]
        created_at = d.get("created_at") or _utc_now_iso()
        return cls(entries=entries, created_at=created_at)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the manifest to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, s: str) -> IndexManifest:
        """Deserialise a manifest from a JSON string."""
        return cls.from_dict(json.loads(s))

    def save(self, path: str | Path) -> Path:
        """Write the manifest as JSON to ``path`` and return the path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> IndexManifest:
        """Read a manifest from the JSON file at ``path``."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_json(text)
