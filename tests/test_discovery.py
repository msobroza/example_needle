"""Tests for filesystem document discovery."""

from __future__ import annotations

from pathlib import Path

from conversational_core.domain.document.checksum import sha256_file
from conversational_core.domain.document.discovery import (
    count_by_extension,
    discover_documents,
)
from conversational_core.domain.document.document import Document, DocumentVersion


def _populate(root: Path) -> None:
    (root / "a.pdf").write_bytes(b"%PDF-1.4 fake")
    (root / "b.txt").write_text("hello")
    (root / "c.md").write_text("# title")
    (root / "ignore.bin").write_bytes(b"not a document")
    (root / "no_extension").write_text("skip me")
    sub = root / "sub"
    sub.mkdir()
    (sub / "d.txt").write_text("nested")


def test_discover_returns_document_version_pairs(tmp_path: Path) -> None:
    _populate(tmp_path)
    results = discover_documents(tmp_path)
    assert results
    for doc, version in results:
        assert isinstance(doc, Document)
        assert isinstance(version, DocumentVersion)


def test_discover_filters_unknown_extensions(tmp_path: Path) -> None:
    _populate(tmp_path)
    results = discover_documents(tmp_path)
    names = {Path(v.document_path).name for _, v in results}
    assert "ignore.bin" not in names
    assert "no_extension" not in names
    assert names == {"a.pdf", "b.txt", "c.md", "d.txt"}


def test_discover_is_sorted_by_path(tmp_path: Path) -> None:
    _populate(tmp_path)
    results = discover_documents(tmp_path)
    paths = [v.document_path for _, v in results]
    assert paths == sorted(paths)


def test_discover_format_metadata(tmp_path: Path) -> None:
    _populate(tmp_path)
    results = discover_documents(tmp_path)
    by_name = {Path(v.document_path).name: v for _, v in results}
    assert by_name["a.pdf"].document_metadata["format"] == "pdf"
    assert by_name["b.txt"].document_metadata["format"] == "txt"
    assert by_name["c.md"].document_metadata["format"] == "md"


def test_discover_suffix_allowlist(tmp_path: Path) -> None:
    _populate(tmp_path)
    results = discover_documents(tmp_path, suffixes=["txt"])
    names = {Path(v.document_path).name for _, v in results}
    assert names == {"b.txt", "d.txt"}


def test_discover_suffix_allowlist_with_dot_and_case(tmp_path: Path) -> None:
    _populate(tmp_path)
    results = discover_documents(tmp_path, suffixes=[".PDF", "MD"])
    names = {Path(v.document_path).name for _, v in results}
    assert names == {"a.pdf", "c.md"}


def test_discover_without_checksum_by_default(tmp_path: Path) -> None:
    _populate(tmp_path)
    results = discover_documents(tmp_path)
    assert all(v.checksum is None for _, v in results)


def test_discover_with_checksum(tmp_path: Path) -> None:
    _populate(tmp_path)
    results = discover_documents(tmp_path, with_checksum=True)
    for _, version in results:
        assert version.checksum == sha256_file(version.document_path)


def test_count_by_extension(tmp_path: Path) -> None:
    _populate(tmp_path)
    counts = count_by_extension(tmp_path)
    assert counts == {"pdf": 1, "txt": 2, "md": 1}


def test_empty_directory(tmp_path: Path) -> None:
    assert discover_documents(tmp_path) == []
    assert count_by_extension(tmp_path) == {}
