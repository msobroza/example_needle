"""Tests for document content checksums."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from conversational_core.domain.document.checksum import (
    sha256_bytes,
    sha256_file,
    short_checksum,
)


def test_sha256_bytes_matches_hashlib() -> None:
    data = b"hello world"
    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_sha256_bytes_deterministic() -> None:
    assert sha256_bytes(b"abc") == sha256_bytes(b"abc")


def test_sha256_file_matches_bytes(tmp_path: Path) -> None:
    data = b"some document content" * 1000
    path = tmp_path / "doc.bin"
    path.write_bytes(data)
    assert sha256_file(path) == sha256_bytes(data)


def test_sha256_file_streamed_large(tmp_path: Path) -> None:
    # Larger than the internal chunk size to exercise streaming.
    data = b"x" * (65536 * 3 + 17)
    path = tmp_path / "big.bin"
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_short_checksum_prefix(tmp_path: Path) -> None:
    path = tmp_path / "doc.bin"
    path.write_bytes(b"content")
    full = sha256_file(path)
    assert short_checksum(path) == full[:12]
    assert short_checksum(path, length=8) == full[:8]
    assert len(short_checksum(path)) == 12


def test_short_checksum_invalid_length(tmp_path: Path) -> None:
    path = tmp_path / "doc.bin"
    path.write_bytes(b"content")
    with pytest.raises(ValueError):
        short_checksum(path, length=0)


def test_different_content_different_digest(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"aaa")
    b.write_bytes(b"bbb")
    assert sha256_file(a) != sha256_file(b)
