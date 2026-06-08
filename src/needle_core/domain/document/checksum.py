"""Content checksums for document files.

Thin wrappers around :mod:`hashlib` SHA-256. ``sha256_file`` streams the file
in fixed-size chunks so it works on large documents without loading them
entirely into memory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

_CHUNK_SIZE = 65536


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Union[str, Path]) -> str:
    """Return the hex SHA-256 digest of the file at ``path`` (streamed)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_checksum(path: Union[str, Path], length: int = 12) -> str:
    """Return the first ``length`` characters of the file's SHA-256 digest."""
    if length < 1:
        raise ValueError("length must be >= 1")
    return sha256_file(path)[:length]
