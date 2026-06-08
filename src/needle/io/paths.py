"""Filesystem path utilities.

Small, dependency-free helpers for the kinds of path bookkeeping the indexing
and CLI layers repeat: creating output directories, enumerating input files by
suffix, and formatting byte counts for humans.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

_SIZE_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and any missing parents) and return it as a ``Path``.

    Existing directories are left untouched.
    """
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _normalize_suffix(suffix: str) -> str:
    """Lower-case a suffix and ensure it carries a single leading dot."""
    cleaned = suffix.strip().lower()
    if not cleaned:
        return ""
    return cleaned if cleaned.startswith(".") else f".{cleaned}"


def iter_files(
    root: str | Path,
    suffixes: Iterable[str] | None = None,
) -> list[Path]:
    """Return files under ``root`` recursively, sorted by path.

    When ``suffixes`` is given, only files whose extension matches are kept.
    Matching is case-insensitive and accepts suffixes with or without a leading
    dot (``"pdf"`` and ``".PDF"`` are equivalent). A non-existent ``root``
    yields an empty list.
    """
    root_path = Path(root)
    if not root_path.exists():
        return []

    allowed: set[str] | None = None
    if suffixes is not None:
        allowed = {_normalize_suffix(s) for s in suffixes}
        allowed.discard("")

    files = (p for p in root_path.rglob("*") if p.is_file())
    if allowed is not None:
        files = (p for p in files if p.suffix.lower() in allowed)
    return sorted(files)


def human_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable string (e.g. ``"1.5 KB"``).

    Uses binary (1024) steps. Bytes are shown without decimals; larger units
    use one decimal place. Negative inputs are formatted with a leading sign.
    """
    if num_bytes < 0:
        return f"-{human_size(-num_bytes)}"

    size = float(num_bytes)
    for unit in _SIZE_UNITS:
        if size < 1024.0 or unit == _SIZE_UNITS[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    # Unreachable: the loop always returns on the last unit.
    return f"{size:.1f} {_SIZE_UNITS[-1]}"
