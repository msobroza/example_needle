"""Tests for the io subpackage (paths + manifests)."""

from __future__ import annotations

from pathlib import Path

from conversational_core.domain.types import DocumentExtension
from needle.io.manifests import IndexManifest, ManifestEntry
from needle.io.paths import ensure_dir, human_size, iter_files


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
def test_ensure_dir_creates_nested_dirs(tmp_path: Path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert result == target
    assert target.is_dir()
    # Idempotent: a second call must not raise.
    assert ensure_dir(target) == target


def test_iter_files_finds_by_suffix(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "sub" / "b.PDF"
    txt_c = tmp_path / "c.txt"
    for path in (pdf_a, pdf_b, txt_c):
        path.write_text("x", encoding="utf-8")

    # No filter -> every file, sorted.
    assert iter_files(tmp_path) == sorted([pdf_a, pdf_b, txt_c])

    # Suffix filter is case-insensitive and dot-optional.
    assert iter_files(tmp_path, suffixes=["pdf"]) == sorted([pdf_a, pdf_b])
    assert iter_files(tmp_path, suffixes=[".PDF"]) == sorted([pdf_a, pdf_b])
    assert iter_files(tmp_path, suffixes=["txt"]) == [txt_c]


def test_iter_files_missing_root(tmp_path: Path):
    assert iter_files(tmp_path / "nope") == []


def test_human_size_formats_units():
    assert human_size(0) == "0 B"
    assert human_size(512) == "512 B"
    assert human_size(1024) == "1.0 KB"
    assert human_size(1536) == "1.5 KB"
    assert human_size(1024 * 1024) == "1.0 MB"
    assert human_size(5 * 1024 * 1024) == "5.0 MB"


# ---------------------------------------------------------------------------
# manifests
# ---------------------------------------------------------------------------
def test_manifest_entry_stores_format_as_string():
    entry = ManifestEntry(
        file="report.pdf",
        format=str(DocumentExtension.PDF),
        pages=3,
        metadata={"year": 2024},
    )
    assert entry.format == "pdf"
    assert entry.to_dict()["format"] == "pdf"


def test_index_manifest_len_and_total_pages():
    manifest = IndexManifest()
    assert len(manifest) == 0
    assert manifest.total_pages == 0

    manifest.add(ManifestEntry("a.pdf", "pdf", 3, {}))
    manifest.add(ManifestEntry("b.png", "png", 1, {"lang": "en"}))

    assert len(manifest) == 2
    assert manifest.total_pages == 4


def test_index_manifest_json_round_trip():
    manifest = IndexManifest()
    manifest.add(ManifestEntry("a.pdf", str(DocumentExtension.PDF), 5, {"y": 2024}))
    manifest.add(ManifestEntry("b.docx", str(DocumentExtension.DOCX), 2, {}))

    restored = IndexManifest.from_json(manifest.to_json())
    assert restored.to_dict() == manifest.to_dict()
    assert len(restored) == 2
    assert restored.total_pages == 7
    assert restored.created_at == manifest.created_at


def test_index_manifest_save_load_round_trip(tmp_path: Path):
    manifest = IndexManifest()
    manifest.add(ManifestEntry("a.pdf", "pdf", 5, {"y": 2024}))
    manifest.add(ManifestEntry("b.png", "png", 1, {}))

    path = tmp_path / "nested" / "manifest.json"
    saved = manifest.save(path)
    assert saved == path
    assert path.is_file()

    loaded = IndexManifest.load(path)
    assert loaded.to_dict() == manifest.to_dict()
    assert loaded.total_pages == 6
    assert len(loaded) == 2
