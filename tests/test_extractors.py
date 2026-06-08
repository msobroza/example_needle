"""Tests for the page extractors (Pillow only — no torch needed)."""

from __future__ import annotations

import pytest

from needle.retrieval.extractors import (
    IMAGE_EXTRACTORS,
    ImageFileExtractor,
    OfficeToImageExtractor,
    PdfToImageExtractor,
    get_extractor,
)
from needle_core.domain.exceptions import UnsupportedExtensionError
from needle_core.domain.types import DocumentExtension


def test_image_file_extractor_single_page(png_factory):
    path = png_factory("one.png", "white", "hi")
    pages = ImageFileExtractor().extract(path, dpi=150)
    assert len(pages) == 1
    assert pages[0].mode == "RGB"
    assert pages[0].size == (256, 320)


def test_image_file_extractor_multi_frame(tmp_path):
    from PIL import Image

    frames = [
        Image.new("RGB", (64, 64), "red"),
        Image.new("RGB", (64, 64), "green"),
        Image.new("RGB", (64, 64), "blue"),
    ]
    path = tmp_path / "anim.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:])

    pages = ImageFileExtractor().extract(path)
    assert len(pages) == 3
    assert all(p.mode == "RGB" for p in pages)


def test_extract_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ImageFileExtractor().extract(tmp_path / "nope.png")


def test_get_extractor_known_and_unknown():
    assert isinstance(get_extractor("png"), ImageFileExtractor)
    assert isinstance(get_extractor(DocumentExtension.PDF), PdfToImageExtractor)
    with pytest.raises(UnsupportedExtensionError):
        get_extractor("txt")


def test_image_extractors_cover_expected_formats():
    expected = {
        DocumentExtension.PDF,
        DocumentExtension.DOCX,
        DocumentExtension.PPTX,
        DocumentExtension.PNG,
        DocumentExtension.JPG,
        DocumentExtension.JPEG,
        DocumentExtension.TIFF,
        DocumentExtension.WEBP,
        DocumentExtension.BMP,
        DocumentExtension.GIF,
    }
    assert expected <= set(IMAGE_EXTRACTORS)


def test_pdf_extractor_on_invalid_pdf_raises(tmp_path):
    try:
        import fitz  # noqa: F401

        has_fitz = True
    except ImportError:
        has_fitz = False

    pdf = tmp_path / "empty.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")  # header only — not a valid PDF body

    if has_fitz:
        # With PyMuPDF installed, opening an invalid PDF raises *some* error.
        with pytest.raises(Exception):  # noqa: B017 - backend-specific error type
            PdfToImageExtractor().extract(pdf)
    else:
        # Without PyMuPDF, we surface a clear, actionable dependency error.
        with pytest.raises(UnsupportedExtensionError):
            PdfToImageExtractor().extract(pdf)


def test_office_extractor_supported_extensions():
    ext = OfficeToImageExtractor()
    assert ext.supports(DocumentExtension.DOCX)
    assert ext.supports(DocumentExtension.PPTX)
    assert not ext.supports(DocumentExtension.PNG)
