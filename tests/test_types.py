"""Tests for :mod:`conversational_core.domain.types`."""

from __future__ import annotations

import pytest

from conversational_core.domain.types import DocumentExtension


def test_from_string_variants():
    assert DocumentExtension.from_string("pdf") is DocumentExtension.PDF
    assert DocumentExtension.from_string("PDF") is DocumentExtension.PDF
    assert DocumentExtension.from_string(".PDF") is DocumentExtension.PDF
    assert DocumentExtension.from_string(DocumentExtension.PDF) is DocumentExtension.PDF


def test_from_string_aliases():
    assert DocumentExtension.from_string("tif") is DocumentExtension.TIFF
    assert DocumentExtension.from_string("markdown") is DocumentExtension.MARKDOWN
    assert DocumentExtension.from_string("htm") is DocumentExtension.HTML
    assert DocumentExtension.from_string("jpeg") is DocumentExtension.JPEG


def test_from_path():
    assert DocumentExtension.from_path("/a/b/c.PNG") is DocumentExtension.PNG
    assert DocumentExtension.from_path("report.pdf") is DocumentExtension.PDF


def test_from_path_without_suffix_raises():
    with pytest.raises(ValueError):
        DocumentExtension.from_path("no_extension_here")


def test_unknown_extension_raises():
    with pytest.raises(ValueError):
        DocumentExtension.from_string("xyz")


def test_str_and_repr():
    assert str(DocumentExtension.PDF) == "pdf"
    assert f"{DocumentExtension.PDF}" == "pdf"
    assert repr(DocumentExtension.PDF) == "DocumentExtension.PDF"


def test_str_enum_equality_and_dict_key():
    assert DocumentExtension.PDF == "pdf"
    table = {DocumentExtension.PDF: 1, DocumentExtension.PNG: 2}
    assert table["pdf"] == 1  # str-enum membership works with raw strings


def test_is_image():
    assert DocumentExtension.PNG.is_image
    assert DocumentExtension.JPG.is_image
    assert not DocumentExtension.PDF.is_image
    assert not DocumentExtension.DOCX.is_image


def test_is_renderable():
    assert DocumentExtension.PDF.is_renderable
    assert DocumentExtension.DOCX.is_renderable
    assert DocumentExtension.PNG.is_renderable
    assert not DocumentExtension.TXT.is_renderable
