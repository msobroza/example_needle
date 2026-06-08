"""Tests for the document aggregate."""

from __future__ import annotations

import pytest

from conversational_core.domain.document.document import (
    Document,
    DocumentPage,
    DocumentVersion,
)
from conversational_core.domain.types import DocumentExtension


def test_document_from_path_infers_extension_and_title():
    doc = Document.from_path("/data/Q3_report.pdf")
    assert doc.document_ext is DocumentExtension.PDF
    assert doc.title == "Q3_report"
    assert doc.document_id.startswith("doc_")


def test_document_coerces_string_extension():
    doc = Document(document_ext="png")
    assert doc.document_ext is DocumentExtension.PNG


def test_document_version_path_helpers():
    version = DocumentVersion(document_path="/tmp/a/b.pdf")
    assert version.filename == "b.pdf"
    assert version.path.name == "b.pdf"
    assert version.document_metadata == {}


def test_document_version_none_metadata_becomes_dict():
    version = DocumentVersion(document_path="x.pdf", document_metadata=None)
    assert version.document_metadata == {}


def test_document_version_with_metadata_merges_immutably():
    version = DocumentVersion(document_path="x.pdf", document_metadata={"a": 1})
    merged = version.with_metadata(b=2)
    assert merged.document_metadata == {"a": 1, "b": 2}
    # original untouched
    assert version.document_metadata == {"a": 1}


def test_document_page_rejects_non_positive_page_number():
    assert DocumentPage(page_number=1).page_number == 1
    with pytest.raises(ValueError):
        DocumentPage(page_number=0)
