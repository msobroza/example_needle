"""Tests for the retriever data carriers."""

from __future__ import annotations

from needle.retrieval.data import (
    InputDocument,
    PageAnnotation,
    PageAnnotationResult,
    PreannotationPageResult,
)
from needle_core.domain.interaction.query import Query
from needle_core.domain.types import DocumentExtension


def test_input_document_from_path():
    doc = InputDocument.from_path("/data/report.pdf", metadata={"year": 2024})
    assert doc.document.document_ext is DocumentExtension.PDF
    assert doc.extension is DocumentExtension.PDF
    assert doc.document_version.document_metadata == {"year": 2024}
    assert doc.document_version.document_path == "/data/report.pdf"


def test_input_document_default_metadata_is_dict():
    doc = InputDocument.from_path("/data/a.png")
    assert doc.document_version.document_metadata == {}


def test_page_annotation_defaults():
    annotation = PageAnnotation()
    assert annotation.score is None
    assert annotation.normalized_score is None
    assert annotation.annotations == {}


def test_preannotation_to_annotation_result_round_trip():
    doc = InputDocument.from_path("/data/report.pdf", metadata={"year": 2024})
    result = PreannotationPageResult(
        query=Query(query_text="total"),
        document=doc.document,
        document_version=doc.document_version,
        page=3,
        score=1.23,
        normalized_score=0.9,
    )
    triple = result.to_annotation_result()
    assert isinstance(triple, PageAnnotationResult)

    # NamedTuple unpacking — the exact shape BasePageRetriever.show() expects.
    version, page, annotation = triple
    assert page.page_number == 3
    assert annotation.score == 1.23
    assert annotation.normalized_score == 0.9
    assert version.document_metadata == {"year": 2024}
