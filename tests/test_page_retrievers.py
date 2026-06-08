"""End-to-end tests for the page retrievers (the centerpiece module).

These exercise the real indexing / scoring / filtering / persistence logic in
``needle.retrieval.page_retrievers`` through the deterministic
``DummyEmbedderRetriever``. The module is skipped wholesale when PyTorch is not
installed (the centerpiece module imports torch at module load time).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

pytest.importorskip("torch")

from conversational_core.domain.interaction.query import Query  # noqa: E402
from needle.retrieval.data import InputDocument  # noqa: E402
from needle.retrieval.extractors import IMAGE_EXTRACTORS  # noqa: E402
from needle.retrieval.page_retrievers import (  # noqa: E402
    BasePageRetriever,
    ColModernVBertRetriever,
    ColPaliRetriever,
    ColQwen2Retriever,
    ModernVBertRetriever,
    MultimodalEmbedderRetriever,
    NomicDenseRetriever,
    TomoroColQwen3Retriever,
)
from needle.testing import DummyEmbedderRetriever  # noqa: E402

pytestmark = pytest.mark.torch


# ---------------------------------------------------------------------------
# Abstract contracts
# ---------------------------------------------------------------------------
def test_abstract_classes_cannot_be_instantiated(tmp_path):
    with pytest.raises(TypeError):
        BasePageRetriever()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        MultimodalEmbedderRetriever(index_path=str(tmp_path / "i.pkl"))  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Pure scoring helpers
# ---------------------------------------------------------------------------
def test_minmax_normalisation():
    out = BasePageRetriever._minmax(np.array([0.0, 5.0, 10.0]))
    assert out.tolist() == [0.0, 0.5, 1.0]


def test_minmax_constant_array_is_zeros():
    out = BasePageRetriever._minmax(np.array([3.0, 3.0, 3.0]))
    assert out.tolist() == [0.0, 0.0, 0.0]


def test_multi_vector_maxsim_score(tmp_path):
    r = DummyEmbedderRetriever(index_path=str(tmp_path / "i.pkl"), multi_vector=True)
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    doc = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    expected = (q @ doc.T).max(axis=1).sum()
    assert r._score(q, doc) == pytest.approx(expected)


def test_dense_cosine_score(tmp_path):
    r = DummyEmbedderRetriever(index_path=str(tmp_path / "i.pkl"), multi_vector=False)
    q = np.array([1.0, 0.0], dtype=np.float32)
    doc = np.array([1.0, 0.0], dtype=np.float32)
    assert r._score(q, doc) == pytest.approx(1.0)
    orthogonal = np.array([0.0, 1.0], dtype=np.float32)
    assert r._score(q, orthogonal) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def test_index_counts_and_payloads(indexed_retriever):
    assert len(indexed_retriever) == 3
    payloads = indexed_retriever._payloads
    # metadata is merged into the payload alongside bookkeeping keys
    assert {"file", "format", "page", "document", "document_version"} <= set(
        payloads[0]
    )
    assert any(p.get("year") == 2024 for p in payloads)
    assert all(p["page"] >= 1 for p in payloads)


def test_supported_extensions_match_extractors(indexed_retriever):
    assert indexed_retriever.supported_extensions == tuple(IMAGE_EXTRACTORS.keys())
    assert repr(indexed_retriever).startswith("<DummyEmbedderRetriever pages=3")


def test_index_skips_unreadable_documents(DummyRetriever, tmp_path):
    retriever = DummyRetriever(index_path=str(tmp_path / "i.pkl"))
    ghost = InputDocument.from_path(tmp_path / "does_not_exist.png")
    retriever.index([ghost], save=False)
    assert len(retriever) == 0  # the missing file was skipped, not crashed


def test_index_without_save_does_not_persist(DummyRetriever, sample_files, tmp_path):
    idx = tmp_path / "i.pkl"
    retriever = DummyRetriever(index_path=str(idx))
    retriever.index([InputDocument.from_path(sample_files["report"])], save=False)
    assert not idx.exists()


def test_index_reinit_false_appends(DummyRetriever, sample_files, tmp_path):
    retriever = DummyRetriever(index_path=str(tmp_path / "i.pkl"))
    retriever.index([InputDocument.from_path(sample_files["report"])], save=False)
    retriever.index(
        [InputDocument.from_path(sample_files["memo"])], reinit=False, save=False
    )
    assert len(retriever) == 2


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def test_search_is_sorted_descending(indexed_retriever):
    results = indexed_retriever.search(Query(query_text="annual report"), top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_top_k_limits_results(indexed_retriever):
    assert len(indexed_retriever.search(Query(query_text="x"), top_k=1)) == 1
    assert len(indexed_retriever.search(Query(query_text="x"), top_k=2)) == 2


def test_search_metadata_filter_masks_results(indexed_retriever):
    query = Query.of("anything").with_filter("year", 2024, "gte")
    results = indexed_retriever.search(query, top_k=10)
    files = sorted({r.document_version.filename for r in results})
    assert files == ["memo.jpg", "report.png"]
    assert all(r.normalized_score >= 0.0 for r in results)


def test_search_in_filter(indexed_retriever):
    query = Query.of("anything", filters={"lang": ["fr"]})
    results = indexed_retriever.search(query, top_k=10)
    assert {r.document_version.filename for r in results} == {"memo.jpg"}


def test_search_empty_index_raises(DummyRetriever, tmp_path):
    retriever = DummyRetriever(index_path=str(tmp_path / "i.pkl"))
    with pytest.raises(RuntimeError):
        retriever.search(Query(query_text="x"))


def test_search_returns_preannotation_results(indexed_retriever):
    results = indexed_retriever.search(Query(query_text="x"), top_k=1)
    result = results[0]
    assert result.page >= 1
    assert isinstance(result.score, float)
    # show() consumes the converted annotation triples without raising.
    indexed_retriever.show([r.to_annotation_result() for r in results])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_index_is_persisted_and_reloaded(DummyRetriever, sample_files, tmp_path):
    idx = tmp_path / "index.pkl"
    first = DummyRetriever(index_path=str(idx))
    first.index(
        [InputDocument.from_path(sample_files["report"], metadata={"year": 2024})]
    )
    assert idx.exists()

    # A fresh instance pointing at the same path auto-loads in __init__.
    second = DummyRetriever(index_path=str(idx))
    assert len(second) == 1
    assert second._payloads[0]["year"] == 2024


def test_save_to_explicit_path(indexed_retriever, tmp_path):
    other = tmp_path / "other.pkl"
    indexed_retriever.save_index(other)
    assert other.exists()


def test_reinit_clears_index(indexed_retriever):
    assert len(indexed_retriever) == 3
    indexed_retriever.reinit_indexes()
    assert len(indexed_retriever) == 0


# ---------------------------------------------------------------------------
# Dense (single-vector) variant
# ---------------------------------------------------------------------------
def test_dense_retriever_full_pipeline(DummyRetriever, sample_files, tmp_path):
    retriever = DummyRetriever(index_path=str(tmp_path / "i.pkl"), multi_vector=False)
    retriever.index(
        [InputDocument.from_path(p) for p in sample_files.values()], save=False
    )
    assert len(retriever) == 3
    results = retriever.search(Query(query_text="report"), top_k=2)
    assert len(results) == 2
    assert all(-1.0001 <= r.score <= 1.0001 for r in results)  # cosine range


# ---------------------------------------------------------------------------
# Concrete backend declarations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cls,multi_vector,default_model_substr",
    [
        (ColPaliRetriever, True, "colpali"),
        (ColQwen2Retriever, True, "colqwen2"),
        (NomicDenseRetriever, False, "nomic"),
        (TomoroColQwen3Retriever, True, "colqwen3"),
        (ColModernVBertRetriever, True, "colmodernvbert"),
        (ModernVBertRetriever, False, "modernvbert"),
    ],
)
def test_concrete_retriever_declarations(cls, multi_vector, default_model_substr):
    assert cls.multi_vector is multi_vector
    default = inspect.signature(cls.__init__).parameters["model_name"].default
    assert default_model_substr in default.lower()
