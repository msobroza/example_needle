"""Tests for the torch-free RetrievalPipeline."""

from __future__ import annotations

import pytest

from conversational_core.domain.exceptions import EmptyIndexError
from conversational_core.domain.interaction.query import Query
from needle.embedders import DeterministicEmbedder
from needle.factory import build_pipeline
from needle.indexing import PickleIndexStore
from needle.retrieval.data import InputDocument


@pytest.fixture
def documents(sample_files):
    return [
        InputDocument.from_path(
            sample_files["invoice"], metadata={"year": 2021, "lang": "en"}
        ),
        InputDocument.from_path(
            sample_files["report"], metadata={"year": 2024, "lang": "en"}
        ),
        InputDocument.from_path(
            sample_files["memo"], metadata={"year": 2024, "lang": "fr"}
        ),
    ]


def test_pipeline_index_and_search(documents):
    pipe = build_pipeline()
    pipe.index(documents)
    assert len(pipe) == 3

    results = pipe.search(Query.of("annual report"), top_k=3)
    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_pipeline_metadata_filter(documents):
    pipe = build_pipeline().index(documents)
    query = Query.of("anything").with_filter("year", 2024, "gte")
    results = pipe.search(query, top_k=10)
    assert sorted({r.document_version.filename for r in results}) == [
        "memo.jpg",
        "report.png",
    ]


def test_pipeline_top_k(documents):
    pipe = build_pipeline().index(documents)
    assert len(pipe.search(Query.of("x"), top_k=1)) == 1
    assert len(pipe.search(Query.of("x"), top_k=2)) == 2


def test_pipeline_empty_index_raises():
    pipe = build_pipeline()
    with pytest.raises(EmptyIndexError):
        pipe.search(Query.of("x"))


def test_pipeline_dense_embedder(documents):
    pipe = build_pipeline(embedder=DeterministicEmbedder(multi_vector=False))
    pipe.index(documents)
    results = pipe.search(Query.of("report"), top_k=2)
    assert len(results) == 2
    assert all(-1.0001 <= r.score <= 1.0001 for r in results)


def test_pipeline_with_pickle_store_round_trip(documents, tmp_path):
    store = PickleIndexStore()
    pipe = build_pipeline(store=store)
    pipe.index(documents)

    path = tmp_path / "pipe.pkl"
    pipe.store.save(path)

    reloaded = PickleIndexStore()
    reloaded.load(path)
    assert len(reloaded) == len(pipe) == 3


def test_pipeline_repr(documents):
    pipe = build_pipeline().index(documents)
    assert "RetrievalPipeline" in repr(pipe)
    assert "pages=3" in repr(pipe)
