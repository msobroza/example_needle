"""Tests for the application services and use-cases (torch-free)."""

from __future__ import annotations

import pytest

from conversational_core.domain.interaction.query import Query
from needle.application import (
    IndexingService,
    IndexRequest,
    RankingService,
    SearchRequest,
    SearchService,
    index_documents,
    search,
)
from needle.application.errors import EmptyQueryError
from needle.factory import build_pipeline
from needle.retrieval.data import InputDocument


@pytest.fixture
def retriever(sample_files):
    pipe = build_pipeline()
    docs = [
        InputDocument.from_path(sample_files["report"], metadata={"year": 2024}),
        InputDocument.from_path(sample_files["memo"], metadata={"year": 2024}),
    ]
    pipe.index(docs)
    return pipe


def test_indexing_service(png_factory):
    pipe = build_pipeline()
    docs = [InputDocument.from_path(png_factory("a.png", "white", "hello"))]
    result = IndexingService(pipe).execute(IndexRequest(documents=docs))
    assert result.num_documents == 1
    assert result.num_pages == 1


def test_search_service_times_and_returns_hits(retriever):
    result = SearchService(retriever).execute(
        SearchRequest(query=Query.of("report"), top_k=2)
    )
    assert len(result) == 2
    assert result.took_ms >= 0.0
    assert result.best is result.hits[0]
    assert not result.is_empty


def test_search_service_rejects_blank_query(retriever):
    query = Query.of("placeholder")
    query.query_text = "   "  # bypass constructor validation to hit the guard
    with pytest.raises(EmptyQueryError):
        SearchService(retriever).execute(SearchRequest(query=query))


def test_use_case_functions(retriever):
    search_result = search(retriever, Query.of("memo"), top_k=1)
    assert len(search_result) == 1


def test_index_use_case(sample_files):
    pipe = build_pipeline()
    result = index_documents(pipe, [InputDocument.from_path(sample_files["report"])])
    assert result.num_pages == 1


def test_ranking_service_minmax():
    assert RankingService.minmax([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]
    assert RankingService.minmax([5.0, 5.0]) == [0.0, 0.0]
    assert RankingService.minmax([]) == []


def test_ranking_service_rrf():
    list_a = ["x", "y", "z"]
    list_b = ["y", "x", "w"]
    fused = RankingService.fuse([list_a, list_b], key=lambda s: s, top_k=2)
    # 'x' and 'y' both appear near the top of both lists -> fused to the front.
    assert set(fused) == {"x", "y"}

    with pytest.raises(ValueError):
        RankingService.reciprocal_rank_fusion([list_a], key=lambda s: s, k=0)
