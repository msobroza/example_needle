"""Tests for the retriever/pipeline factory."""

from __future__ import annotations

import pytest

from needle.config import RetrieverConfig
from needle.factory import build_pipeline
from needle.pipeline import RetrievalPipeline


def test_build_pipeline_is_torch_free():
    pipe = build_pipeline()
    assert isinstance(pipe, RetrievalPipeline)
    assert len(pipe) == 0


def test_build_retriever_uses_registry(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from needle.retrieval import registry
    from needle.testing import DummyEmbedderRetriever

    monkeypatch.setitem(registry.RETRIEVERS, "dummy", DummyEmbedderRetriever)
    from needle.factory import build_retriever

    config = RetrieverConfig(name="dummy", index_path=str(tmp_path / "i.pkl"))
    retriever = build_retriever(config)
    assert isinstance(retriever, DummyEmbedderRetriever)
    assert str(retriever.index_path) == str(tmp_path / "i.pkl")
