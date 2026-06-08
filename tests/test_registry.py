"""Tests for the retriever registry/factory."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from needle.retrieval import registry  # noqa: E402
from needle.retrieval.page_retrievers import MultimodalEmbedderRetriever  # noqa: E402
from needle.testing import DummyEmbedderRetriever  # noqa: E402

pytestmark = pytest.mark.torch


def test_available_retrievers_is_sorted_and_complete():
    names = registry.available_retrievers()
    assert names == sorted(names)
    assert {
        "colpali",
        "colqwen2",
        "nomic",
        "tomoro-colqwen3",
        "colmodernvbert",
        "modernvbert",
    } <= set(names)


def test_all_registered_values_are_retriever_subclasses():
    for cls in registry.RETRIEVERS.values():
        assert issubclass(cls, MultimodalEmbedderRetriever)


def test_aliases_point_to_real_canonical_names():
    for alias, canonical in registry._ALIASES.items():
        assert canonical in registry.RETRIEVERS, f"alias {alias!r} -> {canonical!r}"


def test_get_retriever_unknown_raises():
    with pytest.raises(KeyError):
        registry.get_retriever("does-not-exist")


def test_get_retriever_instantiates_via_name(monkeypatch, tmp_path):
    monkeypatch.setitem(registry.RETRIEVERS, "dummy", DummyEmbedderRetriever)
    retriever = registry.get_retriever("dummy", index_path=str(tmp_path / "i.pkl"))
    assert isinstance(retriever, DummyEmbedderRetriever)


def test_get_retriever_resolves_alias(monkeypatch, tmp_path):
    monkeypatch.setitem(registry.RETRIEVERS, "dummy", DummyEmbedderRetriever)
    monkeypatch.setitem(registry._ALIASES, "dummy-alias", "dummy")
    retriever = registry.get_retriever(
        "DUMMY-ALIAS", index_path=str(tmp_path / "i.pkl")
    )
    assert isinstance(retriever, DummyEmbedderRetriever)
