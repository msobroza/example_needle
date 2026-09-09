from __future__ import annotations

import pytest

from rag_load_test.errors import RagConfigError
from rag_load_test.settings import RagSettings


def test_defaults_match_spec(settings_factory):
    s = settings_factory()
    assert s.topology == "monolith"
    assert s.embedder_backend == "local"
    assert s.reranker_backend == "local"
    assert s.llm_backend == "openai"
    assert s.ovms_rerank_url == "http://localhost:8001"
    assert s.ovms_embeddings_url == "http://localhost:8002"
    assert (s.top_k_retrieve, s.top_k_rerank, s.model_threads) == (20, 5, 4)
    assert (s.host, s.port) == ("0.0.0.0", 8888)


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("RAG_RERANKER_BACKEND", "ovms")
    monkeypatch.setenv("RAG_TOP_K_RERANK", "7")
    monkeypatch.setenv("RAG_DOMINO_TRACING", "true")
    s = RagSettings(_env_file=None)
    assert s.reranker_backend == "ovms"
    assert s.top_k_rerank == 7
    assert s.domino_tracing is True


def test_invalid_backend_reports_setting_value_and_expectation(monkeypatch):
    monkeypatch.setenv("RAG_RERANKER_BACKEND", "ovsm")
    with pytest.raises(RagConfigError) as excinfo:
        RagSettings.from_env(_env_file=None)
    err = excinfo.value
    assert err.setting == "RAG_RERANKER_BACKEND"
    assert err.got == "ovsm"
    assert "ovms" in str(err.expected)


def test_invalid_port_range(monkeypatch):
    monkeypatch.setenv("RAG_PORT", "70000")
    with pytest.raises(RagConfigError) as excinfo:
        RagSettings.from_env(_env_file=None)
    assert excinfo.value.setting == "RAG_PORT"
