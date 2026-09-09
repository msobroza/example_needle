"""SentenceTransformerEmbedder / CrossEncoderReranker with injected fake models.

No weights are loaded: the fakes mimic the ``encode``/``predict`` surface of
sentence-transformers and return numpy arrays, exactly like the real classes.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import types
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

from rag_load_test.adapters.executor import build_model_executor, run_blocking
from rag_load_test.adapters.local_embedder import SentenceTransformerEmbedder
from rag_load_test.adapters.local_reranker import CrossEncoderReranker
from rag_load_test.contracts import EmbedderPort, RerankerPort, ScoredPassage


class _FakeSt:
    """Stands in for ``sentence_transformers.SentenceTransformer``."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def encode(self, texts: list[str], **kw: Any) -> np.ndarray:
        self.calls.append((list(texts), kw))
        return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)


class _FakeCe:
    """Stands in for ``sentence_transformers.CrossEncoder``; longer text scores higher."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[tuple[str, str]], dict[str, Any]]] = []

    def predict(self, pairs: list[tuple[str, str]], **kw: Any) -> np.ndarray:
        self.calls.append((list(pairs), kw))
        return np.array([float(len(p[1])) for p in pairs], dtype=np.float32)


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    pool = build_model_executor(2)
    yield pool
    pool.shutdown(wait=True)


def test_build_model_executor_is_bounded_and_named(executor: ThreadPoolExecutor):
    assert isinstance(executor, ThreadPoolExecutor)
    assert executor._max_workers == 2
    assert executor._thread_name_prefix == "rag-model"


def test_build_model_executor_rejects_non_positive_workers():
    with pytest.raises(ValueError, match="0"):
        build_model_executor(0)


async def test_run_blocking_runs_in_executor_thread(executor: ThreadPoolExecutor):
    def whoami(a: int, b: int) -> tuple[str, int]:
        return threading.current_thread().name, a + b

    name, total = await run_blocking(executor, whoami, 1, 2)
    assert name.startswith("rag-model")
    assert total == 3


async def test_st_embedder_normalises_and_returns_lists(executor: ThreadPoolExecutor):
    model = _FakeSt()
    emb = SentenceTransformerEmbedder(
        model, executor, model_name="fake-st", batch_size=8
    )
    out = await emb.embed_documents(["a", "abc"])
    assert out == [[1.0, 1.0], [3.0, 1.0]]
    # Plain python floats (not numpy scalars) so the vectors are JSON/Chroma safe.
    assert all(type(v) is float for row in out for v in row)
    texts, kw = model.calls[0]
    assert texts == ["a", "abc"]
    assert kw["normalize_embeddings"] is True
    assert kw["convert_to_numpy"] is True
    assert kw["batch_size"] == 8
    assert await emb.embed_documents([]) == []
    assert len(model.calls) == 1  # empty input never reaches the model


async def test_st_embed_query_returns_single_vector(executor: ThreadPoolExecutor):
    emb = SentenceTransformerEmbedder(_FakeSt(), executor, model_name="fake-st")
    assert await emb.embed_query("ab") == [2.0, 1.0]


async def test_ce_reranker_orders_and_truncates(executor: ThreadPoolExecutor):
    reranker = CrossEncoderReranker(_FakeCe(), executor, model_name="fake-ce")
    passages = [
        ScoredPassage(id="short", text="ab", retrieval_score=0.9),
        ScoredPassage(id="long", text="abcdef", retrieval_score=0.1),
        ScoredPassage(id="mid", text="abcd", retrieval_score=0.5),
    ]
    out = await reranker.rerank("q", passages, top_n=2)
    assert [p.id for p in out] == ["long", "mid"]
    assert out[0].rerank_score == 6.0
    assert type(out[0].rerank_score) is float
    assert out[0].retrieval_score == 0.1  # retrieval score preserved
    assert passages[1].rerank_score is None  # inputs are not mutated
    assert await reranker.rerank("q", [], top_n=2) == []


async def test_ce_reranker_passes_query_text_pairs_and_batch_size(
    executor: ThreadPoolExecutor,
):
    model = _FakeCe()
    reranker = CrossEncoderReranker(model, executor, model_name="fake-ce", batch_size=4)
    passage = ScoredPassage(id="a", text="t", retrieval_score=1.0)
    await reranker.rerank("q", [passage], top_n=1)
    pairs, kw = model.calls[0]
    assert pairs == [("q", "t")]
    assert kw["batch_size"] == 4


async def test_ready_reports_model_name(executor: ThreadPoolExecutor):
    emb = SentenceTransformerEmbedder(_FakeSt(), executor, model_name="fake-st")
    reranker = CrossEncoderReranker(_FakeCe(), executor, model_name="fake-ce")
    assert emb.model_name == "fake-st"
    assert await emb.ready() == (True, "fake-st")
    assert reranker.model_name == "fake-ce"
    assert await reranker.ready() == (True, "fake-ce")


def test_local_adapters_satisfy_ports(executor: ThreadPoolExecutor):
    emb = SentenceTransformerEmbedder(_FakeSt(), executor, model_name="x")
    reranker = CrossEncoderReranker(_FakeCe(), executor, model_name="x")
    assert isinstance(emb, EmbedderPort)
    assert isinstance(reranker, RerankerPort)


def test_from_pretrained_imports_lazily_and_pins_cpu(
    executor: ThreadPoolExecutor, monkeypatch: pytest.MonkeyPatch
):
    built: list[tuple[str, dict[str, Any]]] = []

    def _fake_ctor(name: str, **kw: Any) -> _FakeSt:
        built.append((name, kw))
        return _FakeSt()

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _fake_ctor  # type: ignore[attr-defined]
    fake_module.CrossEncoder = _fake_ctor  # type: ignore[attr-defined]
    # Only a lazy (inside-the-method) import can pick up this substituted module.
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    emb = SentenceTransformerEmbedder.from_pretrained("st-id", executor, batch_size=3)
    reranker = CrossEncoderReranker.from_pretrained("ce-id", executor, batch_size=5)
    assert emb.model_name == "st-id"
    assert reranker.model_name == "ce-id"
    assert built == [("st-id", {"device": "cpu"}), ("ce-id", {"device": "cpu"})]


def test_importing_local_adapters_does_not_import_heavy_libraries():
    code = (
        "import sys, rag_load_test.adapters.local_embedder, "
        "rag_load_test.adapters.local_reranker, rag_load_test.adapters.executor; "
        "heavy = {'sentence_transformers', 'torch', 'transformers'} & set(sys.modules); "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
