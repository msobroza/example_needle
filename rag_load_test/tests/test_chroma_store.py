"""ChromaVectorStore against a real chromadb ``EphemeralClient`` (no network)."""

from __future__ import annotations

import subprocess
import sys
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rag_load_test.adapters.chroma_store import EMBEDDER_MODEL_KEY, ChromaVectorStore
from rag_load_test.adapters.executor import build_model_executor
from rag_load_test.contracts import Passage, ScoredPassage, VectorStorePort
from rag_load_test.testing import FakeEmbedder


def _collection_name() -> str:
    # Chroma requires 3..512 chars; EphemeralClient state is process-wide, so
    # every test gets its own collection to stay independent.
    return f"test-{uuid.uuid4().hex[:8]}"


def _passages() -> list[Passage]:
    return [
        Passage(
            id="a",
            text="employees get 25 vacation days",
            metadata={"doc": "hr", "page": 3, "weight": 0.5, "public": True},
        ),
        Passage(id="b", text="the cafeteria serves lunch", metadata={"doc": "food"}),
    ]


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    pool = build_model_executor(2)
    yield pool
    pool.shutdown(wait=True)


@pytest.fixture
def store(executor: ThreadPoolExecutor) -> ChromaVectorStore:
    return ChromaVectorStore.open(
        ":memory:", _collection_name(), executor, embedder_model="fake-embedder"
    )


async def _fill(store: ChromaVectorStore, times: int = 1) -> list[Passage]:
    passages = _passages()
    vectors = await FakeEmbedder().embed_documents([p.text for p in passages])
    for _ in range(times):
        await store.upsert(passages, vectors)
    return passages


def test_chroma_store_satisfies_port(store: ChromaVectorStore):
    assert isinstance(store, VectorStorePort)
    assert EMBEDDER_MODEL_KEY == "embedder_model"


async def test_chroma_round_trip_idempotent_and_ordered(store: ChromaVectorStore):
    passages = await _fill(store, times=2)
    assert await store.count() == 2
    hits = await store.query(await FakeEmbedder().embed_query("vacation days"), top_k=5)
    assert [h.id for h in hits] == ["a", "b"]
    assert isinstance(hits[0], ScoredPassage)
    assert hits[0].text == passages[0].text
    assert hits[0].metadata == passages[0].metadata  # "passage_id" stripped
    assert hits[1].metadata == {"doc": "food"}
    assert hits[0].retrieval_score > hits[1].retrieval_score
    assert all(-1.0 <= h.retrieval_score <= 1.0 + 1e-6 for h in hits)
    assert all(h.rerank_score is None for h in hits)
    assert store.embedder_model() == "fake-embedder"


async def test_chroma_query_on_empty_collection_returns_empty(store: ChromaVectorStore):
    assert await store.count() == 0
    assert await store.query([0.0] * 16, top_k=5) == []


async def test_chroma_query_clamps_top_k(store: ChromaVectorStore):
    await _fill(store)
    hits = await store.query(await FakeEmbedder().embed_query("lunch"), top_k=10)
    assert len(hits) == 2
    assert (
        await store.query(await FakeEmbedder().embed_query("lunch"), top_k=1)
        == hits[:1]
    )


async def test_chroma_upsert_empty_is_noop(store: ChromaVectorStore):
    await store.upsert([], [])
    assert await store.count() == 0


async def test_chroma_upsert_rejects_length_mismatch(store: ChromaVectorStore):
    with pytest.raises(ValueError, match="mismatch"):
        await store.upsert([Passage(id="a", text="x")], [])


async def test_chroma_upsert_accepts_passages_without_metadata(
    store: ChromaVectorStore,
):
    # Chroma rejects empty metadata dicts; the store must still accept bare passages.
    await store.upsert([Passage(id="bare", text="no metadata")], [[1.0] * 16])
    hits = await store.query([1.0] * 16, top_k=1)
    assert hits[0].id == "bare"
    assert hits[0].metadata == {}


def test_chroma_open_without_embedder_model_records_none(executor: ThreadPoolExecutor):
    store = ChromaVectorStore.open(":memory:", _collection_name(), executor)
    assert store.embedder_model() is None


async def test_chroma_persistent_client_survives_reopen(
    tmp_path: Path, executor: ThreadPoolExecutor
):
    name = _collection_name()
    path = str(tmp_path / "chroma")
    first = ChromaVectorStore.open(path, name, executor, embedder_model="bge")
    await _fill(first)
    second = ChromaVectorStore.open(path, name, executor, embedder_model="bge")
    assert await second.count() == 2
    assert second.embedder_model() == "bge"


def test_importing_chroma_store_does_not_import_chromadb():
    code = (
        "import sys, rag_load_test.adapters.chroma_store; "
        "assert 'chromadb' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
