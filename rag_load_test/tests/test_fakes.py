from __future__ import annotations

import pytest

from rag_load_test.contracts import (
    ChatModelPort,
    EmbedderPort,
    Passage,
    RerankerPort,
    ScoredPassage,
    VectorStorePort,
)
from rag_load_test.testing import (
    FakeChatModel,
    FakeEmbedder,
    FakeReranker,
    InMemoryVectorStore,
)


def test_fakes_satisfy_ports():
    assert isinstance(FakeEmbedder(), EmbedderPort)
    assert isinstance(FakeReranker(), RerankerPort)
    assert isinstance(InMemoryVectorStore(), VectorStorePort)
    assert isinstance(FakeChatModel(), ChatModelPort)


async def test_fake_embedder_is_deterministic_and_unit_norm():
    emb = FakeEmbedder(dim=8)
    a = await emb.embed_query("vacation policy")
    b = await emb.embed_query("vacation policy")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6
    assert len(a) == 8


async def test_fake_embedder_similar_texts_are_closer():
    emb = FakeEmbedder()
    q = await emb.embed_query("how many vacation days do employees get")
    docs = await emb.embed_documents(
        ["employees get 25 vacation days per year", "the cafeteria serves lunch"]
    )
    from rag_load_test.testing.fakes import cosine

    assert cosine(q, docs[0]) > cosine(q, docs[1])


async def test_in_memory_store_upsert_is_idempotent_and_orders_by_similarity():
    emb = FakeEmbedder()
    store = InMemoryVectorStore()
    passages = [
        Passage(id="a", text="employees get 25 vacation days", metadata={"doc": "hr"}),
        Passage(id="b", text="the cafeteria serves lunch", metadata={"doc": "food"}),
    ]
    vectors = await emb.embed_documents([p.text for p in passages])
    await store.upsert(passages, vectors)
    await store.upsert(passages, vectors)
    assert await store.count() == 2
    hits = await store.query(await emb.embed_query("vacation days"), top_k=5)
    assert [h.id for h in hits] == ["a", "b"]
    assert isinstance(hits[0], ScoredPassage)
    assert hits[0].metadata == {"doc": "hr"}


async def test_in_memory_store_rejects_length_mismatch():
    store = InMemoryVectorStore()
    with pytest.raises(ValueError, match="mismatch"):
        await store.upsert([Passage(id="a", text="x")], [])


async def test_fake_reranker_scores_overlap_and_truncates():
    passages = [
        ScoredPassage(id="1", text="cafeteria lunch hours", retrieval_score=0.9),
        ScoredPassage(id="2", text="vacation days policy", retrieval_score=0.8),
        ScoredPassage(id="3", text="unrelated", retrieval_score=0.7),
    ]
    out = await FakeReranker().rerank("vacation policy", passages, top_n=2)
    assert [p.id for p in out] == ["2", "1"]
    assert out[0].rerank_score and out[0].rerank_score > (out[1].rerank_score or 0)
    assert out[0].retrieval_score == 0.8
