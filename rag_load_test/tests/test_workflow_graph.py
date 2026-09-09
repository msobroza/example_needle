"""LangGraph workflow exercised end to end with fakes only (no weights, no network)."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import pytest

from rag_load_test.contracts.models import Passage, RagResult, ScoredPassage
from rag_load_test.testing import (
    FakeChatModel,
    FakeEmbedder,
    FakeReranker,
    InMemoryVectorStore,
)
from rag_load_test.workflow.graph import (
    STAGES,
    build_rag_graph,
    route_after_rerank,
    run_rag,
)
from rag_load_test.workflow.nodes import RagDependencies, timed_node
from rag_load_test.workflow.prompts import SYSTEM_PROMPT, build_rag_messages
from rag_load_test.workflow.state import merge_timings

QUESTION = "how many vacation days do employees get"

CORPUS: list[tuple[str, str]] = [
    ("p1", "employees get 25 vacation days per year"),
    ("p2", "vacation requests must be approved by a manager"),
    ("p3", "the cafeteria serves lunch from noon to two"),
    ("p4", "parking permits are issued by facilities"),
    ("p5", "remote work policy allows three days per week"),
    ("p6", "expense reports are due by the fifth of the month"),
]


class SpyVectorStore(InMemoryVectorStore):
    """InMemoryVectorStore that records every ``top_k`` it was queried with."""

    def __init__(self) -> None:
        super().__init__()
        self.top_ks: list[int] = []

    async def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[ScoredPassage]:
        self.top_ks.append(top_k)
        return await super().query(embedding, top_k)


@pytest.fixture
async def deps() -> RagDependencies:
    embedder = FakeEmbedder()
    store = SpyVectorStore()
    passages = [Passage(id=pid, text=text) for pid, text in CORPUS]
    vectors = await embedder.embed_documents([p.text for p in passages])
    await store.upsert(passages, vectors)
    return RagDependencies(
        embedder=embedder,
        vector_store=store,
        reranker=FakeReranker(),
        chat_model=FakeChatModel(),
        top_k_retrieve=4,
        top_k_rerank=3,
    )


async def test_query_mode_runs_all_stages_and_returns_answer(deps: RagDependencies):
    result = await run_rag(build_rag_graph(deps), QUESTION)

    assert isinstance(result, RagResult)
    assert result.question == QUESTION
    assert result.mode == "query"
    assert result.answer is not None and result.answer.startswith("[fake-llm]")
    assert deps.chat_model.calls == 1  # type: ignore[attr-defined]
    assert len(result.passages) == deps.top_k_rerank
    assert all(p.rerank_score is not None for p in result.passages)

    stage_ms = result.timings.as_dict()
    assert set(stage_ms) == set(STAGES) | {"total"}
    assert all(stage_ms[s] >= 0.0 for s in STAGES)
    assert result.timings.total_ms > 0.0
    # perf_counter is monotonic and the stage intervals nest inside ainvoke.
    assert result.timings.total_ms >= sum(stage_ms[s] for s in STAGES) - 0.01


async def test_retrieve_mode_skips_generate(deps: RagDependencies):
    result = await run_rag(build_rag_graph(deps), QUESTION, mode="retrieve")

    assert result.mode == "retrieve"
    assert result.answer is None
    assert deps.chat_model.calls == 0  # type: ignore[attr-defined]
    assert result.timings.generate_ms == 0.0
    assert len(result.passages) == deps.top_k_rerank
    assert result.timings.embed_ms >= 0.0
    assert result.timings.retrieve_ms >= 0.0
    assert result.timings.rerank_ms >= 0.0
    assert result.timings.total_ms > 0.0


async def test_rerank_orders_by_rerank_score_and_truncates(deps: RagDependencies):
    result = await run_rag(
        build_rag_graph(deps), "vacation days policy", rerank_top_k=2
    )

    assert len(result.passages) == 2
    scores = [p.rerank_score for p in result.passages]
    assert all(s is not None for s in scores)
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]
    assert result.passages[0].id in {"p1", "p2"}


async def test_reranker_none_passes_top_candidates_through(deps: RagDependencies):
    no_reranker = dataclasses.replace(deps, reranker=None)
    result = await run_rag(build_rag_graph(no_reranker), QUESTION)

    embedding = await deps.embedder.embed_query(QUESTION)
    candidates = await deps.vector_store.query(embedding, deps.top_k_retrieve)
    expected = candidates[: deps.top_k_rerank]

    assert [p.id for p in result.passages] == [p.id for p in expected]
    assert all(p.rerank_score is None for p in result.passages)
    assert result.timings.rerank_ms >= 0.0
    assert result.answer is not None and result.answer.startswith("[fake-llm]")


async def test_top_k_override_flows_to_store(deps: RagDependencies):
    store = deps.vector_store
    assert isinstance(store, SpyVectorStore)
    graph = build_rag_graph(deps)

    overridden = await run_rag(graph, QUESTION, top_k=2)
    assert store.top_ks == [2]
    # Only two candidates exist, so rerank cannot return more than two.
    assert len(overridden.passages) == 2

    await run_rag(graph, QUESTION)
    assert store.top_ks == [2, deps.top_k_retrieve]


def test_timings_merge_reducer():
    assert merge_timings({"a": 1.0}, {"b": 2.0}) == {"a": 1.0, "b": 2.0}
    assert merge_timings({"a": 1.0}, {"a": 3.0}) == {"a": 3.0}
    left, right = {"a": 1.0}, {"b": 2.0}
    merge_timings(left, right)
    assert left == {"a": 1.0} and right == {"b": 2.0}


def test_build_rag_messages_numbers_passages():
    passages = [
        ScoredPassage(id="x", text="first passage", retrieval_score=0.9),
        ScoredPassage(id="y", text="second passage", retrieval_score=0.8),
    ]
    messages = build_rag_messages("what is it?", passages)

    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"
    user = messages[1]["content"]
    assert "[1] first passage" in user
    assert "[2] second passage" in user
    assert user.index("[1]") < user.index("[2]")
    assert user.endswith("Question: what is it?")
    assert len(messages) == 2
    assert "[n]" in SYSTEM_PROMPT


def test_route_after_rerank_only_generates_in_query_mode():
    from langgraph.graph import END

    assert route_after_rerank({"mode": "query"}) == "generate"
    assert route_after_rerank({"mode": "retrieve"}) == END
    assert route_after_rerank({}) == END


async def test_timed_node_adds_stage_timing_and_keeps_result():
    async def body(state: dict[str, Any]) -> dict[str, Any]:
        return {"answer": state["question"].upper()}

    out = await timed_node("generate", body)({"question": "hi"})

    assert out["answer"] == "HI"
    assert set(out["timings_ms"]) == {"generate"}
    assert out["timings_ms"]["generate"] >= 0.0


def test_graph_exposes_planned_node_names(deps: RagDependencies):
    assert STAGES == ("embed", "retrieve", "rerank", "generate")
    node_names = set(build_rag_graph(deps).get_graph().nodes)
    assert {"embed_query", "retrieve", "rerank", "generate"} <= node_names
