"""Compile the RAG ``StateGraph`` and run it. FastAPI-free by design.

    embed_query -> retrieve -> rerank -> (mode == "query") -> generate -> END
                                     \\-> (mode == "retrieve") ----------> END

Example::

    graph = build_rag_graph(deps)                 # once, at startup
    result = await run_rag(graph, "vacation days?", mode="retrieve")
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..contracts.models import RagMode, RagResult, StageTimings
from .nodes import (
    RagDependencies,
    make_embed_node,
    make_generate_node,
    make_rerank_node,
    make_retrieve_node,
    timed_node,
)
from .state import RagState

STAGES = ("embed", "retrieve", "rerank", "generate")


def route_after_rerank(state: RagState) -> str:
    """Conditional edge: only ``mode == "query"`` continues to the LLM.

    Example::

        route_after_rerank({"mode": "retrieve"})  # END
    """
    return "generate" if state.get("mode") == "query" else END


def build_rag_graph(deps: RagDependencies) -> CompiledStateGraph:
    """Wire the four timed nodes into a compiled graph.

    Example::

        graph = build_rag_graph(deps)
        final_state = await graph.ainvoke({"question": "q?", "mode": "query"})
    """
    builder = StateGraph(RagState)
    builder.add_node("embed_query", timed_node("embed", make_embed_node(deps)))
    builder.add_node("retrieve", timed_node("retrieve", make_retrieve_node(deps)))
    builder.add_node("rerank", timed_node("rerank", make_rerank_node(deps)))
    builder.add_node("generate", timed_node("generate", make_generate_node(deps)))
    builder.add_edge(START, "embed_query")
    builder.add_edge("embed_query", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_conditional_edges(
        "rerank", route_after_rerank, {"generate": "generate", END: END}
    )
    builder.add_edge("generate", END)
    return builder.compile()


async def run_rag(
    graph: CompiledStateGraph,
    question: str,
    *,
    mode: RagMode = "query",
    top_k: int | None = None,
    rerank_top_k: int | None = None,
) -> RagResult:
    """Invoke the graph once and fold the state into a ``RagResult``.

    ``total_ms`` is wall time around ``ainvoke`` so it also covers LangGraph's
    own scheduling overhead, not just the sum of the stages.

    Example::

        result = await run_rag(graph, "vacation days?", top_k=10, rerank_top_k=3)
    """
    initial = _initial_state(question, mode, top_k, rerank_top_k)
    started = time.perf_counter()
    final: dict[str, Any] = await graph.ainvoke(initial)
    total_ms = (time.perf_counter() - started) * 1000.0
    answer = final.get("answer") if mode == "query" else None
    return RagResult(
        question=question,
        mode=mode,
        answer=answer,
        passages=list(final.get("reranked", [])),
        timings=_stage_timings(final.get("timings_ms", {}), total_ms),
    )


def _initial_state(
    question: str, mode: RagMode, top_k: int | None, rerank_top_k: int | None
) -> RagState:
    # Omit unset overrides so the nodes fall back to RagDependencies defaults.
    state: RagState = {"question": question, "mode": mode, "timings_ms": {}}
    if top_k is not None:
        state["top_k"] = top_k
    if rerank_top_k is not None:
        state["rerank_top_k"] = rerank_top_k
    return state


def _stage_timings(timings: dict[str, float], total_ms: float) -> StageTimings:
    per_stage = {f"{stage}_ms": timings.get(stage, 0.0) for stage in STAGES}
    return StageTimings(**per_stage, total_ms=total_ms)
