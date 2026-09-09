"""Node factories for the RAG graph. Nodes see only ports, never concrete adapters.

``RagDependencies`` lives here (not in ``graph.py``) so the API wiring can build
it without importing LangGraph.

Example::

    deps = RagDependencies(
        embedder=emb, vector_store=store, reranker=None, chat_model=llm
    )
    embed = timed_node("embed", make_embed_node(deps))
    update = await embed({"question": "vacation days?"})
    # {"query_embedding": [...], "timings_ms": {"embed": 0.4}}
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..contracts.ports import ChatModelPort, EmbedderPort, RerankerPort, VectorStorePort
from .prompts import build_rag_messages
from .state import RagState


@dataclass
class RagDependencies:
    """Ports plus retrieval defaults that every node closes over.

    ``reranker=None`` means "no reranking": the rerank node passes the top
    candidates through unchanged so topology comparisons can isolate its cost.

    Example::

        RagDependencies(embedder=emb, vector_store=store, reranker=rr, chat_model=llm,
                        top_k_retrieve=20, top_k_rerank=5)
    """

    embedder: EmbedderPort
    vector_store: VectorStorePort
    reranker: RerankerPort | None
    chat_model: ChatModelPort
    top_k_retrieve: int = 20
    top_k_rerank: int = 5


NodeFn = Callable[[RagState], Awaitable[dict[str, Any]]]


def timed_node(stage: str, fn: NodeFn) -> NodeFn:
    """Wrap ``fn`` so its state update also carries ``{"timings_ms": {stage: ms}}``.

    The per-stage dict is merged into the graph state by the ``merge_timings``
    reducer declared on ``RagState.timings_ms``.

    Example::

        node = timed_node("embed", make_embed_node(deps))
    """

    async def _timed(state: RagState) -> dict[str, Any]:
        started = time.perf_counter()
        result = await fn(state)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {**result, "timings_ms": {stage: elapsed_ms}}

    _timed.__name__ = f"timed_{stage}"
    return _timed


def make_embed_node(deps: RagDependencies) -> NodeFn:
    """Node: ``question`` -> ``query_embedding``.

    Example::

        embed = make_embed_node(deps)
    """

    async def embed_query(state: RagState) -> dict[str, Any]:
        return {"query_embedding": await deps.embedder.embed_query(state["question"])}

    return embed_query


def make_retrieve_node(deps: RagDependencies) -> NodeFn:
    """Node: ``query_embedding`` -> ``candidates`` (``top_k`` overrides the default).

    Example::

        retrieve = make_retrieve_node(deps)
    """

    async def retrieve(state: RagState) -> dict[str, Any]:
        top_k = state.get("top_k") or deps.top_k_retrieve
        candidates = await deps.vector_store.query(state["query_embedding"], top_k)
        return {"candidates": candidates}

    return retrieve


def make_rerank_node(deps: RagDependencies) -> NodeFn:
    """Node: ``candidates`` -> ``reranked``; passthrough when there is no reranker.

    Example::

        rerank = make_rerank_node(deps)
    """

    async def rerank(state: RagState) -> dict[str, Any]:
        top_n = state.get("rerank_top_k") or deps.top_k_rerank
        candidates = state.get("candidates", [])
        if deps.reranker is None:
            return {"reranked": list(candidates[:top_n])}
        reranked = await deps.reranker.rerank(state["question"], candidates, top_n)
        return {"reranked": reranked}

    return rerank


def make_generate_node(deps: RagDependencies) -> NodeFn:
    """Node: ``question`` + ``reranked`` -> ``answer`` via the chat model.

    Example::

        generate = make_generate_node(deps)
    """

    async def generate(state: RagState) -> dict[str, Any]:
        messages = build_rag_messages(state["question"], state.get("reranked", []))
        return {"answer": await deps.chat_model.generate(messages)}

    return generate
