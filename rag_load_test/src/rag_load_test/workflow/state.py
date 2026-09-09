"""Graph state shared by every workflow node.

Example::

    state: RagState = {"question": "vacation days?", "mode": "query", "timings_ms": {}}
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from ..contracts.models import ScoredPassage


def merge_timings(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    """LangGraph reducer for ``timings_ms``: each node adds its own stage key.

    Without a reducer LangGraph would replace the dict on every node update and
    only the last stage would survive. Right-hand keys win on conflict.

    Example::

        merge_timings({"embed": 1.5}, {"retrieve": 2.0})
        # {"embed": 1.5, "retrieve": 2.0}
    """
    return {**left, **right}


class RagState(TypedDict, total=False):
    question: str
    mode: str
    top_k: int
    rerank_top_k: int
    query_embedding: list[float]
    candidates: list[ScoredPassage]
    reranked: list[ScoredPassage]
    answer: str | None
    timings_ms: Annotated[dict[str, float], merge_timings]
