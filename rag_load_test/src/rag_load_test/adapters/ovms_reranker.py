"""RerankerPort over OpenVINO Model Server's Cohere-compatible ``/v3/rerank``.

Request/response shapes follow the OVMS docs (spec section 11):
``{"model", "query", "documents", "top_n"}`` ->
``{"results": [{"index", "relevance_score"}]}``. Each result's ``index`` points
back into the passages we sent, so ordering is recomputed client-side.

Example::

    reranker = OvmsReranker(
        httpx.AsyncClient(), "http://localhost:8001", "bge-reranker-base",
        token_provider=NoAuth(), timeout_s=30, max_retries=2,
    )
    best = await reranker.rerank("what is rag?", candidates, top_n=5)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from ..contracts.models import ScoredPassage
from ..errors import RagDependencyError
from .auth import BearerTokenProvider
from .http_retry import excerpt, post_json_with_retry, probe_ready

COMPONENT = "reranker"


class OvmsReranker:
    """Re-score passages with an OVMS rerank servable and keep the best ``top_n``."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        *,
        token_provider: BearerTokenProvider,
        timeout_s: float,
        max_retries: int,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip(
            "/"
        )  # tolerate RAG_OVMS_RERANK_URL=http://h:8001/
        self._model = model
        self._token_provider = token_provider
        self._timeout_s = timeout_s
        self._max_retries = max_retries

    @property
    def model_name(self) -> str:
        return f"ovms:{self._model}"

    @property
    def rerank_url(self) -> str:
        return f"{self._base_url}/v3/rerank"

    @property
    def status_url(self) -> str:
        return f"{self._base_url}/v3/models/{self._model}"

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n!r}")
        if not passages:
            return []  # nothing to score: skip the round-trip entirely
        payload = {
            "model": self._model,
            "query": query,
            "documents": [p.text for p in passages],
            "top_n": top_n,
        }
        body = await post_json_with_retry(
            self._client,
            self.rerank_url,
            payload,
            component=COMPONENT,
            token_provider=self._token_provider,
            timeout_s=self._timeout_s,
            max_retries=self._max_retries,
        )
        rescored = _apply_scores(list(passages), body, target=self.rerank_url)
        rescored.sort(key=lambda p: p.rerank_score or 0.0, reverse=True)
        return rescored[:top_n]

    async def ready(self) -> tuple[bool, str]:
        return await probe_ready(
            self._client,
            self.status_url,
            token_provider=self._token_provider,
            timeout_s=self._timeout_s,
        )


def _apply_scores(
    passages: list[ScoredPassage], body: dict[str, Any], *, target: str
) -> list[ScoredPassage]:
    """Copy each passage named by ``results[i].index`` with its ``relevance_score``."""
    results = body.get("results")
    if not isinstance(results, list):
        detail = f"expected a 'results' list, got: {excerpt(body)}"
        raise RagDependencyError(COMPONENT, detail, target=target)
    try:
        return [_rescore(passages, row) for row in results]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        detail = f"malformed rerank result ({exc!r}) in: {excerpt(results)}"
        raise RagDependencyError(COMPONENT, detail, target=target) from exc


def _rescore(passages: list[ScoredPassage], row: dict[str, Any]) -> ScoredPassage:
    index = int(row["index"])
    if index < 0:  # a negative index would silently wrap around in Python
        raise IndexError(f"index {index} out of range 0..{len(passages) - 1}")
    score = float(row["relevance_score"])
    return passages[index].model_copy(update={"rerank_score": score})
