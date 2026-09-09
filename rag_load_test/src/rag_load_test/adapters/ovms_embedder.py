"""EmbedderPort over OpenVINO Model Server's OpenAI-compatible ``/v3/embeddings``.

Request/response shapes follow the OVMS docs (spec section 11):
``{"model", "input": [...]}`` -> ``{"data": [{"index", "embedding"}]}``. Rows are
re-ordered by ``index`` so output order never depends on server order.

Example::

    embedder = OvmsEmbedder(
        httpx.AsyncClient(), "http://localhost:8002", "bge-small-en-v1.5",
        token_provider=NoAuth(), timeout_s=30, max_retries=2,
    )
    vectors = await embedder.embed_documents(["hello", "world"])
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from ..errors import RagDependencyError
from .auth import BearerTokenProvider
from .http_retry import excerpt, post_json_with_retry, probe_ready

COMPONENT = "embedder"


class OvmsEmbedder:
    """Embed texts by POSTing ``batch_size``-sized batches to an OVMS servable."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        *,
        token_provider: BearerTokenProvider,
        timeout_s: float,
        max_retries: int,
        batch_size: int = 64,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
        self._client = client
        self._base_url = base_url.rstrip(
            "/"
        )  # tolerate RAG_OVMS_EMBEDDINGS_URL=http://h:8002/
        self._model = model
        self._token_provider = token_provider
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        return f"ovms:{self._model}"

    @property
    def embeddings_url(self) -> str:
        return f"{self._base_url}/v3/embeddings"

    @property
    def status_url(self) -> str:
        return f"{self._base_url}/v3/models/{self._model}"

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def ready(self) -> tuple[bool, str]:
        return await probe_ready(
            self._client,
            self.status_url,
            token_provider=self._token_provider,
            timeout_s=self._timeout_s,
        )

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        body = await post_json_with_retry(
            self._client,
            self.embeddings_url,
            {"model": self._model, "input": batch},
            component=COMPONENT,
            token_provider=self._token_provider,
            timeout_s=self._timeout_s,
            max_retries=self._max_retries,
        )
        return _parse_embeddings(body, expected=len(batch), target=self.embeddings_url)


def _parse_embeddings(
    body: dict[str, Any], *, expected: int, target: str
) -> list[list[float]]:
    """Return ``expected`` vectors ordered by their ``index`` field."""
    rows = body.get("data")
    if not isinstance(rows, list) or len(rows) != expected:
        detail = f"expected {expected} rows in 'data', got: {excerpt(body)}"
        raise RagDependencyError(COMPONENT, detail, target=target)
    try:
        ordered = sorted(rows, key=lambda row: int(row["index"]))
        indices = [int(row["index"]) for row in ordered]
        vectors = [[float(x) for x in row["embedding"]] for row in ordered]
    except (KeyError, TypeError, ValueError) as exc:
        detail = f"malformed embedding row ({exc!r}) in: {excerpt(body)}"
        raise RagDependencyError(COMPONENT, detail, target=target) from exc
    if indices != list(range(expected)):
        detail = f"expected indices 0..{expected - 1}, got {indices}"
        raise RagDependencyError(COMPONENT, detail, target=target)
    return vectors
