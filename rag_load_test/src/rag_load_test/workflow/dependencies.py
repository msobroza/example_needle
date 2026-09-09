"""Build the concrete ports behind ``RagDependencies`` from ``RagSettings``.

Every adapter keeps its heavy third-party import inside the method that needs
it, so this module stays cheap to import and the ``fake``/``ovms`` backends
never pay for torch, chromadb or openai they do not use.

Example::

    async with httpx.AsyncClient(timeout=settings.http_timeout_s) as client:
        deps = build_dependencies(settings, http_client=client, executor=executor)
        graph = build_rag_graph(deps)
"""

from __future__ import annotations

from concurrent.futures import Executor

import httpx

from ..adapters.auth import BearerTokenProvider, build_token_provider
from ..adapters.chroma_store import ChromaVectorStore
from ..adapters.fake_chat import FakeChatModel
from ..adapters.local_embedder import SentenceTransformerEmbedder
from ..adapters.local_reranker import CrossEncoderReranker
from ..adapters.openai_chat import OpenAIChatModel
from ..adapters.ovms_embedder import OvmsEmbedder
from ..adapters.ovms_reranker import OvmsReranker
from ..contracts.ports import ChatModelPort, EmbedderPort, RerankerPort, VectorStorePort
from ..errors import RagConfigError
from ..settings import RagSettings
from ..testing.fakes import FakeEmbedder, FakeReranker
from .nodes import RagDependencies


def build_embedder(
    settings: RagSettings,
    *,
    http_client: httpx.AsyncClient,
    executor: Executor,
    token_provider: BearerTokenProvider | None,
) -> EmbedderPort:
    """Select the embedder for ``RAG_EMBEDDER_BACKEND`` (``local`` loads weights now).

    Example::

        embedder = build_embedder(settings, http_client=client, executor=pool,
                                  token_provider=NoAuth())
    """
    backend = settings.embedder_backend
    if backend == "fake":
        return FakeEmbedder()
    if backend == "local":
        return SentenceTransformerEmbedder.from_pretrained(
            settings.embedder_model, executor
        )
    if backend == "ovms":
        return OvmsEmbedder(
            http_client,
            settings.ovms_embeddings_url,
            settings.ovms_embeddings_model,
            token_provider=_ovms_token_provider(token_provider, settings, http_client),
            timeout_s=settings.http_timeout_s,
            max_retries=settings.http_max_retries,
        )
    raise RagConfigError(
        "RAG_EMBEDDER_BACKEND", got=backend, expected="one of local, ovms, fake"
    )


def build_reranker(
    settings: RagSettings,
    *,
    http_client: httpx.AsyncClient,
    executor: Executor,
    token_provider: BearerTokenProvider | None,
) -> RerankerPort | None:
    """Select the reranker for ``RAG_RERANKER_BACKEND``; ``none`` -> ``None``.

    Example::

        reranker = build_reranker(settings, http_client=client, executor=pool,
                                  token_provider=NoAuth())
    """
    backend = settings.reranker_backend
    if backend == "none":
        return None
    if backend == "fake":
        return FakeReranker()
    if backend == "local":
        return CrossEncoderReranker.from_pretrained(settings.reranker_model, executor)
    if backend == "ovms":
        return OvmsReranker(
            http_client,
            settings.ovms_rerank_url,
            settings.ovms_rerank_model,
            token_provider=_ovms_token_provider(token_provider, settings, http_client),
            timeout_s=settings.http_timeout_s,
            max_retries=settings.http_max_retries,
        )
    raise RagConfigError(
        "RAG_RERANKER_BACKEND", got=backend, expected="one of local, ovms, fake, none"
    )


def build_vector_store(
    settings: RagSettings, *, executor: Executor, embedder_model: str
) -> VectorStorePort:
    """Open the Chroma collection (``:memory:`` allowed), recording ``embedder_model``.

    Example::

        store = build_vector_store(settings, executor=pool, embedder_model="bge-small")
    """
    return ChromaVectorStore.open(
        settings.chroma_path,
        settings.chroma_collection,
        executor,
        embedder_model=embedder_model,
    )


def build_chat_model(settings: RagSettings) -> ChatModelPort:
    """Select the LLM for ``RAG_LLM_BACKEND``: ``openai`` or ``fake``.

    Example::

        chat = build_chat_model(settings)
    """
    backend = settings.llm_backend
    if backend == "fake":
        return FakeChatModel(settings.fake_llm_latency_ms)
    if backend == "openai":
        return OpenAIChatModel.from_settings(settings)
    raise RagConfigError("RAG_LLM_BACKEND", got=backend, expected="one of openai, fake")


def build_dependencies(
    settings: RagSettings, *, http_client: httpx.AsyncClient, executor: Executor
) -> RagDependencies:
    """Wire every port plus the retrieval defaults into one ``RagDependencies``.

    Example::

        deps = build_dependencies(settings, http_client=client, executor=pool)
    """
    token_provider = build_token_provider(settings, http_client)
    embedder = build_embedder(
        settings,
        http_client=http_client,
        executor=executor,
        token_provider=token_provider,
    )
    reranker = build_reranker(
        settings,
        http_client=http_client,
        executor=executor,
        token_provider=token_provider,
    )
    # The store remembers which embedder wrote it so /readyz can flag a mismatch.
    vector_store = build_vector_store(
        settings, executor=executor, embedder_model=embedder.model_name
    )
    return RagDependencies(
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        chat_model=build_chat_model(settings),
        top_k_retrieve=settings.top_k_retrieve,
        top_k_rerank=settings.top_k_rerank,
    )


def _ovms_token_provider(
    token_provider: BearerTokenProvider | None,
    settings: RagSettings,
    http_client: httpx.AsyncClient,
) -> BearerTokenProvider:
    # Direct callers may pass None; derive the provider from RAG_OVMS_AUTH then.
    if token_provider is not None:
        return token_provider
    return build_token_provider(settings, http_client)
