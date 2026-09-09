"""Runtime configuration, read from ``RAG_*`` environment variables (or ``.env``).

Example::

    settings = RagSettings.from_env()   # raises RagConfigError on bad values
    settings = RagSettings(_env_file=None, reranker_backend="fake")  # in tests
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import RagConfigError

EmbedderBackend = Literal["local", "ovms", "fake"]
RerankerBackend = Literal["local", "ovms", "fake", "none"]
LlmBackend = Literal["openai", "fake"]
OvmsAuthMode = Literal["none", "static", "domino"]


class RagSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),  # allow the ``model_threads`` field name
    )

    # Label echoed in every response so Locust can tag the run.
    topology: str = "monolith"

    embedder_backend: EmbedderBackend = "local"
    embedder_model: str = "BAAI/bge-small-en-v1.5"
    reranker_backend: RerankerBackend = "local"
    reranker_model: str = "BAAI/bge-reranker-base"

    ovms_embeddings_url: str = "http://localhost:8002"
    ovms_embeddings_model: str = "bge-small-en-v1.5"
    ovms_rerank_url: str = "http://localhost:8001"
    ovms_rerank_model: str = "bge-reranker-base"
    ovms_auth: OvmsAuthMode = "none"
    ovms_bearer_token: str = ""
    domino_access_token_url: str = "http://localhost:8899/access-token"
    http_timeout_s: float = Field(30.0, gt=0)
    http_max_retries: int = Field(2, ge=0)

    llm_backend: LlmBackend = "openai"
    openai_model: str = "gpt-4o-mini"
    openai_timeout_s: float = Field(60.0, gt=0)
    openai_max_retries: int = Field(2, ge=0)
    fake_llm_latency_ms: float = Field(0.0, ge=0)

    chroma_path: str = "./data/chroma"
    chroma_collection: str = "rag_passages"
    top_k_retrieve: int = Field(20, ge=1)
    top_k_rerank: int = Field(5, ge=1)
    model_threads: int = Field(4, ge=1)

    domino_tracing: bool = False
    host: str = "0.0.0.0"
    port: int = Field(8888, ge=1, le=65535)
    log_level: str = "info"

    @classmethod
    def from_env(cls, **overrides: Any) -> RagSettings:
        """Build settings from the environment, raising RagConfigError on bad values."""
        try:
            return cls(**overrides)
        except ValidationError as exc:
            raise RagConfigError.from_validation_error(exc) from exc
