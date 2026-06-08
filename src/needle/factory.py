"""High-level constructors for retrievers and pipelines."""

from __future__ import annotations

from typing import Any, Optional

from needle_core.domain.ports.embedder import Embedder

from .config import RetrieverConfig
from .pipeline import RetrievalPipeline


def build_retriever(config: RetrieverConfig) -> Any:
    """Build a concrete (model-backed) retriever from a :class:`RetrieverConfig`.

    Imports the registry lazily, so this only pulls in torch when called.
    """
    from .retrieval.registry import get_retriever

    return get_retriever(config.name, **config.to_kwargs())


def build_pipeline(
    embedder: Optional[Embedder] = None, **kwargs: Any
) -> RetrievalPipeline:
    """Build a torch-free :class:`RetrievalPipeline`.

    Defaults to the weight-free :class:`~needle.embedders.DeterministicEmbedder`
    so the result runs anywhere.
    """
    if embedder is None:
        from .embedders import DeterministicEmbedder

        embedder = DeterministicEmbedder()
    return RetrievalPipeline(embedder, **kwargs)


__all__ = ["build_retriever", "build_pipeline"]
