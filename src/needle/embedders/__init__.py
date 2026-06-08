"""Embedding backends.

:class:`BaseEmbedder` is a concrete base that satisfies the domain
:class:`~conversational_core.domain.ports.embedder.Embedder` port.
:class:`DeterministicEmbedder` is a weight-free, reproducible embedder used by
tests, examples and benchmarks. Real model loaders live in
:mod:`needle.embedders.colpali_engine` and are imported lazily.
"""

from __future__ import annotations

from .base import BaseEmbedder
from .deterministic import DeterministicEmbedder

__all__ = ["BaseEmbedder", "DeterministicEmbedder"]
