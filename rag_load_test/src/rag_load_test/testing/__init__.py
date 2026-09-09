"""Model-free doubles for the ports (no weights, no network)."""

from __future__ import annotations

from ..adapters.fake_chat import FakeChatModel
from .fakes import FakeEmbedder, FakeReranker, InMemoryVectorStore

__all__ = ["FakeChatModel", "FakeEmbedder", "FakeReranker", "InMemoryVectorStore"]
