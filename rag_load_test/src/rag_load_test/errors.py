"""Typed errors. Every error carries the offending value and what was expected."""

from __future__ import annotations

from typing import Any


class RagError(Exception):
    """Base class for all rag_load_test errors."""


class RagConfigError(RagError):
    """A ``RAG_*`` setting has an invalid value.

    Example::

        raise RagConfigError("RAG_RERANKER_BACKEND", got="ovsm",
                             expected="one of local, ovms, fake, none")
    """

    def __init__(self, setting: str, *, got: Any, expected: Any) -> None:
        self.setting = setting
        self.got = got
        self.expected = expected
        super().__init__(f"{setting}: got {got!r}, expected {expected}")

    @classmethod
    def from_validation_error(
        cls, exc: Exception, *, env_prefix: str = "RAG_"
    ) -> RagConfigError:
        """Translate the first pydantic validation error into a RagConfigError."""
        errors = getattr(exc, "errors", lambda: [])()
        first: dict[str, Any] = errors[0] if errors else {}
        loc = first.get("loc") or ("<unknown>",)
        setting = env_prefix + str(loc[0]).upper()
        return cls(setting, got=first.get("input"), expected=first.get("msg", str(exc)))


class RagDependencyError(RagError):
    """A dependency (embedder, reranker, vector store, llm) failed.

    ``kind`` is ``"unavailable"`` (default) or ``"timeout"``; the API maps them to
    HTTP 503 and 504 respectively.

    Example::

        raise RagDependencyError("reranker", "HTTP 503 from upstream",
                                 target="http://ovms:8001/v3/rerank")
    """

    def __init__(
        self,
        component: str,
        detail: str,
        *,
        target: str | None = None,
        kind: str = "unavailable",
    ) -> None:
        self.component = component
        self.detail = detail
        self.target = target
        self.kind = kind
        suffix = f" (target={target})" if target else ""
        super().__init__(f"{component} {kind}: {detail}{suffix}")
