"""Shared fixtures. No model weights, no network anywhere in this suite."""

from __future__ import annotations

import os

# Locust gevent-patches the interpreter at import unless this is set; the
# service tests use asyncio, so importing locust (shapes/locustfile tests) must
# not patch sockets/threads. Must run before any ``import locust``.
os.environ.setdefault("LOCUST_SKIP_MONKEY_PATCH", "1")

from collections.abc import Callable  # noqa: E402

import pytest  # noqa: E402

from rag_load_test.settings import RagSettings  # noqa: E402


@pytest.fixture
def settings_factory() -> Callable[..., RagSettings]:
    """Build RagSettings without reading a stray ``.env`` file."""

    def _make(**overrides: object) -> RagSettings:
        return RagSettings(_env_file=None, **overrides)  # type: ignore[call-arg]

    return _make
