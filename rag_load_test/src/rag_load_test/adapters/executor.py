"""Bounded thread pool for CPU-bound model and Chroma calls.

sentence-transformers and Chroma's ``PersistentClient`` are synchronous; running
them on a pool of ``RAG_MODEL_THREADS`` workers keeps the FastAPI event loop free
and turns overload into queueing instead of unbounded thread creation.

Example::

    executor = build_model_executor(max_workers=4)
    vectors = await run_blocking(executor, model.encode, ["hello"])
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")

# Thread names show up in dumps/profilers, so model work is easy to spot.
MODEL_THREAD_NAME_PREFIX = "rag-model"


def build_model_executor(max_workers: int) -> ThreadPoolExecutor:
    """Create the shared ``rag-model`` pool sized by ``RAG_MODEL_THREADS``.

    Example::

        executor = build_model_executor(settings.model_threads)
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers!r}")
    return ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix=MODEL_THREAD_NAME_PREFIX
    )


async def run_blocking(executor: Executor, fn: Callable[..., T], *args: Any) -> T:
    """Await a synchronous ``fn(*args)`` executed on ``executor``.

    Example::

        count = await run_blocking(executor, collection.count)
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, functools.partial(fn, *args))
