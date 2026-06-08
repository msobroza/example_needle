"""Lightweight wall-clock timing helpers.

:class:`Timer` is a context manager for measuring a block of code; :func:`timed`
is a decorator that logs the wall time of a function call and records the most
recent duration on the wrapper for inexpensive inspection in tests.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from types import TracebackType
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class Timer:
    """Context manager measuring elapsed wall-clock time.

    Usage::

        with Timer() as t:
            do_work()
        print(t.elapsed, t.elapsed_ms)

    ``elapsed`` reflects the time between ``__enter__`` and ``__exit__``. While
    the block is still running, it reflects the time since entry.
    """

    def __init__(self) -> None:
        self._start: float | None = None
        self._end: float | None = None

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        self._end = None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds.

        Raises :class:`RuntimeError` if the timer was never started.
        """
        if self._start is None:
            raise RuntimeError("Timer has not been started")
        end = self._end if self._end is not None else time.perf_counter()
        return end - self._start

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000.0


def timed(func: F) -> F:
    """Decorator that logs the wall time of ``func`` and records it.

    The wrapped callable returns the original return value unchanged. The most
    recent duration in seconds is attached to the wrapper as ``.last_seconds``
    (``None`` before the first call) and also emitted at ``DEBUG`` level.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with Timer() as timer:
            result = func(*args, **kwargs)
        wrapper.last_seconds = timer.elapsed  # type: ignore[attr-defined]
        logger.debug("%s took %.6fs", func.__qualname__, timer.elapsed)
        return result

    wrapper.last_seconds = None  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]
