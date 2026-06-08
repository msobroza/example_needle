"""Generic batching helpers.

Provides a 3.10-compatible replacement for :func:`itertools.batched`, which
is only available from Python 3.12.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def batched(iterable: Iterable[T], n: int) -> Iterator[list[T]]:
    """Yield successive ``n``-sized chunks from ``iterable`` as lists.

    The final chunk may be shorter than ``n``. Raises :class:`ValueError`
    if ``n < 1``.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    batch: list[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch
