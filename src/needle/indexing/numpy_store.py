"""NumPy-backed index store — ``.npz`` archives for embeddings + payloads.

Compared with :class:`~needle.indexing.pickle_store.PickleIndexStore`:

* **Pros** — ``np.savez_compressed`` shrinks large float arrays well and the
  ``.npz`` container is a familiar, inspectable format.
* **Cons** — embeddings are stored as a single NumPy *object* array and
  payloads are pickled into the same archive, so ``allow_pickle=True`` is
  required on load. That means the file is still Python-specific and unsafe
  to read from untrusted sources, just like a plain pickle.

Wrapping the embeddings in an object array (rather than stacking them) keeps
the store robust for **ragged** multi-vector arrays, where each page may have
a different number of token embeddings ``(num_tokens, dim)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..types import Embedding, PathLike, Payload
from .base import BaseIndexStore


def _to_object_array(items: list) -> np.ndarray:
    """Pack a (possibly ragged) list into a 1-D ``dtype=object`` array.

    Building the array empty-then-filling avoids NumPy trying to broadcast
    same-shaped entries into a regular N-D array, which would defeat the
    ragged-friendly round-trip.
    """
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


class NumpyIndexStore(BaseIndexStore):
    """Persist embeddings and payloads inside a single ``.npz`` archive.

    Set ``compressed=False`` to use :func:`numpy.savez` (faster writes) instead
    of :func:`numpy.savez_compressed` (smaller files).
    """

    def __init__(self, compressed: bool = True) -> None:
        super().__init__()
        self.compressed = compressed

    def save(self, path: PathLike) -> None:
        """Write embeddings (object array) and payloads (pickled) to ``path``."""
        embeddings = _to_object_array(self._embeddings)
        payloads = _to_object_array(self._payloads)
        saver = np.savez_compressed if self.compressed else np.savez
        with open(Path(path), "wb") as fh:
            saver(fh, embeddings=embeddings, payloads=payloads)

    def load(self, path: PathLike) -> NumpyIndexStore:
        """Restore the store from an archive written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=True) as data:
            embeddings: list[Embedding] = list(data["embeddings"])
            payloads: list[Payload] = list(data["payloads"])
        self._embeddings = embeddings
        self._payloads = payloads
        return self
