"""Project-wide default constants.

Keeping defaults in one module avoids magic numbers scattered across the
retrievers, the CLI and the configuration objects.
"""

from __future__ import annotations

from typing import Final

#: Default rasterisation resolution (dots per inch) for page extractors.
DEFAULT_DPI: Final[int] = 150

#: Default number of page images embedded per forward pass.
DEFAULT_BATCH_SIZE: Final[int] = 4

#: Default on-disk index location.
DEFAULT_INDEX_PATH: Final[str] = "index.pkl"

#: Default number of results returned by a search.
DEFAULT_TOP_K: Final[int] = 10

#: Storage dtype for page embeddings (half precision halves index size).
EMBEDDING_DTYPE: Final[str] = "float16"

#: Numerical floor used to avoid division-by-zero in normalisation/cosine.
EPS: Final[float] = 1e-9

__all__ = [
    "DEFAULT_DPI",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_INDEX_PATH",
    "DEFAULT_TOP_K",
    "EMBEDDING_DTYPE",
    "EPS",
]
