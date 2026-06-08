# Contributing a retriever

This tutorial walks through adding a new retriever backend. Thanks to
`MultimodalEmbedderRetriever`, you only implement **three methods** and set one
flag — indexing, search, scoring, persistence and metadata filtering are all
inherited.

The complete, runnable reference implementation is
`needle.testing.DummyEmbedderRetriever` (`src/needle/testing.py`); it embeds bytes
deterministically with no weights, so refer to it whenever a detail is unclear.

## Step 1 — Subclass `MultimodalEmbedderRetriever`

```python
import numpy as np
import torch
from needle.retrieval.page_retrievers import MultimodalEmbedderRetriever


class MyRetriever(MultimodalEmbedderRetriever):
    multi_vector = True  # class flag: chooses the scoring function (see Step 3)

    def __init__(self, model_name: str = "my-org/my-encoder", **kwargs):
        self.model_name = model_name
        super().__init__(**kwargs)  # forwards index_path, batch_size, dpi, extractors
```

Set `model_name` **before** calling `super().__init__`, because the base
constructor may call `_ensure_model()` (and will `load_index()` if `index_path`
already exists). Always forward `**kwargs` so callers keep access to `index_path`,
`batch_size`, `dpi` and `extractors`.

## Step 2 — The contract of the three abstract methods

### `_load_model(self) -> None`

Load weights and set `self.model` and `self.processor`. Import heavy model
dependencies **inside this method** (not at module level) to preserve the package's
lazy-import guarantee. A model-free retriever may simply set both to `None`, as
`DummyEmbedderRetriever` does:

```python
def _load_model(self) -> None:
    self.model = None
    self.processor = None
```

`_ensure_model()` calls this exactly once (it guards on `self._loaded`), the first
time you `index()` or `search()`.

### `_embed_images(self, images: list) -> ...`

Receives a batch of `PIL.Image` objects (size `batch_size`) and must return **one
embedding per image**, in order. The base `_index_one` iterates the return value
and appends each element to `self._embeddings`, so any sequence indexable by
position works (a `list[np.ndarray]` or a stacked `np.ndarray`).

| `multi_vector` | Per-image embedding shape |
| --- | --- |
| `True` | a 2D array `(num_tokens, dim)` |
| `False` | a 1D array `(dim,)` |

### `_embed_query(self, query: str) -> np.ndarray`

Receives the raw `query.query_text` and returns a **single** embedding:

| `multi_vector` | Query embedding shape |
| --- | --- |
| `True` | `(num_query_tokens, dim)` |
| `False` | `(dim,)` |

## Step 3 — How `_score` uses `multi_vector`

You never write scoring yourself; the base class dispatches on the flag:

```python
def _score(self, q_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    doc = doc_emb.astype(np.float32)
    if self.multi_vector:
        return (q_emb @ doc.T).max(axis=1).sum()          # MaxSim
    return float(q_emb @ doc / (np.linalg.norm(q_emb) * np.linalg.norm(doc) + 1e-9))  # cosine
```

* **`multi_vector = True`** → MaxSim over the token matrices. The shapes must line
  up: `q_emb` is `(num_query_tokens, dim)` and each `doc_emb` is
  `(num_tokens, dim)`, so `q_emb @ doc.T` is `(num_query_tokens, num_tokens)`.
* **`multi_vector = False`** → cosine over single vectors. Both `q_emb` and
  `doc_emb` are `(dim,)`.

The flag can be a **class attribute** (as in the built-ins) or, like
`DummyEmbedderRetriever`, an **instance attribute** set in `__init__` so a single
class can run in either regime. `_score` reads `self.multi_vector`, so both work.

## Worked example: `DummyEmbedderRetriever`

`src/needle/testing.py` is a full, dependency-free implementation that exercises
the entire pipeline. The relevant parts:

```python
import hashlib
import numpy as np
from needle.retrieval.page_retrievers import MultimodalEmbedderRetriever


class DummyEmbedderRetriever(MultimodalEmbedderRetriever):
    def __init__(self, *, dim=32, num_tokens=8, multi_vector=True, **kwargs):
        self.dim = dim
        self.num_tokens = num_tokens
        self.multi_vector = multi_vector   # instance flag → run multi-vector OR dense
        super().__init__(**kwargs)

    def _load_model(self) -> None:
        self.model = None                  # no real model
        self.processor = None

    def _embed_images(self, images: list) -> list[np.ndarray]:
        out = []
        for image in images:
            rng = self._rng(self._image_key(image))      # deterministic from bytes
            if self.multi_vector:
                out.append(self._unit_vectors(rng, self.num_tokens).astype(np.float16))  # (num_tokens, dim)
            else:
                out.append(self._unit_vectors(rng, 1)[0].astype(np.float16))             # (dim,)
        return out

    def _embed_query(self, query: str) -> np.ndarray:
        rng = self._rng(("query:" + query).encode())
        if self.multi_vector:
            tokens = max(2, len(query.split()))
            return self._unit_vectors(rng, tokens)        # (num_query_tokens, dim)
        return self._unit_vectors(rng, 1)[0]              # (dim,)
```

It returns a `list[np.ndarray]` from `_embed_images` (one entry per image) and a
single array from `_embed_query`, matching the shapes in the tables above for the
chosen regime.

## Step 4 — Register it

To expose your retriever by string name through `get_retriever` / `RETRIEVERS`, add
it to `src/needle/retrieval/registry.py`:

```python
# src/needle/retrieval/registry.py
from .page_retrievers import (
    ...,
    MyRetriever,   # 1. import it
)

RETRIEVERS: dict[str, type[MultimodalEmbedderRetriever]] = {
    ...,
    "my-retriever": MyRetriever,   # 2. add a canonical name
}

_ALIASES = {
    ...,
    "mine": "my-retriever",        # 3. (optional) friendly aliases
}
```

After that:

```python
from needle.retrieval.registry import get_retriever, available_retrievers

print(available_retrievers())                # includes "my-retriever"
retriever = get_retriever("mine", index_path="index.pkl")  # alias resolves
```

> Names are matched case-insensitively after stripping whitespace; aliases are
> resolved before the lookup. If your retriever also lives at the top of the
> package API, you can additionally add its name to the `_LAZY_RETRIEVERS` sets in
> `needle/__init__.py` and `needle/retrieval/__init__.py` to keep it lazily
> importable.

## Step 5 — Verify

Drive the full index → save → load → search loop, exactly like the built-ins:

```python
from needle.retrieval.data import InputDocument
from conversational_core.domain.interaction.query import Query

retriever = MyRetriever(index_path="index.pkl")
retriever.index([InputDocument.from_path("page.png", metadata={"year": 2024})])

results = retriever.search(Query.of("net revenue", filters={"year": 2024}), top_k=5)
for hit in results:
    print(hit.score, hit.normalized_score, hit.document_version.filename, hit.page)
```

If this round-trips, your `_load_model` / `_embed_images` / `_embed_query` shapes
are correct and you inherit indexing, persistence, scoring and metadata filtering
for free.
