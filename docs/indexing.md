# Indexing

An **index store** holds the `(embedding, payload)` pairs that
[`RetrievalPipeline`](pipeline.md) (and any consumer of the
[`IndexStorePort`](domain-model.md#hexagonal-ports)) produces while indexing and
reads back while searching. Each store keeps the same in-memory backbone — two
parallel lists indexed positionally, so the `i`-th embedding belongs with the
`i`-th payload — and differs only in how (and whether) it persists to disk.

All three concrete stores live in `needle.indexing` and implement
`needle_core.domain.ports.index_store.IndexStorePort`:

```python
from needle.indexing import (
    BaseIndexStore,
    InMemoryIndexStore,
    PickleIndexStore,
    NumpyIndexStore,
)
```

## The `IndexStorePort` contract

`IndexStorePort` is an ABC with a deliberately tiny surface — append items,
enumerate them, clear, persist:

| Method | Purpose |
| --- | --- |
| `add(embedding, payload)` | Append one `(embedding, payload)` pair. |
| `embeddings() -> list[np.ndarray]` | All stored embeddings, in insertion order. |
| `payloads() -> list[dict]` | All stored payloads, in insertion order. |
| `clear()` | Remove every stored item. |
| `save(path)` | Persist the store to `path`. |
| `load(path) -> IndexStorePort` | Load from `path`; returns `self`. |
| `__len__()` | Number of stored items. |

`BaseIndexStore` implements every method except `save` / `load`, which it leaves
abstract so each subclass can choose its own on-disk representation. It also adds
an `extend(embeddings, payloads)` convenience that zips the two iterables.

> **Note:** an empty store is *falsy* because it defines `__len__`. Code that
> checks "did the caller pass a store?" must compare against `None`
> (`store if store is not None else ...`), which is exactly what
> `RetrievalPipeline` does.

## The three stores at a glance

| Store | Persistence | On-disk format | Best for |
| --- | --- | --- | --- |
| `InMemoryIndexStore` | None (volatile) | — | Tests, scratch indexing, pipelines that rebuild every run. |
| `PickleIndexStore` | `save` / `load` | One pickled `{"embeddings", "payloads"}` dict | Drop-in compatibility with the retrievers' index files. |
| `NumpyIndexStore` | `save` / `load` | A single `.npz` archive | Large float arrays where compression matters. |

### `InMemoryIndexStore`

Lives only for the lifetime of the process and **deliberately rejects
persistence**: there is no on-disk representation, so `save` and `load` both raise
`needle.exceptions.IndexStoreError`. This is the default store for
`RetrievalPipeline`.

```python
from needle.indexing import InMemoryIndexStore

store = InMemoryIndexStore()
store.add(embedding, {"file": "a.png", "page": 1})
print(len(store))   # 1
```

### `PickleIndexStore`

Persists the store as a single pickled dict using the **exact layout produced by
the page retrievers** in `needle.retrieval.page_retrievers`:

```python
{"embeddings": [...], "payloads": [...]}
```

So an index written by a retriever's `save_index` can be read by
`PickleIndexStore.load`, and vice versa. Pickle preserves arbitrary Python objects
— including ragged lists of multi-vector arrays and rich payload values
(`Document`, `DocumentVersion`) — which is what keeps this adapter drop-in
compatible.

```python
from needle.indexing import PickleIndexStore

store = PickleIndexStore()
store.add(embedding, {"file": "a.png", "page": 1})
store.save("index.pkl")

reloaded = PickleIndexStore().load("index.pkl")   # load() returns self
print(len(reloaded))
```

!!! warning "Pickle security caveat"
    Pickle is Python-specific and **unsafe to load from untrusted sources** — a
    crafted pickle can execute arbitrary code on `load`. Only load index files you
    produced or otherwise trust.

### `NumpyIndexStore`

Persists embeddings and payloads inside a single `.npz` archive. Compared with
`PickleIndexStore`:

* **Pros** — `np.savez_compressed` shrinks large float arrays well, and `.npz` is
  a familiar, inspectable container.
* **Cons** — embeddings are stored as a single NumPy *object* array and payloads
  are pickled into the same archive, so `allow_pickle=True` is required on load.
  That means the file is still Python-specific and **unsafe to read from untrusted
  sources**, just like a plain pickle.

Embeddings are wrapped in a 1-D `dtype=object` array (rather than stacked) so the
store stays robust for **ragged** multi-vector arrays, where each page may have a
different number of token embeddings `(num_tokens, dim)`.

```python
from needle.indexing import NumpyIndexStore

store = NumpyIndexStore(compressed=True)   # compressed=False for faster writes
store.add(embedding, {"file": "a.png", "page": 1})
store.save("index.npz")

reloaded = NumpyIndexStore().load("index.npz")
```

Pass `compressed=False` to use `numpy.savez` (faster writes) instead of
`numpy.savez_compressed` (smaller files).

## Using a store with the pipeline

Because the store is just a `BaseIndexStore`, choosing a persistence strategy is a
constructor argument on the pipeline:

```python
from needle.factory import build_pipeline
from needle.indexing import PickleIndexStore
from needle.retrieval.data import InputDocument

store = PickleIndexStore()
pipeline = build_pipeline(store=store)
pipeline.index([InputDocument.from_path("a.png")])
store.save("index.pkl")
```

See [Pipeline](pipeline.md) for the full indexing/search flow and
[Types](domain-model.md) for the `Embedding` / `Payload` aliases the stores use.
