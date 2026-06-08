# CLAUDE.md

Guidance for AI assistants (and humans) working in this repository.

## What this project is

`example_needle` is a page-level **multimodal** document retrieval library. It
embeds document pages as **images** (via late-interaction or dense visual
embedders such as ColPali / ColQwen2) and searches them with natural language
plus structured metadata filters.

## Architecture (read before editing)

Two packages live under `src/`:

- **`needle_core`** — the domain layer. Pure Python, **no torch**.
  Documents, queries, and the metadata-filter model live here.
- **`needle`** — the retrieval layer. Depends on PyTorch, imported **lazily**
  (only when an actual retriever is touched).

The centerpiece is `src/needle/retrieval/page_retrievers.py`:
`BasePageRetriever` → `MultimodalEmbedderRetriever` → concrete backends.

### Hard rules

1. **Never import `torch` (or model backends) from `needle_core`.** The
   domain must stay importable without a heavy ML stack.
2. **Keep `import needle` cheap.** Heavy imports go inside methods or behind the
   lazy `__getattr__` shims in `needle/__init__.py` and
   `needle/retrieval/__init__.py`.
3. The producer/consumer filter contract must stay in sync:
   `MultiFieldFilterAdapter.to_backend()` (write side) and
   `matches_filter()` (read side) speak the same Mongo-style dialect.

## Common commands

```bash
make install-dev   # editable install with dev tooling
make lint          # ruff + black --check
make test          # pytest (torch tests skip if torch is absent)
make cover         # pytest with coverage
```

## Testing without models

Use `needle.testing.DummyEmbedderRetriever`: a deterministic embedder that
exercises the full index/search/persist pipeline with no weights, no network,
no GPU. torch-gated tests begin with `pytest.importorskip("torch")`.

## Adding a retriever

Subclass `MultimodalEmbedderRetriever`, implement `_load_model`,
`_embed_images`, `_embed_query`, set `multi_vector`, and register it in
`needle/retrieval/registry.py`. See `docs/contributing-a-retriever.md`.
