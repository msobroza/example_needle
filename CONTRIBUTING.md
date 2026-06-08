# Contributing to example_needle

Thanks for your interest in improving `example_needle`! This guide covers the
local workflow and the conventions we follow.

## Development setup

```bash
git clone https://github.com/msobroza/example_needle
cd example_needle
python -m venv .venv && source .venv/bin/activate
make install-dev          # editable install + pytest/ruff/black/mypy
pre-commit install        # optional but recommended
```

To exercise the real model backends you also need PyTorch and `colpali-engine`:

```bash
pip install -e ".[retrieval]"
```

## Project layout

```
src/
  needle_core/   # dependency-light domain (no torch)
  needle/                # retrieval layer (torch, loaded lazily)
tests/                   # pytest suite (torch tests skip if torch is absent)
docs/                    # markdown documentation
examples/                # runnable examples (quickstart uses the dummy backend)
scripts/                 # operational helper scripts
```

## Running checks

```bash
make lint        # ruff + black --check
make typecheck   # mypy
make test        # pytest
make cover       # pytest with coverage
```

All of these run in CI on every pull request. Please make sure they pass
locally first.

## Coding conventions

- **Style**: `black` (line length 88) and `ruff` enforce formatting + linting.
- **Typing**: public APIs are type-annotated; `mypy` runs in non-strict mode.
- **Domain stays light**: `needle_core` must not import `torch`,
  `transformers`, or any model backend. Heavy imports live in `needle` and are
  done lazily inside methods so `import needle` stays cheap.
- **Tests**: torch-dependent tests start with `pytest.importorskip("torch")`
  and are marked `@pytest.mark.torch`. The deterministic
  `needle.testing.DummyEmbedderRetriever` lets you test the full pipeline
  without any model weights.

## Adding a retriever

1. Subclass `MultimodalEmbedderRetriever` in
   `src/needle/retrieval/page_retrievers.py`.
2. Implement `_load_model`, `_embed_images`, `_embed_query`, and set the
   `multi_vector` class attribute.
3. Register it in `src/needle/retrieval/registry.py` (`RETRIEVERS` and, if
   helpful, `_ALIASES`).
4. Add a row to the table in `docs/retrievers.md` and a parametrised case in
   `tests/test_page_retrievers.py::test_concrete_retriever_declarations`.

See [`docs/contributing-a-retriever.md`](docs/contributing-a-retriever.md) for a
full walkthrough.

## Commit messages & PRs

- Keep commits focused; write imperative subject lines ("Add ColX retriever").
- Fill in the pull-request template checklist.
- Link any related issues.

By contributing you agree that your contributions are licensed under the
project's [Apache-2.0](LICENSE) license.
