# CLI

`example_needle` ships a `needle` command-line interface (registered as the
`needle` console script via `pyproject.toml`). It has three subcommands:

| Command | Purpose |
| --- | --- |
| `needle list` | List available retriever backends. |
| `needle index <files…>` | Build an index from one or more files. |
| `needle search "<query>"` | Query an existing index. |

Heavy imports (torch, model backends) happen **inside** the handlers, so
`needle --help` and `needle list` stay fast and work without the model
dependencies installed.

!!! note "Real backends need the `[retrieval]` extra"
    `needle index` and `needle search` construct a model-backed retriever via the
    registry, which requires the optional `torch` + `colpali-engine` dependencies.
    Install them with `pip install -e ".[retrieval]"` (and the `[pdf]` extra to
    rasterise real PDFs). `needle list` works without them. For a fully torch-free
    workflow, use the [`RetrievalPipeline`](pipeline.md) API instead.

## Global options

```
needle [-v | --verbose] <command> ...
```

* `-v`, `--verbose` — enable `DEBUG` logging (otherwise `INFO`).

## `needle list`

List the canonical retriever backend names, one per line:

```bash
needle list
```

The names come from `needle.retrieval.registry.available_retrievers()` — see
[Retrievers](retrievers.md) for the full table.

## `needle index`

Index one or more documents into an index file.

```bash
needle index report.pdf slides.pptx --retriever colqwen2 --index index.pkl
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `files` (positional, ≥1) | — | One or more document paths to index. |
| `--retriever` | `colqwen2` | Retriever backend name (canonical name or alias). |
| `--index` | `index.pkl` | Path to the index file to write. |
| `--dpi` | `150` | Rasterisation resolution (see [Extractors](extractors.md)). |
| `--batch-size` | `4` | Page images embedded per forward pass. |

Each path is turned into an `InputDocument` with
`InputDocument.from_path(path)`, the retriever indexes them, and the command
prints how many pages were written:

```
Indexed 27 pages into index.pkl
```

## `needle search`

Search an existing index with a natural-language query.

```bash
needle search "quarterly revenue chart" --retriever colqwen2 --index index.pkl --top-k 5
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `query` (positional) | — | Natural-language query text. |
| `--retriever` | `colqwen2` | Retriever backend name (must match the index). |
| `--index` | `index.pkl` | Path to the index file to read. |
| `--top-k` | `5` | Number of results to return. |

The query is wrapped in a `Query(query_text=...)`, the retriever searches and
visualises the hits via `retriever.show(...)`, and each result is printed as a
ranked line:

```
1. [12.834] report.pdf — page 4
2. [11.207] report.pdf — page 9
...
```

> The `--retriever` and `--index` flags are shared by both `index` and `search`.
> Use the **same** retriever backend for searching that you used for indexing, so
> the query embeddings are produced by the same model.
