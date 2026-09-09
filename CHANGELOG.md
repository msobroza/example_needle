# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Hexagonal **ports** in the domain (`Embedder`, `IndexStorePort`,
  `PageRenderer`, `RetrieverPort`) and adapters in `needle`.
- Composable, torch-free `RetrievalPipeline` plus `build_pipeline` /
  `build_retriever` factories and `RetrieverConfig`.
- Pluggable `needle.embedders` (incl. `DeterministicEmbedder`),
  `needle.scoring` strategies (MaxSim / cosine), `needle.indexing` stores
  (in-memory / pickle / numpy) and `needle.preprocessing` helpers.
- `needle.metrics` (recall@k, precision@k, MRR, MAP, nDCG, `Timer`) and
  `needle.io` (paths, JSON index manifests).
- `needle.application` services and use-cases (indexing / search / ranking).
- Domain value objects: identifiers, ranking, pagination, geometry, events,
  metadata schema, conversation/response, document discovery + checksums.
- Packaging/dev extras: Dockerfile, mkdocs site, tox, Read the Docs, release &
  docs workflows, Dependabot, CODEOWNERS, SECURITY and CITATION.
- `rag_load_test/` sub-project: a FastAPI + LangGraph text-RAG service over an
  Elasticsearch vector store (`langchain-elasticsearch`) with switchable
  embedder/reranker transports (in-process sentence-transformers or OpenVINO
  Model Server over HTTP), a Locust suite recording per-stage latencies, the
  `rag-compare` run-comparison command, a single Domino 6.2 entry point
  (`domino/app.sh`, role from `RAG_ROLE`) for the `monolith` /
  `split-reranker` / `split-all` topologies, and a dedicated `rag-load-test`
  CI job.

## [0.1.0] - 2024-01-01

### Added

- `rag_load_test`: `rag-model-server`, a FastAPI model server with the OVMS `/v3`
  contract, `make loadtest-models` (`RAG_LOADTEST_TARGET=rerank|embeddings`) and
  the `fastapi-reranker` / `fastapi-embedder` Domino roles, to compare FastAPI
  and OpenVINO Model Server serving of the same models.
- `rag_load_test`: `RAG_OVMS_PREFIX_PROXY=1` runs nginx in front of OVMS inside the
  Domino app to strip `DOMINO_RUN_HOST_PATH` (OVMS has no base-path option).
- `needle_core` domain package:
  - `DocumentExtension` enum with format detection (`from_path`, `from_string`).
  - `Document`, `DocumentVersion`, `DocumentPage` aggregate.
  - `Query` value object with structured metadata filters.
  - Metadata filter model: `FilterOperator`, `FieldFilter`, `MetadataFilterSpec`.
  - `MultiFieldFilterAdapter` producing Mongo-style backend filters.
- `needle` retrieval package:
  - `BasePageRetriever` and `MultimodalEmbedderRetriever` base classes.
  - Concrete backends: `ColPaliRetriever`, `ColQwen2Retriever`,
    `NomicDenseRetriever`, `TomoroColQwen3Retriever`, `ColModernVBertRetriever`,
    `ModernVBertRetriever`.
  - Page extractors: `ImageFileExtractor`, `PdfToImageExtractor`,
    `OfficeToImageExtractor`, and the `IMAGE_EXTRACTORS` registry.
  - `matches_filter` + `SimilarityMapVisualizer` helpers.
  - Retriever registry/factory (`get_retriever`, `available_retrievers`).
  - `needle.testing.DummyEmbedderRetriever` deterministic test double.
  - `needle` command-line interface (`list` / `index` / `search`).
- Test suite, documentation, examples, and CI.

[Unreleased]: https://github.com/msobroza/example_needle/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/msobroza/example_needle/releases/tag/v0.1.0
