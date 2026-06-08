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

## [0.1.0] - 2024-01-01

### Added

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
