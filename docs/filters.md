# Metadata filters

Beyond semantic ranking, you can constrain results by document metadata — the
`document_metadata` dict you attach to each `DocumentVersion` (or pass via
`InputDocument.from_path(..., metadata=...)`). At index time that metadata is
flattened into each page's payload; at search time it is matched against the
filter carried by the `Query`.

This page walks the filter contract end to end:

```
Query (filters) → MetadataFilterSpec (AND of FieldFilters)
              → MultiFieldFilterAdapter.to_backend() → {field: {"$op": value}}
              → matches_filter(payload[field], criterion) → boolean mask
```

## Building a `Query` with filters

There are two ergonomic ways to attach filters.

### `Query.of(...)` — from a plain dict

`Query.of` builds a query from text plus a `{field: value}` mapping. Scalar values
become equality (`EQ`) filters; list/tuple/set values become `IN` filters.

```python
from needle_core.domain.interaction.query import Query

# language == "en"  AND  year in [2022, 2023]
query = Query.of(
    "operating margin by segment",
    top_k=5,
    filters={"language": "en", "year": [2022, 2023]},
)
```

### `.with_filter(...)` — explicit operators, chainable

`with_filter` returns a **new** `Query` with one extra predicate, so you can chain
calls and choose the operator explicitly. It accepts a `FilterOperator` or its
string value.

```python
from needle_core.domain.interaction.query import Query
from needle_core.domain.metadata.filter_spec import FilterOperator

query = (
    Query.of("operating margin by segment")
    .with_filter("year", 2020, FilterOperator.GTE)
    .with_filter("year", 2024, FilterOperator.LTE)   # range: 2020 <= year <= 2024
    .with_filter("language", "en")                   # operator defaults to EQ
    .with_filter("region", ["EU", "US"], "in")       # operator as a string
)
```

## The `FilterOperator` enum

`needle_core.domain.metadata.filter_spec.FilterOperator` defines the
supported comparisons:

| Member | Value | Meaning |
| --- | --- | --- |
| `EQ` | `"eq"` | equal |
| `NE` | `"ne"` | not equal |
| `IN` | `"in"` | membership (`actual in value`) |
| `NIN` | `"nin"` | non-membership |
| `GT` | `"gt"` | greater than |
| `GTE` | `"gte"` | greater than or equal |
| `LT` | `"lt"` | less than |
| `LTE` | `"lte"` | less than or equal |
| `CONTAINS` | `"contains"` | `value in actual` (e.g. substring / element) |
| `EXISTS` | `"exists"` | field present (truthy) / absent (falsy) |
| `REGEX` | `"regex"` | `re.search(value, actual)` matches |

`FilterOperator.coerce(...)` turns a string like `"gte"` into the enum member, so
both forms are interchangeable wherever an operator is accepted.

## `MetadataFilterSpec` — an AND of predicates

A `Query` holds a `MetadataFilterSpec`, which is an **ordered conjunction (logical
AND)** of `FieldFilter` predicates. A `FieldFilter` is a frozen
`(field, operator, value)` triple:

```python
from needle_core.domain.metadata.filter_spec import (
    FieldFilter, FilterOperator, MetadataFilterSpec,
)

spec = MetadataFilterSpec()
spec.add("year", FilterOperator.GTE, 2020)   # add() returns self, so it chains
spec.add("year", FilterOperator.LTE, 2024)
spec.add(FieldFilter("language", FilterOperator.EQ, "en"))

assert not spec.is_empty()
assert spec.fields() == {"year", "language"}
```

`MetadataFilterSpec.from_dict({...})` is the same conversion `Query.of` uses
(lists → `IN`, scalars → `EQ`). All predicates must hold for a page to pass.

## `MultiFieldFilterAdapter.to_backend()` — the Mongo-style dict

Before search, the retriever translates the spec into a backend representation via
`MultiFieldFilterAdapter`. It produces a `{field: {"$op": value, ...}}` dict and
**merges multiple predicates on the same field into one criterion dict**, which is
exactly what expresses a range.

The operator keys come from `OPERATOR_TO_BACKEND`
(`EQ → "$eq"`, `GTE → "$gte"`, `IN → "$in"`, …).

```python
from needle_core.domain.metadata.backend_filter_adapters import (
    MultiFieldFilterAdapter,
)

# year >= 2020 AND year <= 2024 AND region in ["EU", "US"]
spec = (
    MetadataFilterSpec()
    .add("year", "gte", 2020)
    .add("year", "lte", 2024)
    .add("region", "in", ["EU", "US"])
)

backend = MultiFieldFilterAdapter().to_backend(spec)
print(backend)
```

Produces exactly:

```python
{
    "year": {"$gte": 2020, "$lte": 2024},
    "region": {"$in": ["EU", "US"]},
}
```

Note how the two `year` predicates collapsed into a single criterion dict — that
merge is how a range survives the translation.

## `matches_filter()` — the read side

During `search()`, for each field/criterion pair the retriever builds a boolean
mask by calling
`needle.retrieval.page_retriever_utils.matches_filter(actual, criterion)` for
every page payload, then ANDs the per-field masks together. `matches_filter`
understands the criterion shapes the adapter can emit:

* `None` → always matches (no constraint).
* a dict (`{"$gte": 2020, "$lte": 2024}`) → every operator must hold (AND).
* a list/tuple/set → membership test (`actual in criterion`).
* any scalar → equality.

```python
from needle.retrieval.page_retriever_utils import matches_filter

matches_filter(2023, {"$gte": 2020, "$lte": 2024})  # True  (in range)
matches_filter(2019, {"$gte": 2020, "$lte": 2024})  # False (below range)
matches_filter("EU", {"$in": ["EU", "US"]})         # True
matches_filter(None, {"$exists": True})             # False (field absent)
```

Pages whose payload fails any field's criterion are masked out: their raw score is
set to `-inf` (so they can never enter the top-*k*) and their normalised score to
`0.0`. Everything else is ranked by the model score as usual.

## End-to-end

```python
from needle.retrieval.data import InputDocument
from needle.testing import DummyEmbedderRetriever
from needle_core.domain.interaction.query import Query
from needle_core.domain.metadata.filter_spec import FilterOperator

retriever = DummyEmbedderRetriever(index_path="demo.pkl")
retriever.index([
    InputDocument.from_path("a.png", metadata={"year": 2023, "region": "EU"}),
    InputDocument.from_path("b.png", metadata={"year": 2018, "region": "US"}),
])

query = (
    Query.of("revenue breakdown")
    .with_filter("year", 2020, FilterOperator.GTE)
    .with_filter("year", 2024, FilterOperator.LTE)
    .with_filter("region", ["EU", "US"], "in")
)

# Only the 2023 page survives the year range filter.
for hit in retriever.search(query, top_k=10):
    print(hit.document_version.filename, hit.page, round(hit.normalized_score, 3))
```
