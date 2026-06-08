# Metrics

`needle.metrics` provides two small, dependency-light toolkits:

* **`evaluation`** — binary-relevance information-retrieval metrics over ranked
  id lists.
* **`timing`** — a `Timer` context manager and a `timed` decorator for
  wall-clock measurements.

Everything is re-exported from the package root, so import directly from
`needle.metrics`:

```python
from needle.metrics import (
    recall_at_k, precision_at_k, hit_rate_at_k,
    reciprocal_rank, mean_reciprocal_rank,
    average_precision, mean_average_precision,
    dcg_at_k, ndcg_at_k,
    Timer, timed,
)
```

## Evaluation metrics

Each metric takes a `ranked` sequence of item ids (most relevant first) and a
`relevant` collection of the ids considered relevant for the query. The relevance
model is **binary**: an id is either relevant or it is not. The helpers fall into
three families:

| Family | Functions | Scope |
| --- | --- | --- |
| Cut-off | `recall_at_k`, `precision_at_k`, `hit_rate_at_k`, `dcg_at_k`, `ndcg_at_k` | Top `k` results |
| Rank | `reciprocal_rank`, `average_precision` | Full ranking |
| Means | `mean_reciprocal_rank` (MRR), `mean_average_precision` (MAP) | Several queries |

The cut-off metrics validate that `k >= 1` and raise `ValueError` otherwise. Most
metrics return `0.0` when there are no relevant items; the mean helpers return
`0.0` for an empty query set and raise `ValueError` on a length mismatch between
`rankeds` and `relevants`.

### Worked example

Take a single ranking where positions 2 and 4 are the relevant items:

```python
ranked = [1, 2, 3, 4]
relevant = {2, 4}
```

| Call | Result | Why |
| --- | --- | --- |
| `recall_at_k(ranked, relevant, 2)` | `0.5` | 1 of 2 relevant items in the top 2. |
| `recall_at_k(ranked, relevant, 4)` | `1.0` | Both relevant items in the top 4. |
| `precision_at_k(ranked, relevant, 2)` | `0.5` | 1 hit / `k=2` (denominator is always `k`). |
| `precision_at_k(ranked, relevant, 4)` | `0.5` | 2 hits / `k=4`. |
| `hit_rate_at_k(ranked, relevant, 1)` | `0.0` | No relevant item in the top 1. |
| `hit_rate_at_k(ranked, relevant, 2)` | `1.0` | A relevant item appears in the top 2. |
| `reciprocal_rank(ranked, relevant)` | `0.5` | First relevant item is at rank 2 → `1/2`. |
| `average_precision(ranked, relevant)` | `0.5` | Mean of precisions at the hit ranks: `(1/2 + 2/4) / 2`. |
| `dcg_at_k(ranked, relevant, 4)` | `1.0616` | `1/log2(3) + 1/log2(5)` (hits at ranks 2 and 4). |
| `ndcg_at_k(ranked, relevant, 4)` | `0.6509` | `DCG / IDCG`, where `IDCG = 1 + 1/log2(3) = 1.6309`. |

The means simply average a metric over several queries; with one query they equal
the single-query value:

```python
mean_reciprocal_rank([ranked], [relevant])      # 0.5
mean_average_precision([ranked], [relevant])     # 0.5
```

### Metric reference

* `recall_at_k(ranked, relevant, k)` — fraction of relevant items retrieved within
  the top `k`.
* `precision_at_k(ranked, relevant, k)` — fraction of the top `k` that are
  relevant; the denominator is `k` (the standard precision@k definition).
* `hit_rate_at_k(ranked, relevant, k)` — `1.0` if any relevant item appears in the
  top `k`, else `0.0`.
* `reciprocal_rank(ranked, relevant)` — reciprocal of the 1-based rank of the first
  relevant item (`0.0` when none is present).
* `average_precision(ranked, relevant)` — mean of the precision values computed at
  each rank where a relevant item is found, divided by the number of relevant
  items.
* `dcg_at_k(ranked, relevant, k)` — discounted cumulative gain with the standard
  `gain / log2(rank + 1)` discount and binary gain.
* `ndcg_at_k(ranked, relevant, k)` — DCG divided by the ideal DCG (IDCG); lies in
  `[0, 1]` and is `1.0` when the top `k` are exactly the relevant items.
* `mean_reciprocal_rank(rankeds, relevants)` — mean of `reciprocal_rank` (MRR).
* `mean_average_precision(rankeds, relevants)` — mean of `average_precision` (MAP).

## Timing

### `Timer`

`Timer` is a context manager that measures elapsed wall-clock time
(`time.perf_counter`). It exposes `elapsed` (seconds) and `elapsed_ms`
(milliseconds):

```python
from needle.metrics import Timer

with Timer() as t:
    do_work()

print(t.elapsed, "s", t.elapsed_ms, "ms")
```

`elapsed` reflects the time between `__enter__` and `__exit__`; while the block is
still running it reflects the time since entry. Reading `elapsed` before the timer
has been started raises `RuntimeError`.

The application services use it to report how long a search took — see
`needle.application.services.search_service.SearchService`, whose `SearchResult`
carries a `took_ms` field.

### `timed`

`timed` is a decorator that logs the wall time of a function at `DEBUG` level and
records the most recent duration on the wrapper as `.last_seconds` (`None` before
the first call). The wrapped callable returns its original value unchanged:

```python
from needle.metrics import timed

@timed
def embed_all(images):
    ...

embed_all(images)
print(embed_all.last_seconds)   # duration of the last call, in seconds
```
