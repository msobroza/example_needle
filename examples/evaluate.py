"""Compute retrieval-quality metrics on hand-labelled rankings.

    python examples/evaluate.py

Shows the metrics in :mod:`needle.metrics` without needing any model: each
"query" is a ranked list of document ids plus the set of relevant ids.
"""

from __future__ import annotations

from needle.metrics import (
    average_precision,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Each tuple: (ranked result ids, set of relevant ids).
GOLD = [
    ([3, 1, 2, 8, 5], {1, 2}),
    ([4, 7, 9, 2, 6], {2}),
    ([1, 2, 3], {1, 2, 3}),
]


def main() -> None:
    rankeds = [ranked for ranked, _ in GOLD]
    relevants = [relevant for _, relevant in GOLD]

    print("Per-query metrics (k=3):")
    header = (
        f"  {'ranked':<18}{'recall@3':>10}{'prec@3':>9}"
        f"{'RR':>7}{'AP':>7}{'nDCG@3':>9}"
    )
    print(header)
    for ranked, relevant in GOLD:
        print(
            f"  {str(ranked):<18}"
            f"{recall_at_k(ranked, relevant, 3):>10.3f}"
            f"{precision_at_k(ranked, relevant, 3):>9.3f}"
            f"{reciprocal_rank(ranked, relevant):>7.3f}"
            f"{average_precision(ranked, relevant):>7.3f}"
            f"{ndcg_at_k(ranked, relevant, 3):>9.3f}"
        )

    print("\nCorpus-level metrics:")
    print(f"  MRR  = {mean_reciprocal_rank(rankeds, relevants):.3f}")
    print(f"  MAP  = {mean_average_precision(rankeds, relevants):.3f}")


if __name__ == "__main__":
    main()
