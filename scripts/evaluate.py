#!/usr/bin/env python3
"""Evaluate ranked retrieval results against a gold file.

The gold file is JSON: a list of objects with ``ranked`` (list of ids) and
``relevant`` (list of ids). Run the built-in demo with ``--demo``::

    python scripts/evaluate.py --demo
    python scripts/evaluate.py gold.json --k 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from needle.metrics import (
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

_DEMO = [
    {"ranked": [3, 1, 2, 8, 5], "relevant": [1, 2]},
    {"ranked": [4, 7, 9, 2, 6], "relevant": [2]},
    {"ranked": [1, 2, 3], "relevant": [1, 2, 3]},
]


def load_gold(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", nargs="?", type=Path, help="path to gold JSON")
    parser.add_argument("--demo", action="store_true", help="use the built-in demo")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    if args.demo or args.gold is None:
        rows = _DEMO
    else:
        rows = load_gold(args.gold)

    rankeds = [r["ranked"] for r in rows]
    relevants = [set(r["relevant"]) for r in rows]
    k = args.k

    macro_recall = sum(
        recall_at_k(r, rel, k) for r, rel in zip(rankeds, relevants, strict=True)
    ) / len(rows)
    macro_prec = sum(
        precision_at_k(r, rel, k) for r, rel in zip(rankeds, relevants, strict=True)
    ) / len(rows)
    macro_ndcg = sum(
        ndcg_at_k(r, rel, k) for r, rel in zip(rankeds, relevants, strict=True)
    ) / len(rows)

    print(f"queries     : {len(rows)}")
    print(f"recall@{k}    : {macro_recall:.3f}")
    print(f"precision@{k} : {macro_prec:.3f}")
    print(f"nDCG@{k}      : {macro_ndcg:.3f}")
    print(f"MRR         : {mean_reciprocal_rank(rankeds, relevants):.3f}")
    print(f"MAP         : {mean_average_precision(rankeds, relevants):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
