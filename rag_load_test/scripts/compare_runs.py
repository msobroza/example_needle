#!/usr/bin/env python3
"""Compare locust runs: ``<prefix>_stats.csv`` files -> one markdown table per row.

Stdlib only, so it runs wherever the CSVs are (a laptop or a Domino job).

Example::

    python scripts/compare_runs.py results/monolith results/split-all \\
        --out results/comparison.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

AGGREGATED = "Aggregated"
STAGE_TYPE = "STAGE"
MISSING = "n/a"
BASELINE_DELTA = "—"
REQUIRED_COLUMNS: tuple[str, ...] = (
    "Type",
    "Name",
    "Request Count",
    "Failure Count",
    "Requests/s",
    "50%",
    "95%",
    "99%",
)


@dataclass(frozen=True)
class RowStats:
    """One ``*_stats.csv`` row reduced to the columns the comparison reports."""

    name: str
    request_type: str
    requests: int
    failures: int
    rps: float
    p50: float
    p95: float
    p99: float


def load_stats(prefix: Path) -> dict[str, RowStats]:
    """Read ``<prefix>_stats.csv`` keyed by ``"<Type> <Name>"``.

    The locust total row (``Name == "Aggregated"``) gets the key ``"Aggregated"``.

    Example::

        load_stats(Path("results/monolith"))["POST /query"].p95
    """
    path = Path(f"{prefix}_stats.csv")
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: not found, expected the locust --csv output <prefix>_stats.csv"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _check_columns(reader.fieldnames or (), path)
        rows = [_row_stats(record) for record in reader]
    return {_row_key(row): row for row in rows}


def _check_columns(found: Sequence[str], path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in found]
    if missing:
        raise ValueError(
            f"{path}: missing columns {missing}, "
            f"expected at least {list(REQUIRED_COLUMNS)}"
        )


def _row_stats(record: Mapping[str, str]) -> RowStats:
    return RowStats(
        name=record["Name"],
        request_type=record["Type"],
        requests=int(_number(record["Request Count"])),
        failures=int(_number(record["Failure Count"])),
        rps=_number(record["Requests/s"]),
        p50=_number(record["50%"]),
        p95=_number(record["95%"]),
        p99=_number(record["99%"]),
    )


def _number(cell: str) -> float:
    # locust writes "N/A" for the percentiles of a row with zero requests.
    if cell.strip() in ("", "N/A"):
        return 0.0
    return float(cell)


def _row_key(row: RowStats) -> str:
    if row.name == AGGREGATED:
        return AGGREGATED
    return f"{row.request_type} {row.name}"


def render_markdown(runs: dict[str, dict[str, RowStats]]) -> str:
    """Render one table per row key across all runs; Δp95 is against the first run.

    Example::

        print(render_markdown({"monolith": load_stats(a), "split": load_stats(b)}))
    """
    labels = list(runs)
    if not labels:
        raise ValueError("runs is empty, expected at least one run label -> stats")
    intro = f"Runs: {', '.join(labels)}. Δp95 is relative to `{labels[0]}`."
    lines = ["# Locust run comparison", "", intro, ""]
    for key in _ordered_keys(runs.values()):
        lines.extend(_table(key, labels, runs))
        lines.append("")
    return "\n".join(lines)


def _ordered_keys(stats: Iterable[Mapping[str, RowStats]]) -> list[str]:
    keys: set[str] = set().union(*(set(s) for s in stats))
    return sorted(keys, key=lambda key: (_group_rank(key), key))


def _group_rank(key: str) -> int:
    # Endpoint totals first, then per-stage rows, then the locust total row.
    if key == AGGREGATED:
        return 2
    if key.startswith(f"{STAGE_TYPE} "):
        return 1
    return 0


def _table(
    key: str, labels: Sequence[str], runs: Mapping[str, Mapping[str, RowStats]]
) -> list[str]:
    baseline = runs[labels[0]].get(key)
    header = (
        f"| run | requests | fail % | RPS | p50 | p95 | p99 | Δp95 vs {labels[0]} |"
    )
    rule = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [f"### {key}", "", header, rule]
    for label in labels:
        row = runs[label].get(key)
        delta = _delta_cell(row, baseline, is_baseline=label == labels[0])
        lines.append(f"| {label} | {_stat_cells(row)} | {delta} |")
    return lines


def _stat_cells(row: RowStats | None) -> str:
    if row is None:
        return " | ".join([MISSING] * 6)
    return (
        f"{row.requests} | {_fail_pct(row)} | {row.rps:.1f} | "
        f"{row.p50:.0f} | {row.p95:.0f} | {row.p99:.0f}"
    )


def _fail_pct(row: RowStats) -> str:
    if row.requests == 0:
        return MISSING
    return f"{100.0 * row.failures / row.requests:.1f}%"


def _delta_cell(
    row: RowStats | None, baseline: RowStats | None, *, is_baseline: bool
) -> str:
    if is_baseline:
        return BASELINE_DELTA
    if row is None or baseline is None:
        return MISSING
    delta_ms = row.p95 - baseline.p95
    if baseline.p95 == 0:
        return f"{delta_ms:+.1f} ms ({MISSING})"
    return f"{delta_ms:+.1f} ms ({100.0 * delta_ms / baseline.p95:+.1f}%)"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI: positional ``--csv`` prefixes plus ``--out``.

    Example::

        build_parser().parse_args(["results/monolith", "--out", "cmp.md"])
    """
    parser = argparse.ArgumentParser(
        description="Compare locust runs from their <prefix>_stats.csv files."
    )
    parser.add_argument(
        "prefixes",
        nargs="+",
        type=Path,
        help="locust --csv prefixes, e.g. results/monolith (reads <prefix>_stats.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the markdown here (default: stdout)",
    )
    return parser


def _run_labels(prefixes: Sequence[Path]) -> list[str]:
    labels = [prefix.name for prefix in prefixes]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate run labels {duplicates} from {[str(p) for p in prefixes]}, "
            "expected distinct prefix basenames"
        )
    return labels


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; the run label is each prefix's basename.

    Example::

        main(["results/monolith", "results/split-all", "--out", "cmp.md"])
    """
    args = build_parser().parse_args(argv)
    labels = _run_labels(args.prefixes)
    runs = {
        label: load_stats(prefix)
        for label, prefix in zip(labels, args.prefixes, strict=True)
    }
    markdown = render_markdown(runs)
    if args.out is None:
        print(markdown)
        return 0
    args.out.write_text(markdown, encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
