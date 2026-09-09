"""``scripts/compare_runs.py``: locust ``*_stats.csv`` files -> markdown tables."""

from __future__ import annotations

import csv
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_runs.py"

# Exact header locust 2.46 writes to <prefix>_stats.csv.
HEADER = [
    "Type", "Name", "Request Count", "Failure Count", "Median Response Time",
    "Average Response Time", "Min Response Time", "Max Response Time",
    "Average Content Size", "Requests/s", "Failures/s", "50%", "66%", "75%",
    "80%", "90%", "95%", "98%", "99%", "99.9%", "99.99%", "100%",
]  # fmt: skip


@pytest.fixture(scope="module")
def compare_runs() -> ModuleType:
    name = "compare_runs_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve ``from __future__`` annotations via sys.modules[__module__].
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _row(
    kind: str, name: str, requests: int, failures: int, rps: float,
    p50: object, p95: object, p99: object,
) -> list[str]:  # fmt: skip
    cells = {col: "0" for col in HEADER}
    cells.update({"Type": kind, "Name": name, "Request Count": str(requests)})
    cells.update({"Failure Count": str(failures), "Requests/s": str(rps)})
    cells.update({"50%": str(p50), "95%": str(p95), "99%": str(p99)})
    return [cells[col] for col in HEADER]


def _write_stats(
    prefix: Path, rows: Sequence[Sequence[str]], header: Sequence[str] = HEADER
) -> Path:
    path = Path(f"{prefix}_stats.csv")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


@pytest.fixture
def two_runs(tmp_path: Path) -> tuple[Path, Path]:
    monolith = tmp_path / "monolith"
    split = tmp_path / "split-reranker"
    _write_stats(monolith, [
        _row("POST", "/query", 100, 0, 10.0, 50, 120, 200),
        _row("POST", "/retrieve", 300, 3, 30.0, 20, 40, 60),
        _row("STAGE", "/query:rerank", 100, 0, 10.0, 15, 30, 45),
        _row("", "Aggregated", 400, 3, 40.0, 25, 100, 190),
    ])  # fmt: skip
    _write_stats(split, [
        _row("POST", "/query", 90, 9, 9.0, 55, 132, 210),
        _row("POST", "/retrieve", 270, 0, 27.0, 22, 44, 66),
        _row("STAGE", "/query:rerank", 90, 0, 9.0, 20, 45, 60),
        _row("", "Aggregated", 360, 9, 36.0, 30, 110, 200),
    ])  # fmt: skip
    return monolith, split


def _section(markdown: str, heading: str) -> str:
    """Return the body of the ``### <heading>`` block."""
    marker = f"### {heading}\n"
    assert marker in markdown, f"missing section {heading!r} in:\n{markdown}"
    rest = markdown.split(marker, 1)[1]
    return rest.split("\n### ", 1)[0]


# --- load_stats -------------------------------------------------------------


def test_load_stats_keys_and_values(
    compare_runs: ModuleType, two_runs: tuple[Path, Path]
) -> None:
    stats = compare_runs.load_stats(two_runs[0])
    assert set(stats) == {
        "POST /query",
        "POST /retrieve",
        "STAGE /query:rerank",
        "Aggregated",
    }
    assert stats["POST /query"] == compare_runs.RowStats(
        name="/query", request_type="POST", requests=100, failures=0,
        rps=10.0, p50=50.0, p95=120.0, p99=200.0,
    )  # fmt: skip
    assert stats["Aggregated"].request_type == ""
    assert stats["Aggregated"].requests == 400


def test_load_stats_missing_file_names_expected_path(
    compare_runs: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="nope_stats.csv"):
        compare_runs.load_stats(tmp_path / "nope")


def test_load_stats_treats_na_percentiles_as_zero(
    compare_runs: ModuleType, tmp_path: Path
) -> None:
    prefix = tmp_path / "empty"
    _write_stats(prefix, [_row("POST", "/query", 0, 0, 0.0, "N/A", "N/A", "N/A")])
    row = compare_runs.load_stats(prefix)["POST /query"]
    assert (row.p50, row.p95, row.p99) == (0.0, 0.0, 0.0)


def test_load_stats_missing_column_raises_with_name(
    compare_runs: ModuleType, tmp_path: Path
) -> None:
    prefix = tmp_path / "broken"
    header = [col for col in HEADER if col != "95%"]
    _write_stats(
        prefix, [["POST", "/query"] + ["1"] * (len(header) - 2)], header=header
    )
    with pytest.raises(ValueError, match="95%"):
        compare_runs.load_stats(prefix)


# --- render_markdown --------------------------------------------------------


def test_render_markdown_query_table_has_both_runs_and_delta(
    compare_runs: ModuleType, two_runs: tuple[Path, Path]
) -> None:
    monolith, split = two_runs
    runs = {
        "monolith": compare_runs.load_stats(monolith),
        "split-reranker": compare_runs.load_stats(split),
    }
    markdown = compare_runs.render_markdown(runs)
    query = _section(markdown, "POST /query")
    assert "| monolith |" in query
    assert "| split-reranker |" in query
    assert "Δp95" in query
    assert "+12.0 ms (+10.0%)" in query
    assert "| 100 | 0.0% | 10.0 | 50 | 120 | 200 |" in query
    assert "| 90 | 10.0% | 9.0 | 55 | 132 | 210 |" in query


def test_render_markdown_has_stage_table(
    compare_runs: ModuleType, two_runs: tuple[Path, Path]
) -> None:
    runs = {p.name: compare_runs.load_stats(p) for p in two_runs}
    stage = _section(compare_runs.render_markdown(runs), "STAGE /query:rerank")
    assert "+15.0 ms (+50.0%)" in stage


def test_render_markdown_orders_endpoints_then_stages_then_aggregated(
    compare_runs: ModuleType, two_runs: tuple[Path, Path]
) -> None:
    runs = {p.name: compare_runs.load_stats(p) for p in two_runs}
    markdown = compare_runs.render_markdown(runs)
    order = [
        markdown.index(f"### {h}\n")
        for h in ("POST /query", "POST /retrieve", "STAGE /query:rerank", "Aggregated")
    ]
    assert order == sorted(order)


def test_render_markdown_marks_rows_missing_from_a_run(
    compare_runs: ModuleType, tmp_path: Path
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_stats(first, [_row("POST", "/query", 10, 0, 1.0, 5, 10, 20)])
    _write_stats(second, [_row("POST", "/retrieve", 10, 0, 1.0, 5, 10, 20)])
    runs = {
        "first": compare_runs.load_stats(first),
        "second": compare_runs.load_stats(second),
    }
    markdown = compare_runs.render_markdown(runs)
    assert "n/a" in _section(markdown, "POST /query")
    assert "n/a" in _section(markdown, "POST /retrieve")


def test_render_markdown_delta_handles_zero_baseline(
    compare_runs: ModuleType, tmp_path: Path
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_stats(first, [_row("POST", "/query", 10, 0, 1.0, 0, 0, 0)])
    _write_stats(second, [_row("POST", "/query", 10, 0, 1.0, 5, 10, 20)])
    runs = {
        "first": compare_runs.load_stats(first),
        "second": compare_runs.load_stats(second),
    }
    assert "+10.0 ms (n/a)" in _section(
        compare_runs.render_markdown(runs), "POST /query"
    )


# --- main -------------------------------------------------------------------


def test_main_writes_out_file(
    compare_runs: ModuleType, two_runs: tuple[Path, Path], tmp_path: Path
) -> None:
    monolith, split = two_runs
    out = tmp_path / "comparison.md"
    assert compare_runs.main([str(monolith), str(split), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "### POST /query\n" in text
    assert "| monolith |" in text and "| split-reranker |" in text


def test_main_prints_to_stdout_without_out(
    compare_runs: ModuleType,
    two_runs: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monolith, split = two_runs
    assert compare_runs.main([str(monolith), str(split)]) == 0
    assert "### POST /query" in capsys.readouterr().out
