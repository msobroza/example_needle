"""Tests for the ``needle`` command-line interface."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from needle.cli import main  # noqa: E402
from needle.retrieval import registry  # noqa: E402
from needle.testing import DummyEmbedderRetriever  # noqa: E402

pytestmark = pytest.mark.torch


def test_list_command(capsys):
    exit_code = main(["list"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "colqwen2" in out
    assert "colpali" in out


def test_no_command_errors():
    with pytest.raises(SystemExit):
        main([])


def test_index_then_search_round_trip(monkeypatch, capsys, png_factory, tmp_path):
    monkeypatch.setitem(registry.RETRIEVERS, "dummy", DummyEmbedderRetriever)
    page = png_factory("doc.png", "white", "REPORT")
    idx = tmp_path / "cli.pkl"

    code = main(["index", str(page), "--retriever", "dummy", "--index", str(idx)])
    assert code == 0
    assert idx.exists()
    assert "Indexed 1 pages" in capsys.readouterr().out

    code = main(
        [
            "search",
            "report",
            "--retriever",
            "dummy",
            "--index",
            str(idx),
            "--top-k",
            "1",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "doc.png" in out
    assert "page 1" in out
