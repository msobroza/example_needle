#!/usr/bin/env python3
"""Build a retrieval index from a directory of documents.

Examples
--------
Real backend (needs ``pip install -e ".[retrieval]"`` and ``[pdf]`` for PDFs)::

    python scripts/build_index.py ./corpus --retriever colqwen2 --index corpus.pkl

Offline smoke run with the deterministic dummy embedder::

    python scripts/build_index.py ./corpus --dummy --index corpus.pkl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from needle.retrieval.data import InputDocument
from needle_core.domain.types import DocumentExtension

_SUPPORTED_SUFFIXES = {f".{ext.value}" for ext in DocumentExtension}


def discover(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    )


def build_retriever(args: argparse.Namespace):
    if args.dummy:
        from needle.testing import DummyEmbedderRetriever

        return DummyEmbedderRetriever(
            index_path=args.index, dpi=args.dpi, batch_size=args.batch_size
        )
    from needle.retrieval.registry import get_retriever

    return get_retriever(
        args.retriever, index_path=args.index, dpi=args.dpi, batch_size=args.batch_size
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory to scan for documents")
    parser.add_argument("--retriever", default="colqwen2", help="retriever backend")
    parser.add_argument("--dummy", action="store_true", help="use the dummy embedder")
    parser.add_argument("--index", default="index.pkl", help="output index path")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    files = discover(args.root)
    if not files:
        parser.error(f"no supported documents found under {args.root}")
    print(f"Found {len(files)} document(s) under {args.root}")

    retriever = build_retriever(args)
    documents = [InputDocument.from_path(p) for p in files]
    retriever.index(documents)
    print(f"Indexed {len(retriever)} pages -> {args.index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
