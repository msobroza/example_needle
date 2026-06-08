"""``needle`` command-line interface.

Subcommands:

* ``needle list`` — list available retriever backends.
* ``needle index`` — build an index from one or more files.
* ``needle search`` — query an existing index.

Heavy imports (torch, model backends) happen inside the handlers so that
``needle --help`` and ``needle list`` stay fast.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="needle", description=__doc__)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list available retriever backends")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--retriever", default="colqwen2", help="retriever backend name"
    )
    common.add_argument("--index", default="index.pkl", help="path to the index file")

    p_index = sub.add_parser("index", parents=[common], help="index documents")
    p_index.add_argument("files", nargs="+", help="document paths to index")
    p_index.add_argument("--dpi", type=int, default=150)
    p_index.add_argument("--batch-size", type=int, default=4)

    p_search = sub.add_parser("search", parents=[common], help="search the index")
    p_search.add_argument("query", help="natural-language query text")
    p_search.add_argument("--top-k", type=int, default=5)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "list":
        from .retrieval.registry import available_retrievers

        for name in available_retrievers():
            print(name)
        return 0

    from .retrieval.data import InputDocument
    from .retrieval.registry import get_retriever

    retriever = get_retriever(
        args.retriever,
        index_path=args.index,
        **(
            {"dpi": args.dpi, "batch_size": args.batch_size}
            if args.command == "index"
            else {}
        ),
    )

    if args.command == "index":
        documents = [InputDocument.from_path(path) for path in args.files]
        retriever.index(documents)
        print(f"Indexed {len(retriever)} pages into {args.index}")
        return 0

    if args.command == "search":
        from conversational_core.domain.interaction.query import Query

        results = retriever.search(Query(query_text=args.query), top_k=args.top_k)
        retriever.show([r.to_annotation_result() for r in results])
        for rank, r in enumerate(results, 1):
            print(
                f"{rank}. [{r.score:.3f}] {r.document_version.filename} — page {r.page}"
            )
        return 0

    parser.error(f"unknown command: {args.command}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
