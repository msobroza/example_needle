"""``python -m rag_load_test.api`` / ``rag-serve``: run the service under uvicorn.

Example::

    rag-serve --host 0.0.0.0 --port 8888     # RAG_* env vars supply the rest
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from typing import Any

import uvicorn

from ..settings import RagSettings
from .app import create_app


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse ``--host``/``--port``; both default to ``RAG_HOST``/``RAG_PORT``.

    Example::

        parse_args(["--port", "9000"]).port  # 9000
    """
    parser = argparse.ArgumentParser(
        prog="rag-serve",
        description="Serve the RAG workflow over HTTP (settings from RAG_* env vars).",
    )
    parser.add_argument("--host", help="bind address (default: RAG_HOST, 0.0.0.0)")
    parser.add_argument("--port", type=int, help="bind port (default: RAG_PORT, 8888)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build settings (CLI overrides env), create the app and block in uvicorn.

    Example::

        raise SystemExit(main())
    """
    args = parse_args(argv)
    cli_overrides: dict[str, Any] = {
        key: value
        for key, value in {"host": args.host, "port": args.port}.items()
        if value is not None
    }
    settings = RagSettings.from_env(**cli_overrides)
    log_level = settings.log_level.lower()
    _configure_logging(log_level)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=log_level,
    )
    return 0


def _configure_logging(log_level: str) -> None:
    # uvicorn configures only its own loggers; without a root handler the
    # rag_load_test.api JSON lines (INFO) would never reach stdout.
    logging.basicConfig(level=log_level.upper(), format="%(message)s")


if __name__ == "__main__":
    raise SystemExit(main())
