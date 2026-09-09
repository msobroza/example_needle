#!/usr/bin/env bash
# Domino App entry point, topology "monolith": embedder, Chroma, reranker,
# LangGraph workflow and the OpenAI client all run in this one process.
# Select this script as the entry point when publishing the App.
set -euo pipefail

cd "$(dirname "$0")/.."

# Domino only routes traffic to apps listening on 0.0.0.0:8888.
export RAG_HOST=0.0.0.0 RAG_PORT=8888
export RAG_TOPOLOGY=${RAG_TOPOLOGY:-monolith} RAG_EMBEDDER_BACKEND=local RAG_RERANKER_BACKEND=local

exec rag-serve
