#!/usr/bin/env bash
# Domino App entry point for the workflow app of the split topologies.
#   split-reranker: reranker on OVMS (app_ovms_reranker.sh), embedder in-process.
#   split-all:      reranker AND embedder on OVMS (set RAG_EMBEDDER_BACKEND=ovms
#                   plus RAG_OVMS_EMBEDDINGS_URL in the App's environment variables).
# Required env: RAG_OVMS_RERANK_URL (URL of the OVMS reranker App, no trailing slash).
set -euo pipefail

cd "$(dirname "$0")/.."

# Domino only routes traffic to apps listening on 0.0.0.0:8888.
export RAG_HOST=0.0.0.0 RAG_PORT=8888

: "${RAG_OVMS_RERANK_URL:?app_workflow.sh: set RAG_OVMS_RERANK_URL to the OVMS reranker App URL (adapter appends /v3/rerank)}"

export RAG_RERANKER_BACKEND=ovms
export RAG_EMBEDDER_BACKEND=${RAG_EMBEDDER_BACKEND:-local}

default_topology=split-reranker
if [ "$RAG_EMBEDDER_BACKEND" = ovms ]; then
  : "${RAG_OVMS_EMBEDDINGS_URL:?app_workflow.sh: RAG_EMBEDDER_BACKEND=ovms needs RAG_OVMS_EMBEDDINGS_URL (URL of the OVMS embedder App, adapter appends /v3/embeddings)}"
  default_topology=split-all
fi
export RAG_TOPOLOGY=${RAG_TOPOLOGY:-$default_topology}

# App-to-app calls inside Domino need a bearer token; "domino" fetches one from
# http://localhost:8899/access-token (RAG_DOMINO_ACCESS_TOKEN_URL).
export RAG_OVMS_AUTH=${RAG_OVMS_AUTH:-domino}

exec rag-serve
