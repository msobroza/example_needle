#!/usr/bin/env bash
# Domino App entry point: OpenVINO Model Server serving the embedder
# (bge-small-en-v1.5, int8, CLS pooling) on 0.0.0.0:8888 as POST /v3/embeddings.
# Expects the repository produced by ovms/export_models.sh in a Domino Dataset
# mounted at RAG_OVMS_MODELS_DIR (default /mnt/data/ovms-models); the Domino
# environment must be built from domino/environment/Dockerfile.ovms.
set -euo pipefail

cd "$(dirname "$0")/.."

# Domino only routes traffic to apps listening on 0.0.0.0:8888.
export RAG_HOST=0.0.0.0 RAG_PORT=8888

MODELS_DIR=${RAG_OVMS_MODELS_DIR:-/mnt/data/ovms-models}
CONFIG="$MODELS_DIR/config_embedder.json"

if [ ! -f "$CONFIG" ]; then
  echo "app_ovms_embedder.sh: OVMS config not found: $CONFIG" >&2
  echo "  expected the model repository from ovms/export_models.sh in a Domino Dataset mounted at $MODELS_DIR (override with RAG_OVMS_MODELS_DIR)" >&2
  exit 1
fi

if ! command -v ovms >/dev/null 2>&1; then
  echo "app_ovms_embedder.sh: 'ovms' binary not on PATH; build the Domino environment from domino/environment/Dockerfile.ovms" >&2
  exit 1
fi

exec ovms --rest_port 8888 --config_path "$CONFIG"
