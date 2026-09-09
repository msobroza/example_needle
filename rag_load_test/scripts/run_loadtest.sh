#!/usr/bin/env bash
# Usage: scripts/run_loadtest.sh <run-name> <host> [users=20] [spawn-rate=5] [run-time=2m]
# Writes results/<run-name>_stats.csv, _failures.csv and .html; prints the stats path.
set -euo pipefail
NAME=${1:?run name}; HOST=${2:?host url}; USERS=${3:-20}; SPAWN=${4:-5}; TIME=${5:-2m}
cd "$(dirname "$0")/.."
mkdir -p results
locust -f loadtest/locustfile.py --headless --host "$HOST" -u "$USERS" -r "$SPAWN" -t "$TIME" \
  --csv "results/$NAME" --html "results/$NAME.html" --only-summary
echo "results/${NAME}_stats.csv"
