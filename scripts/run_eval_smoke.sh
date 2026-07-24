#!/bin/bash
# Smoke eval against an adapter (default: latest pointer / known smoke run).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ADAPTER="${ADAPTER:-}"
if [[ -n "$ADAPTER" ]]; then
  bash eval/eval.sh --smoke --adapter "$ADAPTER"
else
  bash eval/eval.sh --smoke
fi
echo "EVAL_DONE"
