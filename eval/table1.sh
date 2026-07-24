#!/bin/bash
# Table 1 downstream reasoning eval → runs/<timestamp>/table1/
#
# Usage:
#   bash eval/table1.sh --run_dir runs/<timestamp>
#   TABLE1_LIMIT=20 bash eval/table1.sh --run_dir runs/<timestamp>   # quick check

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export ROOT
cd "${ROOT}"

# shellcheck disable=SC1091
source "${ROOT}/scripts/setup_env.sh"

RUN_DIR=""
LIMIT="${TABLE1_LIMIT:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_dir|--run-dir) RUN_DIR="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: bash eval/table1.sh --run_dir runs/<timestamp>" >&2
  exit 1
fi
if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${ROOT}/${RUN_DIR}"
fi
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"

OUT_DIR="${RUN_DIR}/table1"
mkdir -p "${OUT_DIR}"

echo "=== Table 1 eval ==="
echo "RUN_DIR=${RUN_DIR}"
echo "OUT=${OUT_DIR}"
echo "LIMIT=${LIMIT}"

python -m pip install -q datasets 2>/dev/null || true

python "${ROOT}/eval/table1_eval.py" \
  --run-dir "${RUN_DIR}" \
  --out-dir "${OUT_DIR}" \
  --limit "${LIMIT}" \
  2>&1 | tee "${OUT_DIR}/console.log"

echo "=== done: ${OUT_DIR}/table1.md ==="
