#!/bin/bash
# Eval — writes runs/<timestamp>/eval/ (reuse RUN_ID to nest under same run as LoRA)
#
# Usage:
#   bash eval/eval.sh --smoke
#   RUN_ID=20260715_120000 bash eval/eval.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
source "${ROOT}/scripts/setup_env.sh"

python "${ROOT}/eval/evaluate.py" "$@"
