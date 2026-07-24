#!/bin/bash
# Eval — writes runs/<timestamp>/eval/ (sibling of lora/ when --adapter is set).
#
# Usage:
#   bash eval/eval.sh --smoke
#   bash eval/eval.sh --adapter runs/<ts>/lora
#   bash eval/eval.sh --games tictactoe --mbti INTJ

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
source "${ROOT}/scripts/setup_env.sh"

SMOKE=0
GAMES=()
MBTI=()
ADAPTER=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --game|--games) shift; while [[ $# -gt 0 && "$1" != --* ]]; do GAMES+=("$1"); shift; done ;;
    --mbti) shift; while [[ $# -gt 0 && "$1" != --* ]]; do MBTI+=("$1"); shift; done ;;
    --adapter) ADAPTER="$2"; shift 2 ;;
    --run_id) export RUN_ID="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

ARGS=()
if [[ "${SMOKE}" == "1" ]]; then
  ARGS+=(--smoke)
fi
if [[ -n "${ADAPTER}" ]]; then
  ARGS+=(--adapter "${ADAPTER}")
fi
if [[ -n "${RUN_ID:-}" ]]; then
  ARGS+=(--run_id "${RUN_ID}")
fi
if [[ ${#GAMES[@]} -gt 0 ]]; then
  ARGS+=(--games "${GAMES[@]}")
fi
if [[ ${#MBTI[@]} -gt 0 ]]; then
  ARGS+=(--agent_mbti "${MBTI[@]}")
fi

echo "=== Eval ==="
python "${ROOT}/eval/evaluate.py" "${ARGS[@]}" "${EXTRA[@]}"
