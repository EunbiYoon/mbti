#!/bin/bash
# LoRA train — writes runs/<timestamp>/lora/
#
# Usage:
#   bash lora/train.sh --smoke
#   bash lora/train.sh --games tictactoe --mbti INTJ ENFP
#   RUN_ID=20260715_120000 bash lora/train.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
source "${ROOT}/scripts/setup_env.sh"

SMOKE=0
GAMES=()
MBTI=()
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --game|--games) shift; while [[ $# -gt 0 && "$1" != --* ]]; do GAMES+=("$1"); shift; done ;;
    --mbti) shift; while [[ $# -gt 0 && "$1" != --* ]]; do MBTI+=("$1"); shift; done ;;
    --run_id) export RUN_ID="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

GEN_ARGS=()
TRAIN_ARGS=()
if [[ "${SMOKE}" == "1" ]]; then
  GEN_ARGS+=(--smoke)
  TRAIN_ARGS+=(--smoke)
fi
if [[ ${#GAMES[@]} -gt 0 ]]; then
  GEN_ARGS+=(--games "${GAMES[@]}")
fi
if [[ ${#MBTI[@]} -gt 0 ]]; then
  GEN_ARGS+=(--mbti "${MBTI[@]}")
fi

echo "=== generate SFT data ==="
python "${ROOT}/lora/generate_data.py" "${GEN_ARGS[@]}"

# Reuse same RUN_ID / latest lora dir for train.jsonl
echo "=== LoRA train ==="
python "${ROOT}/lora/train.py" "${TRAIN_ARGS[@]}" "${EXTRA[@]}"
