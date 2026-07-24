#!/bin/bash
# LoRA train — writes runs/<timestamp>/lora/
# Model policy: --smoke → Qwen3-0.6B, otherwise → Qwen3-4B
# RUN_ID is auto-generated if unset.
#
# Usage:
#   bash lora/train.sh --smoke
#   bash lora/train.sh --games tictactoe kuhn_poker minihanabi
#   bash lora/train.sh --run_id my_run

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

# smoke → 0.6B, full → 4B
if [[ "${SMOKE}" == "1" ]]; then
  export PRETRAIN="${PRETRAIN_SMOKE:-${MODELS_ROOT}/Qwen3-0.6B}"
else
  export PRETRAIN="${PRETRAIN_FULL:-${MODELS_ROOT}/Qwen3-4B}"
fi
if [[ -n "${FORCE_PRETRAIN:-}" ]]; then
  export PRETRAIN="${FORCE_PRETRAIN}"
fi
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
echo "=== RUN_ID=${RUN_ID} → runs/${RUN_ID}/lora/ ==="
echo "=== PRETRAIN=${PRETRAIN} (smoke=${SMOKE}) ==="
if [[ ! -f "${PRETRAIN}/config.json" ]]; then
  echo "ERROR: model not found at ${PRETRAIN}" >&2
  exit 1
fi

GEN_ARGS=(--run_id "${RUN_ID}")
TRAIN_ARGS=(--run_id "${RUN_ID}" --base_model "${PRETRAIN}")
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

echo "=== LoRA train ==="
python "${ROOT}/lora/train.py" "${TRAIN_ARGS[@]}" "${EXTRA[@]}"

echo "=== done: ${ROOT}/runs/${RUN_ID}/lora ==="
echo "=== model was: ${PRETRAIN} ==="
