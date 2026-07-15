#!/bin/bash
# End-to-end smoke: games → data → LoRA → eval under one timestamp.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_smoke}"

echo "=== smoke RUN_ID=${RUN_ID} ==="
bash "${ROOT}/lora/train.sh" --smoke --games tictactoe kuhn_poker minihanabi --mbti INTJ ENFP
bash "${ROOT}/eval/eval.sh" --smoke --run_id "${RUN_ID}"

echo "=== smoke done: ${ROOT}/runs/${RUN_ID} ==="
ls -la "${ROOT}/runs/${RUN_ID}"
ls -la "${ROOT}/runs/${RUN_ID}/lora" | head
ls -la "${ROOT}/runs/${RUN_ID}/eval" | head
