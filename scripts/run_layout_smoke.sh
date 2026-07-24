#!/bin/bash
# One timestamp, both folders: runs/$RUN_ID/{lora,eval}
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_layout}"
echo "=== layout smoke RUN_ID=$RUN_ID ==="
bash lora/train.sh --smoke --games tictactoe --mbti INTJ
bash eval/eval.sh --smoke --run_id "$RUN_ID" --adapter "runs/$RUN_ID/lora"
test -d "runs/$RUN_ID/lora" && test -d "runs/$RUN_ID/eval"
test -f "runs/$RUN_ID/lora/adapter_config.json"
test -f "runs/$RUN_ID/eval/metrics.json"
echo "LAYOUT_OK runs/$RUN_ID/{lora,eval}"
