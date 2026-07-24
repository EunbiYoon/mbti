#!/bin/bash
set -euo pipefail
ROOT=/scratch/workspace/eunbiyoon_umass_edu-paper/mbti
cd "$ROOT"
# One GPU; avoid dual-GPU hang after prior CUDA assert
export CUDA_VISIBLE_DEVICES=0
# Kill any leftover evaluate from a suspended session
pkill -f "mbti/eval/evaluate.py" 2>/dev/null || true
sleep 2

bash eval/eval.sh \
  --adapter runs/20260715_125114/lora \
  --games tictactoe kuhn_poker minihanabi \
  --agent_mbti INTJ \
  --opponent_mbti ISTJ \
  --episodes 3 \
  --save_episodes \
  2>&1 | tee runs/_m2_eval_125114.log
echo M2_EVAL_DONE
ls -la runs/20260715_125114/
ls -la runs/20260715_125114/eval/
cat runs/20260715_125114/eval/result.md
