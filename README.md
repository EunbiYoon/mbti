# MBTI LoRA agents on Mini-Hanabi / TicTacToe / Kuhn Poker

Self-contained project. Reuses the **`marshal`** conda env and shared weights under `../models/`.

## Layout

```
mbti/
  opponent/     # 16 MBTI type defs, prompts, heuristic policies
  game/         # OpenSpiel: tictactoe, kuhn_poker, minihanabi
  lora/         # SFT data gen + PEFT LoRA train
  eval/         # bot matchups + LoRA format check
  runs/<timestamp>/
    lora/       # train.jsonl, adapter, run_info.json
    eval/       # metrics.json, result.md
  run_paths.py
  scripts/
```

## Quick start (GPU node)

```bash
cd /scratch/workspace/eunbiyoon_umass_edu-paper/mbti
bash scripts/smoke.sh
```

Or step by step:

```bash
export RUN_ID=$(date -u +%Y%m%d_%H%M%S)
bash lora/train.sh --smoke --games tictactoe --mbti INTJ
RUN_ID=$RUN_ID bash eval/eval.sh --smoke
```

## Full-ish train

```bash
export RUN_ID=$(date -u +%Y%m%d_%H%M%S)
bash lora/train.sh --games tictactoe kuhn_poker minihanabi
RUN_ID=$RUN_ID bash eval/eval.sh
```

## Env

- Conda: `marshal` (override with `CONDA_ENV`)
- Model: `PRETRAIN` / `../models/Qwen3-0.6B` (smoke) or `Qwen3-4B`
- LoRA: `LORA_RANK=16` `LORA_ALPHA=32` `LORA_DROPOUT=0.05`
# mbti
