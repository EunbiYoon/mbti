# MBTI LoRA agents (Mini-Hanabi / TicTacToe / Kuhn Poker)

Uses the **`marshal`** conda env and weights under `../models/`.

```
runs/<timestamp>/
  lora/    # train.jsonl, adapter, run_info.json
  eval/    # metrics.json, result.md, episodes.jsonl
  table1/  # table1.md, results.json (downstream reasoning)
```

`RUN_ID` is created automatically — no need to `export` it.

---

## Commands

```bash
cd /scratch/workspace/eunbiyoon_umass_edu-paper/mbti
```

### Smoke (train + eval)

```bash
bash scripts/smoke.sh
```

### Train

```bash
# smoke → Qwen3-0.6B
bash lora/train.sh --smoke --games tictactoe --mbti INTJ

# full → Qwen3-4B
bash lora/train.sh --games tictactoe kuhn_poker minihanabi
```

→ `runs/<timestamp>/lora/`

Full 4B runs default to 200 training steps and save checkpoints every 20 steps
(`checkpoint-20` ... `checkpoint-200`).

### Eval

```bash
# smoke (uses latest lora if --adapter omitted)
bash eval/eval.sh --smoke

# full
bash eval/eval.sh --adapter runs/<timestamp>/lora
```

→ `runs/<timestamp>/eval/` (`metrics.json`, `result.md`, `episodes.jsonl` on full)

Optional: `--games` · `--mbti` · `--run_id` (same style as train).

### Table 1 (downstream reasoning)

```bash
bash eval/table1.sh --run_dir runs/<timestamp>
```

→ `runs/<timestamp>/table1/` (`table1.md`, `results.json`)

Quick check: `TABLE1_LIMIT=20 bash eval/table1.sh --run_dir runs/<timestamp>`

---

## What eval reports

1. **Bot matchups** — MBTI heuristic vs MBTI opponent  
2. **LoRA matchups** — adapter on agent seats vs MBTI bot  
3. **Format check** — share of turns with legal `<answer>...</answer>`

Pointers: `runs/latest_lora.json`, `runs/latest_eval.json`

---

## Env (optional)

| var | default |
|-----|---------|
| `CONDA_ENV` | `marshal` |
| `PRETRAIN_SMOKE` | `../models/Qwen3-0.6B` (`--smoke`) |
| `PRETRAIN_FULL` | `../models/Qwen3-4B` (full train) |
| `LORA_RANK` / `LORA_ALPHA` / `LORA_DROPOUT` | `16` / `32` / `0.05` |
| `RUN_ID` | auto timestamp |
| `FORCE_PRETRAIN` | optional path override |