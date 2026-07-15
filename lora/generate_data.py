#!/usr/bin/env python3
"""Generate SFT jsonl from MBTI-style self-play on OpenSpiel games."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game import list_games, make_env, resolve_game
from opponent import list_types, get_mbti
from opponent.prompts import sft_rationale_prefix
from run_paths import new_lora_dir, write_latest_pointer


def episode_to_examples(ep: dict) -> list[dict]:
    env = make_env(ep["game"])
    examples = []
    # Rebuild states for prompts is heavy; use stored history + prompts at start
    agent_mbti = ep.get("agent_mbti")
    for turn in ep["history"]:
        if turn["player"] != ep.get("agent_player", 0):
            continue
        mbti = turn.get("mbti") or agent_mbti
        if mbti is None:
            continue
        prompt = env.get_prompt(player_id=turn["player"], mbti=mbti)
        obs_line = f"Observation / history context for seed={ep['seed']} turn_action={turn['action_str']}"
        user = (
            f"{prompt['user']}\n"
            f"CURRENT CONTEXT:\n{obs_line}\n"
            f"Choose one legal action. Preferred demonstration: {turn['action_str']}\n"
        )
        completion = (
            f"{sft_rationale_prefix(mbti)}\n"
            f"<answer>{turn['action_str']}</answer>"
        )
        examples.append(
            {
                "game": ep["game"],
                "mbti": get_mbti(mbti).code if isinstance(mbti, str) else mbti,
                "prompt": f"System: {prompt['system']}\n\nUser: {user}",
                "completion": completion,
            }
        )
    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=list_games())
    ap.add_argument("--mbti", nargs="+", default=None, help="MBTI codes (default: all 16)")
    ap.add_argument("--episodes", type=int, default=4, help="Episodes per game×mbti")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--out", default=None, help="Override output jsonl path")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.episodes = min(args.episodes, 1)
        args.games = [resolve_game(g) for g in (args.games[:1] or ["tictactoe"])]
        args.mbti = (args.mbti or ["INTJ", "ENFP"])[:2]

    games = [resolve_game(g) for g in args.games]
    types = args.mbti or list_types()
    rng = random.Random(args.seed)

    lora_dir = new_lora_dir(args.run_id)
    write_latest_pointer("lora", lora_dir)
    out_path = Path(args.out) if args.out else lora_dir / "train.jsonl"

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for game in games:
            for mbti in types:
                for i in range(args.episodes):
                    env = make_env(game, seed=args.seed + i)
                    opp = rng.choice(types)
                    ep = env.play_episode(
                        agent_player=0,
                        agent_mbti=mbti,
                        opponent_mbti=opp,
                        seed=args.seed + 1000 * (hash(game) % 1000) + i,
                    )
                    for ex in episode_to_examples(ep):
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        n += 1

    meta = {
        "games": games,
        "mbti": types,
        "episodes_per_pair": args.episodes,
        "num_examples": n,
        "train_file": str(out_path),
    }
    (lora_dir / "data_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {n} examples -> {out_path}")
    print(f"Run dir: {lora_dir.parent}")


if __name__ == "__main__":
    main()
