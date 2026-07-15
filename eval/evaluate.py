#!/usr/bin/env python3
"""Eval heuristic / LoRA agents vs MBTI opponents; writes runs/<ts>/eval/."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game import list_games, make_env, resolve_game
from opponent import list_types
from run_paths import new_eval_dir, read_latest_path, write_latest_pointer


def run_bot_matchup(
    game: str,
    *,
    agent_mbti: str,
    opponent_mbti: str,
    episodes: int,
    seed: int,
) -> Dict[str, Any]:
    returns_agent = []
    wins = 0
    draws = 0
    for i in range(episodes):
        env = make_env(game, seed=seed + i)
        ep = env.play_episode(
            agent_player=0,
            agent_mbti=agent_mbti,
            opponent_mbti=opponent_mbti,
            seed=seed + i,
        )
        r0 = float(ep["rewards"][0]) if ep["rewards"] else 0.0
        returns_agent.append(r0)
        info = ep.get("info") or {}
        if env.zero_sum:
            w = info.get("winner", -1)
            if w == 0:
                wins += 1
            elif w == -1:
                draws += 1
        else:
            # coop: treat positive score as success proxy
            if r0 > 0:
                wins += 1
    return {
        "game": game,
        "agent_mbti": agent_mbti,
        "opponent_mbti": opponent_mbti,
        "episodes": episodes,
        "mean_return": sum(returns_agent) / max(1, len(returns_agent)),
        "wins": wins,
        "draws": draws,
        "returns": returns_agent,
    }


def try_lora_legal_action_rate(
    adapter_dir: Path,
    base_model: str,
    game: str,
    mbti: str,
    *,
    n: int = 3,
) -> Optional[Dict[str, Any]]:
    """Lightweight generation check: model outputs contain <answer> tags."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        return {"error": f"import failed: {e}"}

    if not (adapter_dir / "adapter_config.json").is_file():
        return {"error": f"no adapter at {adapter_dir}"}

    tok = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    env = make_env(game)
    step = env.reset(0)
    prompt = env.get_prompt(player_id=0, mbti=mbti)
    legal = list(step.legal_actions.values())
    user = (
        f"{prompt['user']}\nCURRENT STATE:\n{step.observation}\n"
        f"LEGAL ACTIONS: {', '.join(legal)}\n"
    )
    messages = f"System: {prompt['system']}\n\nUser: {user}\n\nANSWER:\n"
    inputs = tok(messages, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    ok = 0
    samples = []
    with torch.no_grad():
        for _ in range(n):
            out = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.7)
            text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            samples.append(text)
            if "<answer>" in text.lower() and "</answer>" in text.lower():
                ok += 1
    return {
        "adapter": str(adapter_dir),
        "game": game,
        "mbti": mbti,
        "format_ok": ok,
        "n": n,
        "samples": samples[:3],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=list_games())
    ap.add_argument("--agent_mbti", nargs="+", default=["INTJ", "ENFP"])
    ap.add_argument("--opponent_mbti", nargs="+", default=["ISTJ", "ENTP"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_id", default=None, help="Share timestamp with lora run")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (default: latest)")
    ap.add_argument("--base_model", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.games = [resolve_game(g) for g in args.games[:2]]
        args.agent_mbti = args.agent_mbti[:1]
        args.opponent_mbti = args.opponent_mbti[:1]
        args.episodes = min(args.episodes, 2)

    games = [resolve_game(g) for g in args.games]
    eval_dir = new_eval_dir(args.run_id)
    write_latest_pointer("eval", eval_dir)

    results: List[Dict[str, Any]] = []
    for g in games:
        for a in args.agent_mbti:
            for o in args.opponent_mbti:
                results.append(
                    run_bot_matchup(
                        g,
                        agent_mbti=a,
                        opponent_mbti=o,
                        episodes=args.episodes,
                        seed=args.seed,
                    )
                )

    lora_metrics = None
    adapter = Path(args.adapter) if args.adapter else read_latest_path("lora")
    if adapter and adapter.is_dir():
        base = args.base_model or os_environ_model()
        info_path = adapter / "run_info.json"
        if info_path.is_file() and not args.base_model:
            base = json.loads(info_path.read_text()).get("base_model", base)
        lora_metrics = try_lora_legal_action_rate(
            adapter,
            base,
            games[0],
            args.agent_mbti[0],
            n=2 if args.smoke else 5,
        )

    payload = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "run_id": eval_dir.parent.name,
        "bot_matchups": results,
        "lora_check": lora_metrics,
        "types_available": list_types(),
    }
    (eval_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = ["# MBTI eval", ""]
    for r in results:
        lines.append(
            f"- {r['game']} | {r['agent_mbti']} vs {r['opponent_mbti']}: "
            f"mean_return={r['mean_return']:.3f} wins={r['wins']}/{r['episodes']}"
        )
    if lora_metrics and "format_ok" in lora_metrics:
        lines.append(
            f"\nLoRA format check: {lora_metrics['format_ok']}/{lora_metrics['n']} "
            f"on {lora_metrics['game']}"
        )
    (eval_dir / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Eval dir: {eval_dir}")
    print(f"Run dir: {eval_dir.parent}")


def os_environ_model() -> str:
    import os

    scratch = os.environ.get("SCRATCH_ROOT", str(ROOT.parent))
    for name in ("Qwen3-0.6B", "Qwen3-4B"):
        p = Path(scratch) / "models" / name
        if (p / "config.json").is_file():
            return str(p)
    return str(Path(scratch) / "models" / "Qwen3-0.6B")


if __name__ == "__main__":
    main()
