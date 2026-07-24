#!/usr/bin/env python3
"""Evaluate MBTI bots and/or a LoRA adapter; writes runs/<timestamp>/eval/.

Presets
-------
- --smoke: 2 episodes, 1 MBTI pair (quick wiring check)
- full (default): 20 episodes, save episodes.jsonl

Always runs bot matchups + LoRA matchups (unless --bot-only / --lora-only).

Outputs
-------
runs/<run_id>/eval/
  metrics.json
  result.md
  episodes.jsonl   (full by default; optional with --save_episodes on smoke)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game import list_games, make_env, resolve_game
from game.base import parse_answer
from opponent import list_types
from run_paths import (
    new_eval_dir,
    read_latest_path,
    resolve_eval_run_id,
    write_latest_pointer,
)


def os_environ_model() -> str:
    import os

    scratch = os.environ.get("SCRATCH_ROOT", str(ROOT.parent))
    pretrain = os.environ.get("PRETRAIN")
    if pretrain:
        return pretrain
    for name in ("Qwen3-4B", "Qwen3-0.6B"):
        p = Path(scratch) / "models" / name
        if (p / "config.json").is_file():
            return str(p)
    return str(Path(scratch) / "models" / "Qwen3-4B")


def run_bot_matchup(
    game: str,
    *,
    agent_mbti: str,
    opponent_mbti: str,
    episodes: int,
    seed: int,
) -> Dict[str, Any]:
    returns_agent: List[float] = []
    wins = draws = losses = 0
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
                losses += 1
        else:
            if r0 > 0:
                wins += 1
            elif r0 == 0:
                draws += 1
            else:
                losses += 1
    return {
        "mode": "bot",
        "game": game,
        "agent_mbti": agent_mbti,
        "opponent_mbti": opponent_mbti,
        "episodes": episodes,
        "mean_return": sum(returns_agent) / max(1, len(returns_agent)),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "returns": returns_agent,
    }


def _match_legal(parsed: Optional[str], legal: Dict[int, str]) -> Optional[int]:
    if not parsed:
        return None
    p = parsed.strip()
    for a, s in legal.items():
        if s == p or s.strip("<>") == p.strip("<>") or p == s.strip("<>"):
            return a
    # substring fallback
    for a, s in legal.items():
        if p in s or s in p:
            return a
    return None


class LoraAgent:
    def __init__(self, adapter_dir: Path, base_model: str, *, max_new_tokens: int = 96):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.adapter_corrupt = False

        # Full train run 20260715_125114 had NaN grads → all LoRA weights NaN.
        # Check BEFORE loading onto GPU so we can skip a poisoned adapter.
        try:
            from safetensors.torch import load_file

            sd = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"), device="cpu")
            n_bad = sum(1 for v in sd.values() if not torch.isfinite(v).all())
            if n_bad:
                self.adapter_corrupt = True
                print(
                    f"WARNING: adapter has {n_bad}/{len(sd)} non-finite tensors; "
                    "using base model only for generation",
                    flush=True,
                )
        except Exception as e:
            print(f"WARNING: could not preflight adapter weights: {e}", flush=True)

        self.tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # Keep the whole 4B model on one GPU. device_map="auto" splits across
        # visible GPUs and makes token-by-token generate pathologically slow.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        print(f"Loading base model: {base_model} -> {self.device}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        if self.adapter_corrupt:
            self.model = model
            print("Skipping PeftModel load (corrupt adapter)", flush=True)
        else:
            print(f"Attaching LoRA from {adapter_dir}", flush=True)
            self.model = PeftModel.from_pretrained(model, str(adapter_dir))
            self.model.to(self.device)
        self.model.eval()
        gc = getattr(self.model, "generation_config", None)
        if gc is not None:
            gc.do_sample = False
            gc.temperature = 1.0
            gc.top_p = 1.0
            gc.top_k = 50
        print(f"LoraAgent ready on {self.device} corrupt={self.adapter_corrupt}", flush=True)

    def act(self, env, *, player_id: int, mbti: str, step) -> Tuple[int, str, str]:
        prompt = env.get_prompt(player_id=player_id, mbti=mbti)
        legal = step.legal_actions
        legal_strs = ", ".join(legal.values())
        user = (
            f"{prompt['user']}\nCURRENT STATE:\n{step.observation}\n"
            f"LEGAL ACTIONS: {legal_strs}\n"
            "Reply with <answer>{one legal action}</answer> only.\n"
        )
        text_in = f"System: {prompt['system']}\n\nUser: {user}\n\nANSWER:\n"
        inputs = self.tok(
            text_in,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = self.torch.ones_like(inputs["input_ids"])

        gen = ""
        # Greedy decode avoids multinomial CUDA assert on degenerate probs.
        try:
            with self.torch.no_grad():
                out = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tok.eos_token_id,
                    eos_token_id=self.tok.eos_token_id,
                )
            gen = self.tok.decode(
                out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
        except RuntimeError as e:
            gen = f"[generate_error: {e}]"
            action = random.choice(list(legal.keys()))
            return action, gen, "fallback"

        parsed = parse_answer(gen)
        action = _match_legal(parsed, legal)
        format_ok = action is not None
        if action is None:
            action = random.choice(list(legal.keys()))
        return action, gen, ("ok" if format_ok else "fallback")


def run_lora_matchup(
    game: str,
    agent: LoraAgent,
    *,
    agent_mbti: str,
    opponent_mbti: str,
    episodes: int,
    seed: int,
    agent_player: int = 0,
    max_steps: int = 50,
    save_episodes: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    from opponent.policies import choose_action

    returns_agent: List[float] = []
    wins = draws = losses = 0
    format_ok = format_total = 0
    episode_logs: List[Dict[str, Any]] = []

    for i in range(episodes):
        env = make_env(game, seed=seed + i)
        step = env.reset(seed + i)
        history: List[Dict[str, Any]] = []
        for _ in range(max_steps):
            if step.done:
                break
            pid = step.current_player
            if pid == agent_player:
                action, gen, status = agent.act(env, player_id=pid, mbti=agent_mbti, step=step)
                format_total += 1
                if status == "ok":
                    format_ok += 1
                history.append(
                    {
                        "player": pid,
                        "action": action,
                        "action_str": step.legal_actions[action],
                        "status": status,
                        "gen": gen[:500],
                    }
                )
                step = env.step(action)
            else:
                legal = list(step.legal_actions.keys())
                scores = None
                if game == "kuhn_poker":
                    scores = {a: (1.0 if a == 1 else 0.2) for a in legal}
                elif game == "minihanabi":
                    scores = {}
                    for a, s in step.legal_actions.items():
                        sl = s.lower()
                        sc = 0.3
                        if any(k in sl for k in ("reveal", "hint", "color", "rank")):
                            sc += 1.0
                        if "play" in sl:
                            sc += 0.6
                        if "discard" in sl:
                            sc += 0.2
                        scores[a] = sc
                action = choose_action(
                    legal, mbti=opponent_mbti, rng=env._rng, action_scores=scores
                )
                history.append(
                    {
                        "player": pid,
                        "action": action,
                        "action_str": step.legal_actions[action],
                        "mbti": opponent_mbti,
                    }
                )
                step = env.step(action)

        r0 = float(step.rewards[0]) if step.rewards else 0.0
        returns_agent.append(r0)
        info = step.info or {}
        if env.zero_sum:
            w = info.get("winner", -1)
            if w == 0:
                wins += 1
            elif w == -1:
                draws += 1
            else:
                losses += 1
        else:
            if r0 > 0:
                wins += 1
            elif r0 == 0:
                draws += 1
            else:
                losses += 1
        if save_episodes:
            episode_logs.append(
                {
                    "game": game,
                    "seed": seed + i,
                    "agent_mbti": agent_mbti,
                    "opponent_mbti": opponent_mbti,
                    "rewards": step.rewards,
                    "info": info,
                    "history": history,
                }
            )

    summary = {
        "mode": "lora",
        "game": game,
        "agent_mbti": agent_mbti,
        "opponent_mbti": opponent_mbti,
        "episodes": episodes,
        "mean_return": sum(returns_agent) / max(1, len(returns_agent)),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "returns": returns_agent,
        "action_format_ok": format_ok,
        "action_format_total": format_total,
        "action_format_rate": (format_ok / format_total) if format_total else 0.0,
    }
    return summary, episode_logs


def try_lora_format_check(
    adapter_dir: Path,
    base_model: str,
    game: str,
    mbti: str,
    *,
    n: int = 3,
) -> Dict[str, Any]:
    if not (adapter_dir / "adapter_config.json").is_file():
        return {"error": f"no adapter at {adapter_dir}"}
    try:
        agent = LoraAgent(adapter_dir, base_model)
    except Exception as e:
        return {"error": f"load failed: {e}"}
    env = make_env(game)
    step = env.reset(0)
    ok = 0
    samples = []
    for _ in range(n):
        _, gen, status = agent.act(env, player_id=0, mbti=mbti, step=step)
        samples.append(gen)
        if status == "ok":
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
    ap = argparse.ArgumentParser(description="MBTI / LoRA evaluation")
    ap.add_argument("--games", nargs="+", default=list_games())
    ap.add_argument("--agent_mbti", nargs="+", default=["INTJ", "ENFP"])
    ap.add_argument("--opponent_mbti", nargs="+", default=["ISTJ", "ENTP"])
    ap.add_argument("--episodes", type=int, default=None, help="Default: 2 smoke / 20 full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_id", default=None, help="Share timestamp folder with a LoRA run")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (default: runs/latest_lora)")
    ap.add_argument("--base_model", default=None)
    ap.add_argument("--bot-only", action="store_true", help="Skip LoRA; MBTI bots only")
    ap.add_argument("--lora-only", action="store_true", help="Skip bot matchups; LoRA only")
    ap.add_argument("--save_episodes", action="store_true", help="Write episodes.jsonl (on by default for full)")
    ap.add_argument("--smoke", action="store_true", help="Quick wiring check (tiny episode count)")
    args = ap.parse_args()

    if args.smoke:
        # Keep caller-specified games; shrink MBTI × episodes for a quick check.
        args.agent_mbti = args.agent_mbti[:1]
        args.opponent_mbti = args.opponent_mbti[:1]
        args.episodes = 2 if args.episodes is None else min(args.episodes, 2)
    else:
        # Full eval defaults.
        if args.episodes is None:
            args.episodes = 20
        args.save_episodes = True

    games = [resolve_game(g) for g in args.games]

    # Resolve adapter first so eval nests under the same timestamp as lora/.
    adapter: Optional[Path] = None
    if not args.bot_only:
        adapter = Path(args.adapter) if args.adapter else read_latest_path("lora")
    elif args.adapter:
        adapter = Path(args.adapter)

    run_id = resolve_eval_run_id(args.run_id, adapter)
    eval_dir = new_eval_dir(run_id)  # also creates sibling runs/<id>/lora/
    write_latest_pointer("eval", eval_dir)
    print(f"Run layout: {eval_dir.parent}/{{lora,eval}}  (run_id={run_id})")

    bot_results: List[Dict[str, Any]] = []
    if not args.lora_only:
        for g in games:
            for a in args.agent_mbti:
                for o in args.opponent_mbti:
                    bot_results.append(
                        run_bot_matchup(
                            g,
                            agent_mbti=a,
                            opponent_mbti=o,
                            episodes=args.episodes,
                            seed=args.seed,
                        )
                    )

    lora_results: List[Dict[str, Any]] = []
    lora_check: Optional[Dict[str, Any]] = None
    episode_rows: List[Dict[str, Any]] = []
    if not args.bot_only:
        if adapter and (adapter / "adapter_config.json").is_file():
            base = args.base_model or os_environ_model()
            info_path = adapter / "run_info.json"
            if info_path.is_file() and not args.base_model:
                base = json.loads(info_path.read_text()).get("base_model", base)
            print(f"Loading LoRA adapter: {adapter}")
            agent = LoraAgent(adapter, base)
            # format smoke with the same loaded agent
            n_check = 2 if args.smoke else 5
            env0 = make_env(games[0])
            step0 = env0.reset(0)
            ok = 0
            samples = []
            for i in range(n_check):
                print(f"  format check {i + 1}/{n_check}...", flush=True)
                _, gen, status = agent.act(
                    env0, player_id=0, mbti=args.agent_mbti[0], step=step0
                )
                samples.append(gen)
                if status == "ok":
                    ok += 1
            lora_check = {
                "adapter": str(adapter),
                "adapter_corrupt": bool(getattr(agent, "adapter_corrupt", False)),
                "game": games[0],
                "mbti": args.agent_mbti[0],
                "format_ok": ok,
                "n": n_check,
                "samples": samples[:3],
            }
            print(f"  format check done: {ok}/{n_check}", flush=True)
            for g in games:
                for a in args.agent_mbti:
                    for o in args.opponent_mbti:
                        print(
                            f"LoRA matchup: {g} | {a} vs {o} "
                            f"({args.episodes} eps)...",
                            flush=True,
                        )
                        summary, eps = run_lora_matchup(
                            g,
                            agent,
                            agent_mbti=a,
                            opponent_mbti=o,
                            episodes=args.episodes,
                            seed=args.seed + 10_000,
                            save_episodes=args.save_episodes,
                            max_steps=20 if args.smoke else 50,
                        )
                        lora_results.append(summary)
                        episode_rows.extend(eps)
                        print(
                            f"  -> mean={summary['mean_return']:.3f} "
                            f"format={summary['action_format_rate']:.0%}",
                            flush=True,
                        )
        else:
            lora_check = {"error": f"adapter missing: {adapter}"}

    payload = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "run_id": eval_dir.parent.name,
        "adapter": str(adapter) if adapter else None,
        "bot_matchups": bot_results,
        "lora_matchups": lora_results,
        "lora_check": lora_check,
        "types_available": list_types(),
        "config": {
            "games": games,
            "agent_mbti": args.agent_mbti,
            "opponent_mbti": args.opponent_mbti,
            "episodes": args.episodes,
            "smoke": bool(args.smoke),
            "bot_only": bool(args.bot_only),
            "lora_only": bool(args.lora_only),
        },
    }
    (eval_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if episode_rows:
        with (eval_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
            for row in episode_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = ["# MBTI eval", "", "## Bot matchups (MBTI vs MBTI)", ""]
    if not bot_results:
        lines.append("_skipped_")
    for r in bot_results:
        lines.append(
            f"- {r['game']} | {r['agent_mbti']} vs {r['opponent_mbti']}: "
            f"mean_return={r['mean_return']:.3f} "
            f"W/D/L={r['wins']}/{r['draws']}/{r['losses']}"
        )
    lines += ["", "## LoRA matchups (adapter vs MBTI bot)", ""]
    if not lora_results:
        lines.append("_skipped or no adapter_")
    for r in lora_results:
        lines.append(
            f"- {r['game']} | LoRA({r['agent_mbti']}) vs {r['opponent_mbti']}: "
            f"mean_return={r['mean_return']:.3f} "
            f"W/D/L={r['wins']}/{r['draws']}/{r['losses']} "
            f"format={r['action_format_ok']}/{r['action_format_total']} "
            f"({r['action_format_rate']:.0%})"
        )
    if lora_check and "format_ok" in lora_check:
        lines.append(
            f"\nLoRA format smoke: {lora_check['format_ok']}/{lora_check['n']} "
            f"on {lora_check.get('game')}"
        )
    elif lora_check and "error" in lora_check:
        lines.append(f"\nLoRA note: {lora_check['error']}")

    (eval_dir / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nEval dir: {eval_dir}")
    print(f"Run dir:  {eval_dir.parent}")


if __name__ == "__main__":
    main()
