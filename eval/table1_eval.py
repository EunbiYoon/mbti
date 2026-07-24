#!/usr/bin/env python3
"""Table 1-style downstream reasoning eval (Single / MAD / AutoGen).

Writes results under <RUN_DIR>/table1/ as JSON + markdown matching table1.md columns.
Adapter/base-model resolution follows the mbti run layout: runs/<timestamp>/lora/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BENCHMARKS = ("MATH", "GSM8K", "AQUA", "AIME", "AMC", "MMLU", "GPQA")
SETTINGS = ("Single Agent", "MAD (Competitive)", "AutoGen (Cooperative)")


@dataclass
class Example:
    question: str
    answer: str
    choices: list[str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=int(os.environ.get("TABLE1_LIMIT", "0")))
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model-name", default="MARSHAL")
    return parser.parse_args()


def resolve_adapter(run_dir: Path) -> Path:
    """Find a LoRA adapter under runs/<ts>/lora (or run_dir itself).

    Prefers the top-level adapter, then the highest checkpoint-N.
    """
    candidates = [run_dir / "lora", run_dir]
    for root in candidates:
        if (root / "adapter_config.json").is_file():
            return root
        ckpts = sorted(
            root.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
        )
        for ckpt in reversed(ckpts):
            if (ckpt / "adapter_config.json").is_file():
                return ckpt
    raise FileNotFoundError(f"no LoRA adapter under {run_dir}")


def resolve_base_model(run_dir: Path, adapter: Path, explicit: Path | None) -> str:
    if explicit is not None:
        return str(explicit)
    # Prefer the base recorded with this adapter (smoke=0.6B, full=4B).
    for info_path in (run_dir / "lora" / "run_info.json", adapter / "run_info.json"):
        if info_path.is_file():
            try:
                base = json.loads(info_path.read_text(encoding="utf-8")).get("base_model")
                if base:
                    return str(base)
            except Exception:
                pass
    cfg = adapter / "adapter_config.json"
    if cfg.is_file():
        try:
            base = json.loads(cfg.read_text(encoding="utf-8")).get("base_model_name_or_path")
            if base:
                return str(base)
        except Exception:
            pass
    return os.environ.get("PRETRAIN_FULL") or os.environ.get("PRETRAIN") or str(
        Path(os.environ.get("MODELS_ROOT", str(ROOT.parent / "models"))) / "Qwen3-4B"
    )


def load_model(base_model: str, adapter: Path | None, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    ).to(device)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
        model.to(device)
    model.eval()
    return model, tokenizer


def chat_generate(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    return text.strip()


def extract_final_answer(text: str, choices: list[str] | None = None) -> str:
    patterns = [
        r"(?i)final answer\s*[:\-]\s*(.+)",
        r"(?i)answer\s*[:\-]\s*(.+)",
        r"\\boxed\{([^}]+)\}",
        r"<answer>\s*(.*?)\s*</answer>",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            candidate = match.group(1).strip().splitlines()[0].strip()
            return normalize_answer(candidate, choices)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return normalize_answer(lines[-1], choices)


def normalize_answer(answer: str, choices: list[str] | None = None) -> str:
    answer = answer.strip()
    answer = re.sub(r"^[\s\$\\]+|[\s\.\,\;]+$", "", answer)
    if choices:
        letter = re.match(r"^([A-Ea-e])\b", answer)
        if letter:
            return letter.group(1).upper()
        for idx, choice in enumerate(choices):
            if answer.lower() == choice.lower():
                return chr(ord("A") + idx)
    answer = answer.replace(",", "")
    boxed = re.search(r"([-+]?\d+(?:\.\d+)?(?:/\d+)?)", answer)
    if boxed and not choices:
        return boxed.group(1)
    return answer


def answers_match(pred: str, gold: str, choices: list[str] | None = None) -> bool:
    pred_n = normalize_answer(pred, choices)
    gold_n = normalize_answer(gold, choices)
    if not pred_n or not gold_n:
        return False
    if pred_n.lower() == gold_n.lower():
        return True
    try:
        return abs(float(pred_n) - float(gold_n)) < 1e-6
    except ValueError:
        return False


def _take(examples: list[Example], limit: int) -> list[Example]:
    if limit and limit > 0:
        return examples[:limit]
    return examples


def load_gsm8k(limit: int) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    out: list[Example] = []
    for row in ds:
        gold = str(row["answer"]).split("####")[-1].strip()
        out.append(Example(question=row["question"], answer=gold))
    return _take(out, limit)


def load_math(limit: int) -> list[Example]:
    from datasets import load_dataset

    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        out = [Example(question=row["problem"], answer=str(row["answer"])) for row in ds]
    except Exception:
        ds = load_dataset("lighteval/MATH", "all", split="test")
        out = [
            Example(question=row["problem"], answer=str(row["solution"]).split("####")[-1].strip())
            for row in ds
        ]
    return _take(out, limit)


def load_aqua(limit: int) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("aqua_rat", split="test")
    out: list[Example] = []
    for row in ds:
        choices = [c.split(")", 1)[-1].strip() if ")" in c else c for c in row["options"]]
        out.append(Example(question=row["question"], answer=str(row["correct"]).upper(), choices=choices))
    return _take(out, limit)


def load_aime(limit: int) -> list[Example]:
    from datasets import load_dataset

    for name in ("math-ai/aime24", "HuggingFaceH4/aime_2024"):
        try:
            ds = load_dataset(name, split="train")
            out = [
                Example(
                    question=row.get("problem") or row.get("question"),
                    answer=str(row.get("answer") or row.get("final_answer")),
                )
                for row in ds
            ]
            return _take(out, limit)
        except Exception:
            continue
    return []


def load_amc(limit: int) -> list[Example]:
    from datasets import load_dataset

    for name in ("math-ai/amc23", "AI-MO/aimo-validation-amc"):
        try:
            ds = load_dataset(name, split="train")
            out = []
            for row in ds:
                q = row.get("problem") or row.get("question")
                a = row.get("answer") or row.get("final_answer")
                if q is None or a is None:
                    continue
                out.append(Example(question=str(q), answer=str(a)))
            if out:
                return _take(out, limit)
        except Exception:
            continue
    return []


def load_mmlu(limit: int) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    out: list[Example] = []
    for row in ds:
        choices = list(row["choices"])
        gold = chr(ord("A") + int(row["answer"]))
        stem = "\n".join(f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices))
        out.append(Example(question=f"{row['question']}\n{stem}", answer=gold, choices=choices))
    return _take(out, limit)


def load_gpqa(limit: int) -> list[Example]:
    from datasets import load_dataset
    import random as _random

    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception:
        return []
    rng = _random.Random(0)
    out: list[Example] = []
    for row in ds:
        options = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        order = list(range(4))
        rng.shuffle(order)
        labeled = [options[i] for i in order]
        gold_idx = order.index(0)
        gold = chr(ord("A") + gold_idx)
        stem = "\n".join(f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(labeled))
        out.append(Example(question=f"{row['Question']}\n{stem}", answer=gold, choices=labeled))
    return _take(out, limit)


LOADERS: dict[str, Callable[[int], list[Example]]] = {
    "MATH": load_math,
    "GSM8K": load_gsm8k,
    "AQUA": load_aqua,
    "AIME": load_aime,
    "AMC": load_amc,
    "MMLU": load_mmlu,
    "GPQA": load_gpqa,
}


def single_agent_prompt(ex: Example) -> list[dict[str, str]]:
    extra = ""
    if ex.choices:
        extra = " Reply with the choice letter only in the final answer."
    return [
        {
            "role": "system",
            "content": "You are a careful reasoning assistant. Solve the problem step by step.",
        },
        {
            "role": "user",
            "content": f"{ex.question}\n\nPut the final answer after 'Final answer:'.{extra}",
        },
    ]


def mad_prompt(ex: Example) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are coordinating a Multi-Agent Debate (MAD). "
                "Simulate Affirmative and Negative agents arguing, then decide as Judge."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Problem:\n{ex.question}\n\n"
                "Format:\n"
                "Affirmative: ...\nNegative: ...\nJudge: ...\n"
                "Final answer: <answer>"
            ),
        },
    ]


def autogen_prompt(ex: Example) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are coordinating a cooperative AutoGen team "
                "(Solver + Critic). Collaborate to find the correct answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Problem:\n{ex.question}\n\n"
                "Format:\n"
                "Solver: ...\nCritic: ...\nSolver (revised): ...\n"
                "Final answer: <answer>"
            ),
        },
    ]


SETTING_PROMPTS = {
    "Single Agent": single_agent_prompt,
    "MAD (Competitive)": mad_prompt,
    "AutoGen (Cooperative)": autogen_prompt,
}


def evaluate_setting(
    model,
    tokenizer,
    setting: str,
    examples_by_bench: dict[str, list[Example]],
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    prompt_fn = SETTING_PROMPTS[setting]
    scores: dict[str, float] = {}
    details: dict[str, list[dict[str, Any]]] = {}
    for bench, examples in examples_by_bench.items():
        if not examples:
            scores[bench] = float("nan")
            details[bench] = []
            continue
        correct = 0
        rows: list[dict[str, Any]] = []
        for idx, ex in enumerate(examples):
            messages = prompt_fn(ex)
            text = chat_generate(model, tokenizer, messages, max_new_tokens, temperature)
            pred = extract_final_answer(text, ex.choices)
            ok = answers_match(pred, ex.answer, ex.choices)
            correct += int(ok)
            rows.append(
                {
                    "index": idx,
                    "pred": pred,
                    "gold": ex.answer,
                    "correct": ok,
                    "response": text[:2000],
                }
            )
            print(
                f"[{setting}] {bench} {idx+1}/{len(examples)} correct={ok} "
                f"pred={pred!r} gold={ex.answer!r}",
                flush=True,
            )
        scores[bench] = 100.0 * correct / max(1, len(examples))
        details[bench] = rows
    vals = [v for v in scores.values() if v == v]
    average = sum(vals) / max(1, len(vals)) if vals else float("nan")
    return {"setting": setting, "average": average, "scores": scores, "details": details}


def write_markdown(
    path: Path, model_name: str, results: list[dict[str, Any]], paper_ref: Path | None
) -> None:
    lines = [
        "# Table 1: Evaluation results on downstream reasoning benchmarks within multi-agent systems",
        "",
        f"Model: **{model_name}** (this run)",
        "",
        "| Setting | Model | Average | MATH | GSM8K | AQUA | AIME | AMC | MMLU | GPQA |",
        "|---------|-------|-------:|------:|------:|------:|------:|------:|------:|------:|",
    ]
    for result in results:
        scores = result["scores"]
        cells = [
            f"{scores.get(b, float('nan')):.2f}" if scores.get(b) == scores.get(b) else "—"
            for b in BENCHMARKS
        ]
        avg = result["average"]
        avg_s = f"{avg:.2f}" if avg == avg else "—"
        lines.append(f"| {result['setting']} | {model_name} | {avg_s} | " + " | ".join(cells) + " |")
    lines.append("")
    if paper_ref and paper_ref.is_file():
        lines.append("## Paper Table 1 reference")
        lines.append("")
        lines.append(paper_ref.read_text(encoding="utf-8"))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or (run_dir / "table1")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = args.adapter.resolve() if args.adapter else resolve_adapter(run_dir)
    base_model = resolve_base_model(run_dir, adapter, args.base_model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"RUN_DIR={run_dir}", flush=True)
    print(f"BASE={base_model}", flush=True)
    print(f"ADAPTER={adapter}", flush=True)
    print(f"OUT={out_dir}", flush=True)
    print(f"LIMIT={args.limit}", flush=True)

    model, tokenizer = load_model(base_model, adapter, device)

    examples_by_bench: dict[str, list[Example]] = {}
    for bench, loader in LOADERS.items():
        try:
            examples_by_bench[bench] = loader(args.limit)
            print(f"loaded {bench}: {len(examples_by_bench[bench])} examples", flush=True)
        except Exception as exc:
            print(f"WARN: failed to load {bench}: {exc}", flush=True)
            examples_by_bench[bench] = []

    started = time.time()
    results = []
    for setting in SETTINGS:
        result = evaluate_setting(
            model,
            tokenizer,
            setting,
            examples_by_bench,
            args.max_new_tokens,
            args.temperature,
        )
        results.append(result)
        slim = {k: v for k, v in result.items() if k != "details"}
        slim["n"] = {b: len(result["details"][b]) for b in BENCHMARKS}
        (out_dir / f"{setting.split()[0].lower()}_details.json").write_text(
            json.dumps(slim, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (out_dir / f"{setting.split()[0].lower()}_predictions.jsonl").write_text(
            "\n".join(
                json.dumps({"benchmark": b, **row}, sort_keys=True)
                for b, rows in result["details"].items()
                for row in rows
            ),
            encoding="utf-8",
        )

    summary = {
        "run_dir": str(run_dir),
        "adapter": str(adapter),
        "base_model": str(base_model),
        "model_name": args.model_name,
        "limit": args.limit,
        "elapsed_sec": time.time() - started,
        "results": [
            {
                "setting": r["setting"],
                "average": r["average"],
                "scores": r["scores"],
                "n": {b: len(r["details"][b]) for b in BENCHMARKS},
            }
            for r in results
        ],
    }
    (out_dir / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    paper_ref = ROOT / "table1.md"
    write_markdown(out_dir / "table1.md", args.model_name, results, paper_ref if paper_ref.is_file() else None)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"=== wrote {out_dir}/table1.md ===", flush=True)


if __name__ == "__main__":
    main()
