#!/usr/bin/env python3
"""LoRA SFT training; outputs under runs/<timestamp>/lora/."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_paths import new_lora_dir, write_latest_pointer, read_latest_path
from lora.helpers import build_lora_config


def default_model() -> str:
    scratch = os.environ.get(
        "SCRATCH_ROOT",
        str(ROOT.parent),
    )
    for name in ("Qwen3-0.6B", "Qwen3-4B"):
        p = Path(scratch) / "models" / name
        if (p / "config.json").is_file():
            return str(p)
    return str(Path(scratch) / "models" / "Qwen3-0.6B")


def format_example(ex: dict) -> str:
    return ex["prompt"] + "\n\nANSWER:\n" + ex["completion"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", default=None)
    ap.add_argument("--base_model", default=None)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # if user passed .../runs/ts/lora keep; else nest under new run
    else:
        out_dir = new_lora_dir(args.run_id)
    write_latest_pointer("lora", out_dir)

    if args.train_file is None:
        # Prefer train.jsonl in this out_dir, else latest lora dir
        candidate = out_dir / "train.jsonl"
        if not candidate.is_file():
            latest = read_latest_path("lora")
            if latest and (latest / "train.jsonl").is_file():
                candidate = latest / "train.jsonl"
        args.train_file = str(candidate)

    train_path = Path(args.train_file)
    if not train_path.is_file():
        raise SystemExit(
            f"Train file not found: {train_path}\n"
            "Generate first: python lora/generate_data.py --smoke"
        )

    base_model = args.base_model or os.environ.get("PRETRAIN") or default_model()
    if args.smoke:
        args.epochs = min(args.epochs, 1.0)
        args.max_steps = 2 if args.max_steps < 0 else args.max_steps
        args.max_seq_len = min(args.max_seq_len, 512)

    import torch
    from datasets import load_dataset
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = get_peft_model(model, build_lora_config())
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=str(train_path), split="train")

    sft_args = dict(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=1,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        report_to=[],
        max_seq_length=args.max_seq_len,
        packing=False,
    )
    # TRL version differences: max_seq_length vs max_length
    try:
        config = SFTConfig(**sft_args)
    except TypeError:
        sft_args.pop("max_seq_length", None)
        sft_args["max_seq_length"] = args.max_seq_len
        try:
            config = SFTConfig(**sft_args)
        except TypeError:
            sft_args.pop("max_seq_length", None)
            config = SFTConfig(**{k: v for k, v in sft_args.items() if k != "packing"})

    if args.max_steps > 0:
        config.max_steps = args.max_steps

    try:
        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=ds,
            processing_class=tokenizer,
            formatting_func=format_example,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=ds,
            tokenizer=tokenizer,
            formatting_func=format_example,
        )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    info = {
        "base_model": base_model,
        "train_file": str(train_path),
        "out_dir": str(out_dir),
        "run_id": out_dir.parent.name,
        "cuda": torch.cuda.is_available(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "smoke": bool(args.smoke),
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"LoRA adapter saved -> {out_dir}")
    print(f"Run dir: {out_dir.parent}")


if __name__ == "__main__":
    main()
