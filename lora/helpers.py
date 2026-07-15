"""LoRA helpers (PEFT defaults aligned with marshal/maf)."""

from __future__ import annotations

import os
from typing import Iterable, Optional

DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def lora_settings_from_env() -> dict:
    return {
        "rank": int(os.environ.get("LORA_RANK", "16")),
        "alpha": int(os.environ.get("LORA_ALPHA", "32")),
        "dropout": float(os.environ.get("LORA_DROPOUT", "0.05")),
        "target_modules": os.environ.get("LORA_TARGET_MODULES", "").split(",")
        if os.environ.get("LORA_TARGET_MODULES")
        else list(DEFAULT_LORA_TARGET_MODULES),
    }


def build_lora_config(
    *,
    rank: Optional[int] = None,
    alpha: Optional[int] = None,
    dropout: Optional[float] = None,
    target_modules: Optional[Iterable[str]] = None,
):
    from peft import LoraConfig, TaskType

    s = lora_settings_from_env()
    return LoraConfig(
        r=rank if rank is not None else s["rank"],
        lora_alpha=alpha if alpha is not None else s["alpha"],
        lora_dropout=dropout if dropout is not None else s["dropout"],
        target_modules=list(target_modules) if target_modules is not None else s["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
