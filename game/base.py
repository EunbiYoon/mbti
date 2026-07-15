"""Shared helpers for OpenSpiel game wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StepResult:
    observation: str
    rewards: List[float]
    done: bool
    info: Dict[str, Any]
    legal_actions: Dict[int, str]
    current_player: int


def format_answer(action_str: str) -> str:
    return f"<answer>{action_str}</answer>"


def parse_answer(text: str) -> Optional[str]:
    import re

    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()
