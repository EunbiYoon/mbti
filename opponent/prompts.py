"""Personality prompt wrappers for LLM agents."""

from __future__ import annotations

from .mbti_types import MBTIType, get_mbti


def personality_system_addon(mbti: MBTIType | str) -> str:
    if isinstance(mbti, str):
        mbti = get_mbti(mbti)
    return (
        f"You are an agent with MBTI type {mbti.code} ({mbti.name}). "
        f"{mbti.summary} "
        f"Decision style: {mbti.play_style} "
        f"(risk≈{mbti.risk:.2f}, info_focus≈{mbti.info_focus:.2f}, aggression≈{mbti.aggression:.2f}). "
        "Stay in character while still outputting a single legal action in the required format."
    )


def wrap_system_prompt(base_system: str, mbti: MBTIType | str | None) -> str:
    if mbti is None:
        return base_system
    return f"{personality_system_addon(mbti)}\n\n{base_system}"


def sft_rationale_prefix(mbti: MBTIType | str) -> str:
    if isinstance(mbti, str):
        mbti = get_mbti(mbti)
    return (
        f"[{mbti.code}] Thinking as {mbti.name}: {mbti.play_style} "
        f"I choose the following action."
    )
