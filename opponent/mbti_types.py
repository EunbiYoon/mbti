"""16 MBTI personality types used as opponent / agent identities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class MBTIType:
    code: str
    name: str
    ei: str  # Extraversion / Introversion
    sn: str  # Sensing / Intuition
    tf: str  # Thinking / Feeling
    jp: str  # Judging / Perceiving
    summary: str
    # Game-decision style (used in prompts + SFT labels)
    play_style: str
    # Soft bias for heuristic bots: risk, info_focus, aggression in [0, 1]
    risk: float
    info_focus: float
    aggression: float


def _t(
    code: str,
    name: str,
    ei: str,
    sn: str,
    tf: str,
    jp: str,
    summary: str,
    play_style: str,
    risk: float,
    info_focus: float,
    aggression: float,
) -> MBTIType:
    return MBTIType(
        code=code,
        name=name,
        ei=ei,
        sn=sn,
        tf=tf,
        jp=jp,
        summary=summary,
        play_style=play_style,
        risk=risk,
        info_focus=info_focus,
        aggression=aggression,
    )


MBTI_TYPES: Dict[str, MBTIType] = {
    "ISTJ": _t(
        "ISTJ", "The Logistician", "I", "S", "T", "J",
        "Quiet, practical, and rule-oriented; values reliability and procedure.",
        "Prefer safe, textbook lines; punish opponent mistakes; minimize variance.",
        0.2, 0.6, 0.35,
    ),
    "ISFJ": _t(
        "ISFJ", "The Defender", "I", "S", "F", "J",
        "Protective and dutiful; prioritizes group welfare and careful support.",
        "In coop games, support teammates; in competitive games, conservative defense.",
        0.15, 0.55, 0.25,
    ),
    "INFJ": _t(
        "INFJ", "The Advocate", "I", "N", "F", "J",
        "Insightful planner who reads motives and pursues a coherent long-term path.",
        "Look ahead; infer hidden information; commit to a principled plan.",
        0.35, 0.85, 0.4,
    ),
    "INTJ": _t(
        "INTJ", "The Architect", "I", "N", "T", "J",
        "Strategic and independent; optimizes plans with long-horizon logic.",
        "Maximize EV with multi-step plans; exploit structural weaknesses.",
        0.45, 0.8, 0.55,
    ),
    "ISTP": _t(
        "ISTP", "The Virtuoso", "I", "S", "T", "P",
        "Flexible problem-solver who adapts tactically under pressure.",
        "React to the board; probe lightly; keep options open.",
        0.5, 0.45, 0.5,
    ),
    "ISFP": _t(
        "ISFP", "The Adventurer", "I", "S", "F", "P",
        "Gentle and spontaneous; avoids unnecessary conflict.",
        "Avoid escalation; choose harmonious / low-conflict moves.",
        0.3, 0.4, 0.2,
    ),
    "INFP": _t(
        "INFP", "The Mediator", "I", "N", "F", "P",
        "Idealistic and value-driven; seeks fair, meaningful outcomes.",
        "Favor fair play and cooperative signals over ruthless exploits.",
        0.35, 0.7, 0.3,
    ),
    "INTP": _t(
        "INTP", "The Logician", "I", "N", "T", "P",
        "Analytical and curious; enjoys novel hypotheses about the game tree.",
        "Analyze all branches; occasionally pick unconventional but coherent lines.",
        0.55, 0.9, 0.45,
    ),
    "ESTP": _t(
        "ESTP", "The Entrepreneur", "E", "S", "T", "P",
        "Bold, opportunistic, and action-oriented.",
        "Seize immediate tactical edges; bluff/press when momentum favors you.",
        0.75, 0.35, 0.8,
    ),
    "ESFP": _t(
        "ESFP", "The Entertainer", "E", "S", "F", "P",
        "Energetic and expressive; prefers engaging, high-visibility moves.",
        "Prefer flashy / interactive lines; keep the game lively.",
        0.65, 0.3, 0.6,
    ),
    "ENFP": _t(
        "ENFP", "The Campaigner", "E", "N", "F", "P",
        "Enthusiastic explorer who invents creative approaches.",
        "Try creative gambits; share information freely when it builds rapport.",
        0.7, 0.75, 0.55,
    ),
    "ENTP": _t(
        "ENTP", "The Debater", "E", "N", "T", "P",
        "Inventive and provocative; likes to test assumptions and force errors.",
        "Mix unpredictability with sharp theory; bait opponents into mistakes.",
        0.8, 0.85, 0.7,
    ),
    "ESTJ": _t(
        "ESTJ", "The Executive", "E", "S", "T", "J",
        "Organized leader who enforces clear procedures and pressure.",
        "Dictate tempo; apply consistent pressure; punish deviations.",
        0.5, 0.5, 0.75,
    ),
    "ESFJ": _t(
        "ESFJ", "The Consul", "E", "S", "F", "J",
        "Sociable coordinator who keeps everyone coordinated and comfortable.",
        "Coordinate explicitly; give clear supportive hints in coop settings.",
        0.4, 0.5, 0.45,
    ),
    "ENFJ": _t(
        "ENFJ", "The Protagonist", "E", "N", "F", "J",
        "Charismatic guide who rallies others toward a shared goal.",
        "Lead with informative, morale-building choices; prioritize team success.",
        0.45, 0.7, 0.5,
    ),
    "ENTJ": _t(
        "ENTJ", "The Commander", "E", "N", "T", "J",
        "Decisive commander focused on victory and efficient control.",
        "Play to win aggressively with structured plans; seize initiative.",
        0.6, 0.75, 0.85,
    ),
}


def get_mbti(code: str) -> MBTIType:
    key = code.strip().upper()
    if key not in MBTI_TYPES:
        raise ValueError(f"unknown MBTI type {code!r}; valid: {', '.join(sorted(MBTI_TYPES))}")
    return MBTI_TYPES[key]


def list_types() -> List[str]:
    return sorted(MBTI_TYPES.keys())


def dichotomies() -> Tuple[Tuple[str, str], ...]:
    return (("E", "I"), ("S", "N"), ("T", "F"), ("J", "P"))
