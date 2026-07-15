"""Heuristic action selection biased by MBTI style."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .mbti_types import MBTIType, get_mbti


@dataclass(frozen=True)
class BotStyle:
    risk: float
    info_focus: float
    aggression: float


def style_for_mbti(mbti: MBTIType | str) -> BotStyle:
    if isinstance(mbti, str):
        mbti = get_mbti(mbti)
    return BotStyle(risk=mbti.risk, info_focus=mbti.info_focus, aggression=mbti.aggression)


def choose_action(
    legal_actions: Sequence[int],
    *,
    mbti: MBTIType | str | None = None,
    rng: random.Random | None = None,
    action_scores: Dict[int, float] | None = None,
) -> int:
    """Pick among legal actions.

    If ``action_scores`` is provided (higher=better for a baseline policy),
    MBTI style reweights: high aggression/risk upweights exploration of lower
    baseline ranks; high info_focus stays closer to the top baseline action.
    Without scores, samples uniformly (with slight preference index bias).
    """
    if not legal_actions:
        raise ValueError("no legal actions")
    rng = rng or random.Random()
    actions = list(legal_actions)

    if mbti is None:
        return rng.choice(actions)

    style = style_for_mbti(mbti)
    if action_scores is None:
        # Prefer earlier (often lower-index) actions when cautious.
        weights = []
        for i, _a in enumerate(actions):
            # low aggression → prefer early/safe indices
            w = 1.0 + (1.0 - style.aggression) * (len(actions) - i) + style.risk * i
            weights.append(max(1e-6, w))
        return rng.choices(actions, weights=weights, k=1)[0]

    ranked = sorted(actions, key=lambda a: action_scores.get(a, 0.0), reverse=True)
    # Temperature from risk; stickiness from info_focus
    temp = 0.15 + 1.35 * style.risk
    stick = 0.2 + 0.8 * style.info_focus
    weights: List[float] = []
    for rank, a in enumerate(ranked):
        base = action_scores.get(a, 0.0)
        # Aggression boosts mid/low ranks a bit
        rank_boost = style.aggression * (rank / max(1, len(ranked) - 1))
        score = (base + rank_boost) / temp
        # Convert to positive weight via soft rank prior
        w = (stick / (1 + rank)) * (2.718 ** min(20.0, score))
        weights.append(max(1e-6, w))
    return rng.choices(ranked, weights=weights, k=1)[0]
