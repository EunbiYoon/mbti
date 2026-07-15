"""Kuhn Poker via OpenSpiel."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from opponent.mbti_types import MBTIType
from opponent.policies import choose_action
from opponent.prompts import wrap_system_prompt

from .base import StepResult

# OpenSpiel Kuhn: 0=Pass/Check/Fold, 1=Bet/Call
_ACTION_NAMES = {0: "pass", 1: "bet"}
_CARD_NAMES = {0: "Jack", 1: "Queen", 2: "King"}


class KuhnPokerEnv:
    name = "kuhn_poker"
    aliases = ("kuhn",)
    num_players = 2
    zero_sum = True

    def __init__(self, seed: int = 0):
        import pyspiel

        self._game = pyspiel.load_game("kuhn_poker")
        self.state = None
        self.seed = seed
        self._rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> StepResult:
        if seed is not None:
            self.seed = seed
            self._rng = random.Random(seed)
        self.state = self._game.new_initial_state()
        # Chance nodes: deal cards
        while self.state.current_player() == -1 and not self.state.is_terminal():
            outcomes = self.state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = self._rng.choices(actions, weights=probs, k=1)[0]
            self.state.apply_action(action)
        return self._obs()

    def current_player(self) -> int:
        if self.state is None or self.state.is_terminal():
            return -1
        return int(self.state.current_player())

    def legal_actions(self) -> Dict[int, str]:
        if self.state is None or self.state.is_terminal():
            return {}
        out = {}
        for a in self.state.legal_actions():
            out[a] = f"<{_ACTION_NAMES.get(a, str(a))}>"
        return out

    def action_to_string(self, player_id: int, action: int) -> str:
        return f"<{_ACTION_NAMES.get(action, str(action))}>"

    def string_to_action(self, action_str: str) -> int:
        s = action_str.strip("<>").lower()
        for k, v in _ACTION_NAMES.items():
            if s == v:
                return k
        raise ValueError(f"unknown kuhn action {action_str!r}")

    def step(self, action: int | str) -> StepResult:
        if isinstance(action, str):
            action = self.string_to_action(action)
        self.state.apply_action(int(action))
        while (
            self.state is not None
            and not self.state.is_terminal()
            and self.state.current_player() == -1
        ):
            outcomes = self.state.chance_outcomes()
            actions, probs = zip(*outcomes)
            a = self._rng.choices(actions, weights=probs, k=1)[0]
            self.state.apply_action(a)
        return self._obs()

    def opponent_step(self, mbti: MBTIType | str | None = None) -> StepResult:
        legal = list(self.legal_actions().keys())
        # Bias: aggression prefers betting (action 1) when legal
        scores = {a: (1.0 if a == 1 else 0.2) for a in legal}
        action = choose_action(legal, mbti=mbti, rng=self._rng, action_scores=scores)
        return self.step(action)

    def render(self) -> str:
        if self.state is None:
            return ""
        return self.state.to_string()

    def private_card(self, player_id: int) -> str:
        if self.state is None:
            return "?"
        # Observation tensor: first 3 bits are private card one-hot for current viewer is hard;
        # use information state string.
        try:
            info = self.state.information_state_string(player_id)
            # e.g. starts with card letter
            return info
        except Exception:
            return "?"

    def get_prompt(self, player_id: int = 0, mbti: MBTIType | str | None = None) -> Dict[str, str]:
        base = (
            "You are an AI agent that plays Kuhn Poker. "
            "Always output exactly one legal action as <answer>...</answer>."
        )
        rules = (
            "Kuhn Poker: 3-card deck (J,Q,K), each player ante 1, private card each. "
            "Actions are <pass> or <bet>. Highest card wins if showdown."
        )
        info = f"You are player {player_id}. Prefer legal actions only."
        user = f"GAME RULES:\n{rules}\n\nPLAYER INFORMATION:\n{info}\n"
        return {"system": wrap_system_prompt(base, mbti), "user": user}

    def _obs(self) -> StepResult:
        done = self.state.is_terminal()
        if done:
            returns = list(self.state.returns())
            info = {"returns": returns, "winner": int(returns.index(max(returns)))}
            rewards = returns
        else:
            info = {}
            rewards = [0.0, 0.0]
        return StepResult(
            observation=self.render(),
            rewards=rewards,
            done=done,
            info=info,
            legal_actions=self.legal_actions(),
            current_player=self.current_player(),
        )

    def play_episode(
        self,
        *,
        agent_player: int = 0,
        agent_mbti: MBTIType | str | None = None,
        opponent_mbti: MBTIType | str | None = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        step = self.reset(seed)
        history: List[Dict[str, Any]] = []
        while not step.done:
            pid = step.current_player
            mbti = agent_mbti if pid == agent_player else opponent_mbti
            legal = list(step.legal_actions.keys())
            scores = {a: (1.0 if a == 1 else 0.2) for a in legal}
            action = choose_action(legal, mbti=mbti, rng=self._rng, action_scores=scores)
            action_str = step.legal_actions[action]
            step = self.step(action)
            history.append(
                {
                    "player": pid,
                    "action": action,
                    "action_str": action_str,
                    "mbti": getattr(mbti, "code", mbti),
                }
            )
        return {
            "game": self.name,
            "seed": self.seed,
            "history": history,
            "rewards": step.rewards,
            "info": step.info,
            "agent_player": agent_player,
            "agent_mbti": getattr(agent_mbti, "code", agent_mbti),
            "opponent_mbti": getattr(opponent_mbti, "code", opponent_mbti),
        }
