"""Mini-Hanabi (small OpenSpiel Hanabi) — cooperative 2P."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from opponent.mbti_types import MBTIType
from opponent.policies import choose_action
from opponent.prompts import wrap_system_prompt

from .base import StepResult


def _minihanabi_params() -> Dict[str, Any]:
    # Compact coop variant used by marshal MiniHanabi
    return {
        "players": 2,
        "colors": 2,
        "ranks": 2,
        "hand_size": 3,
        "max_information_tokens": 3,
        "max_life_tokens": 3,
    }


class MiniHanabiEnv:
    name = "minihanabi"
    aliases = ("hanabi", "mini_hanabi", "mini-hanabi")
    num_players = 2
    zero_sum = False

    def __init__(self, seed: int = 0):
        import pyspiel

        params = _minihanabi_params()
        self._game = pyspiel.load_game("hanabi", params)
        self.state = None
        self.seed = seed
        self._rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> StepResult:
        if seed is not None:
            self.seed = seed
            self._rng = random.Random(seed)
        self.state = self._game.new_initial_state()
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
            try:
                label = self.state.action_to_string(self.current_player(), a)
            except Exception:
                label = str(a)
            out[a] = f"<{label}>"
        return out

    def action_to_string(self, player_id: int, action: int) -> str:
        try:
            return f"<{self.state.action_to_string(player_id, action)}>"
        except Exception:
            return f"<{action}>"

    def string_to_action(self, action_str: str) -> int:
        target = action_str.strip()
        if not (target.startswith("<") and target.endswith(">")):
            target = f"<{target}>"
        for a, s in self.legal_actions().items():
            if s == target or s.strip("<>") == target.strip("<>"):
                return a
        raise ValueError(f"unknown / illegal hanabi action {action_str!r}")

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
        legal_map = self.legal_actions()
        legal = list(legal_map.keys())
        # Prefer hint-like / informative strings when info_focus is high (string heuristic)
        scores = {}
        for a, s in legal_map.items():
            sl = s.lower()
            score = 0.3
            if "reveal" in sl or "hint" in sl or "color" in sl or "rank" in sl:
                score += 1.0
            if "play" in sl:
                score += 0.6
            if "discard" in sl:
                score += 0.2
            scores[a] = score
        action = choose_action(legal, mbti=mbti, rng=self._rng, action_scores=scores)
        return self.step(action)

    def render(self) -> str:
        if self.state is None:
            return ""
        return self.state.to_string()

    def get_prompt(self, player_id: int = 0, mbti: MBTIType | str | None = None) -> Dict[str, str]:
        base = (
            "You are an AI agent cooperating in Mini-Hanabi. "
            "Always output exactly one legal action as <answer>...</answer>."
        )
        rules = (
            "Mini-Hanabi is a cooperative imperfect-information card game. "
            "Players give hints, play cards, or discard to build color stacks. "
            "Maximize shared score; avoid losing life tokens."
        )
        info = f"You are player {player_id}. Coordinate with your teammate."
        user = f"GAME RULES:\n{rules}\n\nPLAYER INFORMATION:\n{info}\n"
        return {"system": wrap_system_prompt(base, mbti), "user": user}

    def _obs(self) -> StepResult:
        done = self.state.is_terminal()
        if done:
            returns = list(self.state.returns())
            info = {"returns": returns, "score": float(sum(returns)) / max(1, len(returns))}
            rewards = returns
        else:
            info = {}
            rewards = [0.0] * self.num_players
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
        max_steps: int = 40,
    ) -> Dict[str, Any]:
        step = self.reset(seed)
        history: List[Dict[str, Any]] = []
        for _ in range(max_steps):
            if step.done:
                break
            pid = step.current_player
            mbti = agent_mbti if pid == agent_player else opponent_mbti
            # reuse opponent_step scoring via temporary rng call
            legal_map = step.legal_actions
            legal = list(legal_map.keys())
            scores = {}
            for a, s in legal_map.items():
                sl = s.lower()
                score = 0.3
                if "reveal" in sl or "hint" in sl or "color" in sl or "rank" in sl:
                    score += 1.0
                if "play" in sl:
                    score += 0.6
                if "discard" in sl:
                    score += 0.2
                scores[a] = score
            action = choose_action(legal, mbti=mbti, rng=self._rng, action_scores=scores)
            action_str = legal_map[action]
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
