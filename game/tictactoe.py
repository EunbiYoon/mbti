"""Tic-Tac-Toe via OpenSpiel."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from opponent.mbti_types import MBTIType
from opponent.policies import choose_action
from opponent.prompts import wrap_system_prompt

from .base import StepResult


class TicTacToeEnv:
    name = "tictactoe"
    aliases = ("tic-tac-toe", "ttt")
    num_players = 2
    zero_sum = True

    def __init__(self, seed: int = 0):
        import pyspiel

        self._game = pyspiel.load_game("tic_tac_toe")
        self.state = None
        self.seed = seed
        self._rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> StepResult:
        if seed is not None:
            self.seed = seed
            self._rng = random.Random(seed)
        self.state = self._game.new_initial_state()
        return self._obs()

    def current_player(self) -> int:
        if self.state is None or self.state.is_terminal():
            return -1
        return int(self.state.current_player())

    def legal_actions(self) -> Dict[int, str]:
        if self.state is None or self.state.is_terminal():
            return {}
        pid = self.current_player()
        out = {}
        for a in self.state.legal_actions():
            out[a] = self.action_to_string(pid, a)
        return out

    def action_to_string(self, player_id: int, action: int) -> str:
        mark = "X" if player_id == 0 else "O"
        row, col = divmod(action, 3)
        return f"<{mark}({row},{col})>"

    def string_to_action(self, action_str: str) -> int:
        # <X(r,c)> or <O(r,c)>
        row = int(action_str[3])
        col = int(action_str[5])
        return row * 3 + col

    def step(self, action: int | str) -> StepResult:
        if isinstance(action, str):
            action = self.string_to_action(action)
        self.state.apply_action(int(action))
        return self._obs()

    def opponent_step(self, mbti: MBTIType | str | None = None) -> StepResult:
        legal = list(self.legal_actions().keys())
        action = choose_action(legal, mbti=mbti, rng=self._rng)
        return self.step(action)

    def render(self) -> str:
        if self.state is None:
            return ""
        return self.state.to_string()

    def get_prompt(self, player_id: int = 0, mbti: MBTIType | str | None = None) -> Dict[str, str]:
        mark = "X" if player_id == 0 else "O"
        base = (
            "You are an AI agent that makes decisions in tic-tac-toe. "
            "Always output exactly one legal action as <answer>...</answer>."
        )
        rules = (
            "Tic-tac-toe is played on a 3x3 grid. Marks are X and O. "
            "First to three-in-a-row wins; otherwise the game draws."
        )
        info = (
            f"Your mark is {mark}. Legal actions look like <{mark}(row,col)> "
            "with 0-indexed row/col."
        )
        user = f"GAME RULES:\n{rules}\n\nPLAYER INFORMATION:\n{info}\n"
        return {"system": wrap_system_prompt(base, mbti), "user": user}

    def _obs(self) -> StepResult:
        done = self.state.is_terminal()
        rewards = list(self.state.rewards()) if done else [0.0, 0.0]
        if done:
            returns = list(self.state.returns())
            winner = int(returns.index(max(returns))) if returns[0] != returns[1] else -1
            info = {"winner": winner, "returns": returns}
            rewards = returns
        else:
            info = {}
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
        """Both sides heuristic MBTI bots (for data / smoke)."""
        step = self.reset(seed)
        history: List[Dict[str, Any]] = []
        while not step.done:
            pid = step.current_player
            mbti = agent_mbti if pid == agent_player else opponent_mbti
            legal = list(step.legal_actions.keys())
            action = choose_action(legal, mbti=mbti, rng=self._rng)
            action_str = step.legal_actions[action]
            step = self.step(action)
            history.append(
                {
                    "player": pid,
                    "action": action,
                    "action_str": action_str,
                    "mbti": str(mbti) if not hasattr(mbti, "code") else getattr(mbti, "code", mbti),
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
