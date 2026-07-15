"""Game registry."""

from __future__ import annotations

from typing import Any, Dict, Type

from .kuhn_poker import KuhnPokerEnv
from .minihanabi import MiniHanabiEnv
from .tictactoe import TicTacToeEnv

GAMES: Dict[str, Type] = {
    "tictactoe": TicTacToeEnv,
    "kuhn_poker": KuhnPokerEnv,
    "minihanabi": MiniHanabiEnv,
}

_ALIAS: Dict[str, str] = {}
for key, cls in GAMES.items():
    _ALIAS[key] = key
    for a in getattr(cls, "aliases", ()):
        _ALIAS[a.lower().replace("-", "_")] = key


def resolve_game(name: str) -> str:
    key = _ALIAS.get(name.lower().replace("-", "_"))
    if key is None:
        raise ValueError(f"unknown game {name!r}; valid: {', '.join(sorted(GAMES))}")
    return key


def make_env(name: str, **kwargs: Any):
    key = resolve_game(name)
    return GAMES[key](**kwargs)


def list_games():
    return sorted(GAMES.keys())
