"""Run layout: runs/<timestamp>/{lora,eval}/."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
OPPONENT_DIR = PROJECT_ROOT / "opponent"
GAME_DIR = PROJECT_ROOT / "game"
LORA_DIR = PROJECT_ROOT / "lora"
EVAL_DIR = PROJECT_ROOT / "eval"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def resolve_run_id(run_id: str | None = None) -> str:
    return run_id or os.environ.get("RUN_ID") or utc_stamp()


def new_run_dir(run_id: str | None = None) -> Path:
    rid = resolve_run_id(run_id)
    run_dir = RUNS_DIR / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def new_lora_dir(run_id: str | None = None) -> Path:
    """runs/<timestamp>/lora/"""
    out = new_run_dir(run_id) / "lora"
    out.mkdir(parents=True, exist_ok=True)
    return out


def new_eval_dir(run_id: str | None = None) -> Path:
    """runs/<timestamp>/eval/"""
    out = new_run_dir(run_id) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_latest_pointer(kind: str, run_subdir: Path) -> None:
    """kind in {'lora','eval'}; pointer at runs/latest_<kind>.json."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rel = run_subdir.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    payload = {
        "run_id": run_subdir.parent.name,
        "path": rel,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (RUNS_DIR / f"latest_{kind}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def read_latest_path(kind: str) -> Path | None:
    latest = RUNS_DIR / f"latest_{kind}.json"
    if not latest.is_file():
        return None
    data = json.loads(latest.read_text(encoding="utf-8"))
    path = data.get("path")
    if not path:
        return None
    p = PROJECT_ROOT / path
    return p if p.is_dir() else None
