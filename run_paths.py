"""Run layout: runs/<timestamp>/{lora,eval}/ — both subdirs always exist."""

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
    """Create runs/<timestamp>/ with both lora/ and eval/ subfolders."""
    rid = resolve_run_id(run_id)
    run_dir = RUNS_DIR / rid
    (run_dir / "lora").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    return run_dir


def new_lora_dir(run_id: str | None = None) -> Path:
    """runs/<timestamp>/lora/ (also ensures sibling eval/)."""
    run_dir = new_run_dir(run_id)
    return run_dir / "lora"


def new_eval_dir(run_id: str | None = None) -> Path:
    """runs/<timestamp>/eval/ (also ensures sibling lora/)."""
    run_dir = new_run_dir(run_id)
    return run_dir / "eval"


def run_id_from_adapter(adapter: Path | str) -> str | None:
    """If adapter is runs/<ts>/lora[/...], return <ts>; else None."""
    p = Path(adapter).resolve()
    try:
        rel = p.relative_to(RUNS_DIR.resolve())
    except ValueError:
        # try string match against project runs/
        parts = p.parts
        if "runs" not in parts:
            return None
        i = parts.index("runs")
        if i + 1 >= len(parts):
            return None
        return parts[i + 1]
    if len(rel.parts) >= 2 and rel.parts[1] == "lora":
        return rel.parts[0]
    if len(rel.parts) >= 1:
        # runs/<ts>/lora or deeper under lora
        return rel.parts[0]
    return None


def resolve_eval_run_id(
    run_id: str | None = None,
    adapter: Path | str | None = None,
) -> str:
    """Prefer explicit run_id, else adapter's parent timestamp, else RUN_ID/new stamp."""
    if run_id:
        return run_id
    if adapter is not None:
        inferred = run_id_from_adapter(adapter)
        if inferred:
            return inferred
    return resolve_run_id(None)


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


def read_latest_run_id(kind: str = "lora") -> str | None:
    latest = RUNS_DIR / f"latest_{kind}.json"
    if not latest.is_file():
        return None
    return json.loads(latest.read_text(encoding="utf-8")).get("run_id")
