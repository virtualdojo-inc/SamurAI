"""Build prompt-tuning eval cases from saved conversations.

The self-tuning loop (later) edits ONE mutable prompt doc and must only promote a
change if it doesn't regress an eval set. This module builds that eval set from
the per-turn conversation log SamurAI already writes to ``/data/raw/<date>/*.json``
(see ``conversation_log.py``), using objective success signals already present in
the data — so the loop is graded against reality, not its own opinion.

Label per turn (from the saved ``tools`` field + the response):
  good = interactive turn, every task-tool returned ``ok``, response isn't a
         give-up → an exemplar of "these tools were the right call for this message"
  bad  = any tool ``error`` OR a give-up response → don't learn from it
  skip = background task, or no task-tools to learn from

Eval case schema (one shape for mined + safety-seed cases):
  {message, expected_tools, forbidden_tools, source, must_pass}
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable, Iterator

DATA_DIR = os.environ.get("SAMURAI_DATA_DIR", "/data")
RAW_DIR = Path(DATA_DIR) / "raw"
EVAL_PATH = Path(DATA_DIR) / "selftune" / "evals.jsonl"
SAFETY_SEED = Path(__file__).parent / "seeds" / "safety.jsonl"

# Status/bookkeeping tools — not "task" choices, so they don't count toward the
# tool-selection signal we evaluate.
NOISE_TOOLS = {"update_progress", "get_progress", "clear_progress"}

# Phrases that mark a turn where the agent gave up / stalled instead of completing
# (the exact failure mode seen in Jason's stop-before-finishing run).
_GIVE_UP_RE = re.compile(
    r"(would you like me to continue|i wasn'?t able to|i was unable to|"
    r"stopped before finishing|cut short|i'?m not sure what the next steps|"
    r"let me know if you'?d like me to|i can'?t (do|complete|help with) that|"
    r"please try again)",
    re.IGNORECASE,
)
_TOOL_RE = re.compile(r"^([A-Za-z0-9_]+):\s+(ok|error)\b")


def parse_tools(tools: Iterable[str]) -> list[tuple[str, str]]:
    """Parse ['name: ok -> …', 'name: error -> …'] into [(name, outcome)]."""
    out: list[tuple[str, str]] = []
    for entry in tools or []:
        m = _TOOL_RE.match(str(entry).strip())
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def is_give_up(response: str) -> bool:
    return bool(_GIVE_UP_RE.search(response or ""))


def label_turn(turn: dict) -> str:
    """Return 'good' | 'bad' | 'skip' for a saved turn record."""
    if turn.get("is_background_task"):
        return "skip"
    if not (turn.get("user_message") or "").strip():
        return "skip"
    task = [(n, o) for n, o in parse_tools(turn.get("tools", [])) if n not in NOISE_TOOLS]
    if not task:
        return "skip"  # no task-tool selection to learn from
    if any(o == "error" for _, o in task):
        return "bad"
    if is_give_up(turn.get("assistant_response", "")):
        return "bad"
    return "good"


def turn_to_case(turn: dict) -> dict | None:
    """A 'good' turn → a mined eval case. Returns None if not usable."""
    if label_turn(turn) != "good":
        return None
    expected = sorted({n for n, o in parse_tools(turn.get("tools", []))
                       if o == "ok" and n not in NOISE_TOOLS})
    if not expected:
        return None
    return {
        "message": turn["user_message"].strip(),
        "expected_tools": expected,
        "forbidden_tools": [],
        "source": "mined",
        "must_pass": False,
    }


def mine_cases(turns: Iterable[dict], max_cases: int = 200) -> list[dict]:
    """Build mined eval cases (newest-first, deduped by message, capped)."""
    ordered = sorted(turns, key=lambda t: t.get("ts", ""), reverse=True)
    seen: set[str] = set()
    cases: list[dict] = []
    for turn in ordered:
        case = turn_to_case(turn)
        if not case:
            continue
        key = case["message"].lower()
        if key in seen:
            continue
        seen.add(key)
        cases.append(case)
        if len(cases) >= max_cases:
            break
    return cases


def read_raw_turns(days: int = 7, raw_dir: Path | None = None) -> Iterator[dict]:
    """Yield turn records from /data/raw for the last ``days`` date partitions.

    Runs in-boundary on samurai-bot (reads the GCS-FUSE mount). Best-effort.
    """
    base = raw_dir or RAW_DIR
    if not base.is_dir():
        return
    day_dirs = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)[:days]
    for d in day_dirs:
        for f in d.glob("*.json"):
            try:
                yield json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue


def load_safety_seed(path: Path | None = None) -> list[dict]:
    """Load the committed, hand-curated must-pass safety cases."""
    p = path or SAFETY_SEED
    cases: list[dict] = []
    if not p.is_file():
        return cases
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError:
            continue
        c.setdefault("expected_tools", [])
        c.setdefault("forbidden_tools", [])
        c["source"] = "safety-seed"
        c["must_pass"] = True
        cases.append(c)
    return cases


def build_eval_set(days: int = 7, max_mined: int = 200) -> list[dict]:
    """Mined cases from recent conversations + the committed safety seed."""
    return mine_cases(read_raw_turns(days=days), max_cases=max_mined) + load_safety_seed()


def save_eval_set(cases: list[dict], path: Path | None = None) -> None:
    p = path or EVAL_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")


def load_eval_set(path: Path | None = None) -> list[dict]:
    p = path or EVAL_PATH
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
