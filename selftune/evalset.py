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

# Dislike categories the self-tuning loop can actually ACT on: tool routing and
# anti-stall are the only levers the hints doc has. Other dislikes are real
# negatives but not fixable by routing hints — a factual error ("incorrect") is a
# grounding/verifier problem, "other" is unknown — so they're QUARANTINED from the
# proposer (else it tries to fix a grounding bug with a routing hint) and left on
# the turn record for verifier/human triage.
ROUTING_FAILURE_CATEGORIES = {"wrong_tool", "gave_up"}

# Mutating / outward-facing tools (the autonomy-approval surface from CLAUDE.md:
# publishing, messaging, GitHub/CRM/Smartsheet writes, FedRAMP/OSCAL edits, task
# creation, CI triggers). The scorer DEFAULT-DENIES these on benign must-pass
# cases and never accepts one as an anti-stall "recovery" — otherwise a plausible
# "when unsure, just call a tool" hint could promote itself by making the agent
# fire irreversible actions on a greeting. Read/investigative tools are allowed.
# Single source of truth; keep in sync as tools are added (test_write_tools_cover).
WRITE_TOOLS = {
    # outward publishing / messaging
    "social_publish_post", "social_schedule_post", "social_update_post",
    "social_delete_post", "send_teams_message",
    # GitHub mutations
    "github_create_issue", "github_close_issue", "github_edit_issue", "github_set_issue_type",
    "github_create_draft_issue", "github_add_item_to_project", "github_update_item_field",
    # other data writes
    "smartsheet_update_row",
    # background-task control
    "create_background_task", "cancel_background_task",
    "pause_background_task", "resume_background_task",
    # FedRAMP / OSCAL writes — incl. the OSCAL generators, which git-COMMIT to the
    # FedRAMP repo via _commit_file (the 'generate_' naming hides that they mutate).
    "fedramp_commit_document", "fedramp_propose_edit",
    "oscal_update_control", "oscal_migrate_from_markdown", "oscal_link_evidence",
    "oscal_generate_ssp", "oscal_generate_poam",
    # troubleshooting-store writes
    "save_troubleshooting_step", "delete_troubleshooting_step",
    # self-improvement triggers (kick off CI / compile)
    "trigger_wiki_compile", "trigger_engineering_compile",
    # skill authoring (writes support/skills/ in the in-boundary bucket)
    "save_skill", "delete_skill",
    # code sandbox: judge-gated execution of generated scripts (matches judge.WRITE_TOOL_NAMES)
    "run_code",
}

# "Meta" lookup tools that are ALWAYS bound (core group) and don't resolve a task.
# Excluded from what counts as an anti-stall "commitment" — otherwise a useless
# hint ("when unsure, call search_wiki") would flip the whole recovery bucket and
# game the gate. A real recovery commits to an investigative/task tool.
META_TOOLS = {
    "search_wiki", "read_knowledge", "get_skill",
    "search_memory", "manage_memory",
    "search_core_memory", "manage_core_memory",
    "search_team_memory", "manage_team_memory",
}

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
    # Human 👍/👎 (from the Teams feedback card) is the INDEPENDENT, authoritative
    # signal — it overrides the self-referential heuristic. A 👎 is 'bad' even if
    # tools returned ok; a 👍 is trusted over the error/give-up heuristics (which
    # can misfire). This is what breaks the "grade the model against its own past
    # choices" circularity. See conversation_log.record_feedback.
    reaction = (turn.get("feedback") or {}).get("reaction")
    if reaction == "dislike":
        return "bad"
    task = [(n, o) for n, o in parse_tools(turn.get("tools", [])) if n not in NOISE_TOOLS]
    if not task:
        return "skip"  # no task-tool selection to learn from
    if reaction == "like":
        return "good"
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
    # A 👍'd turn is a human-verified positive — a trustworthy label, unlike the
    # self-labeled 'mined' cases. Tagged so the gate can weight it higher (next step).
    human_verified = (turn.get("feedback") or {}).get("reaction") == "like"
    return {
        "message": turn["user_message"].strip(),
        "expected_tools": expected,
        "forbidden_tools": [],
        "source": "human-verified" if human_verified else "mined",
        "must_pass": False,
        "human_verified": human_verified,
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


def failure_route(turn: dict) -> str | None:
    """Where a failed turn should go.

    Returns:
      'tune'       — a failure the routing/anti-stall tuner should learn from
                     (heuristic tool-error/give-up, or a 👎 categorized wrong_tool/gave_up)
      'quarantine' — a real negative the tuner CANNOT fix (a 👎 categorized
                     'incorrect'/'other'/uncategorized) — kept for verifier/human triage
      None         — not a failure ('good'/'skip')

    For a human 👎 the category decides (we trust the human's read of the result);
    a heuristic 'bad' (no human verdict) is routing/anti-stall relevant by nature.
    """
    if label_turn(turn) != "bad":
        return None
    fb = turn.get("feedback") or {}
    if fb.get("reaction") == "dislike":
        return "tune" if (fb.get("category") or "") in ROUTING_FAILURE_CATEGORIES else "quarantine"
    return "tune"


def mine_failures(turns: Iterable[dict], max_failures: int = 20) -> list[dict]:
    """Recent tuner-actionable failures — the propose step's context.

    Only 'tune'-routed failures (see ``failure_route``); a factual-error 👎 is
    quarantined so the proposer doesn't try to fix grounding with a routing hint.
    Includes the human category + note (the propose model runs IN-BOUNDARY, so
    surfacing the user's note to it is compliant).
    """
    ordered = sorted(turns, key=lambda t: t.get("ts", ""), reverse=True)
    out: list[dict] = []
    for turn in ordered:
        if failure_route(turn) != "tune":
            continue
        tools = parse_tools(turn.get("tools", []))
        fb = turn.get("feedback") or {}
        out.append({
            "message": (turn.get("user_message") or "")[:200],
            "errored_tools": [n for n, o in tools if o == "error"],
            "gave_up": is_give_up(turn.get("assistant_response", "")) or fb.get("category") == "gave_up",
            "category": fb.get("category") or "",
            "note": (fb.get("text") or "")[:200],
        })
        if len(out) >= max_failures:
            break
    return out


def mine_recovery_cases(turns: Iterable[dict], max_cases: int = 40) -> list[dict]:
    """Stall turns → 'recovery' eval cases that make the anti-stall goal MEASURABLE.

    A turn where the agent gave up (heuristic give-up, or a 👎 categorized
    'gave_up') becomes a case whose pass condition is the OPPOSITE of stalling:
    for this message the agent should COMMIT to a task tool, not ask permission /
    bail. Scored via ``require_any_tool`` (pass = selects ≥1 non-noise tool). This
    is the bucket a real anti-stall hint can move — the gate can finally reward it.
    """
    ordered = sorted(turns, key=lambda t: t.get("ts", ""), reverse=True)
    seen: set[str] = set()
    out: list[dict] = []
    for turn in ordered:
        fb = turn.get("feedback") or {}
        stalled = is_give_up(turn.get("assistant_response", "")) or fb.get("category") == "gave_up"
        # Only tuner-actionable stalls (a quarantined dislike is excluded).
        if not stalled or failure_route(turn) != "tune":
            continue
        msg = (turn.get("user_message") or "").strip()
        if not msg or msg.lower() in seen:
            continue
        seen.add(msg.lower())
        out.append({
            "message": msg,
            "expected_tools": [],
            # Belt-and-suspenders: anti-stall recovery must commit to a SAFE
            # (read/investigative) tool, never an irreversible write — even if
            # _case_passes changes. The scorer enforces this too.
            "forbidden_tools": sorted(WRITE_TOOLS),
            "source": "recovery",
            "must_pass": False,
            "require_any_tool": True,
        })
        if len(out) >= max_cases:
            break
    return out


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
    """Mined + human-verified cases + anti-stall recovery cases + the safety seed.

    Buckets (see selftune.score): 'human' (👍, trusted driver), 'recovery'
    (anti-stall driver), 'mined' (self-labeled — floor only), 'safety' (must-pass).
    """
    turns = list(read_raw_turns(days=days))
    return (
        mine_cases(turns, max_cases=max_mined)
        + mine_recovery_cases(turns)
        + load_safety_seed()
    )


def save_eval_set(cases: list[dict], path: Path | None = None) -> None:
    p = path or EVAL_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")


def load_eval_set(path: Path | None = None) -> list[dict]:
    p = path or EVAL_PATH
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
