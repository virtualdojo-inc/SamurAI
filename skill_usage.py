"""SamurAI skill-usage telemetry emit (plan Part E, SamurAI side).

SamurAI's GitHub App is ``contents:read`` only — it cannot push files or open PRs to the
shared catalog. So skill-usage telemetry crosses the same way skill drafts do: via a
labeled **issue** (the bot has ``issues:write``), which a workflow in virtualdojo-skills
turns into a ``telemetry/raw/samurai/`` file and then closes.

Names + counts only (C5-clean). Collection is a fast in-memory counter incremented from
``skills.get_skill`` (no I/O on the event loop); a scheduled flush batches it into one
issue per period. On emit failure the snapshot is restored so counts aren't lost (across
a process restart they are — best-effort, which is fine for usage telemetry).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import Counter

logger = logging.getLogger(__name__)

SKILLS_REPO = os.environ.get("SKILLS_REPO", "virtualdojo-inc/virtualdojo-skills")
_NAME_RE = re.compile(r"^[a-z0-9-]{1,64}$")

_counts: Counter = Counter()
_lock = threading.Lock()


def usage_enabled() -> bool:
    return os.environ.get("SKILLS_USAGE_ENABLED", "off").lower() not in ("off", "", "0", "false", "no")


def record(name: str) -> None:
    """Increment the in-memory usage counter for a skill. Fast, no I/O, never raises."""
    try:
        if isinstance(name, str) and _NAME_RE.match(name):
            with _lock:
                _counts[name] += 1
    except Exception:  # pragma: no cover - telemetry must never break a turn
        pass


def _snapshot_and_reset() -> dict:
    with _lock:
        snap = dict(_counts)
        _counts.clear()
    return snap


def _restore(snap: dict) -> None:
    with _lock:
        for k, v in snap.items():
            _counts[k] += v


def _issue_body(snap: dict) -> str:
    payload = [{"skill": n, "count": c} for n, c in sorted(snap.items())]
    return (
        "<!-- skill-usage -->\n"
        "SamurAI skill-usage telemetry (names + counts only). A workflow in this repo "
        "records it under telemetry/raw/samurai/ and closes this issue.\n\n"
        "```json\n" + json.dumps(payload) + "\n```\n"
    )


def emit_usage(force: bool = False) -> dict:
    """Flush the in-memory counter into one labeled ``skill-usage`` issue. Best-effort."""
    if not force and not usage_enabled():
        return {"skipped": True}
    snap = _snapshot_and_reset()
    if not snap:
        return {"emitted": 0}
    try:
        from tools.github import _github
        repo = _github().get_repo(SKILLS_REPO)
        repo.create_issue(
            title=f"skill-usage: samurai ({sum(snap.values())} invocations)",
            body=_issue_body(snap),
            labels=["skill-usage"],
        )
        logger.info("[skills.usage] emitted %d invocations across %d skills",
                    sum(snap.values()), len(snap))
        return {"emitted": sum(snap.values()), "skills": len(snap)}
    except Exception as e:
        _restore(snap)  # don't lose counts on a transient failure
        logger.warning("[skills.usage] emit failed (counts restored): %s", e)
        return {"error": str(e)}
