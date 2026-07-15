"""DH Tech Issue Tracker triage pipeline (read-only, fact-grounded, in-boundary).

Runs frequently during business hours. For each new/changed tracker row it asks
the agent (via the ``tech-issue-triage`` skill) to diagnose the item against
Cloud Logging + code, pressure-test a candidate fix, and categorize it — then
parks the result in ``tracker_diagnostics`` so a team member gets it instantly.

**The pipeline never acts.** The agent is invoked read-only; the only write is
this trusted Python code upserting the diagnosis into the in-boundary ``/data``
cache. Filing an issue or changing config happens later, with a human present,
through the normal judge-gated path. See ``docs/tech_issue_triage_plan.md``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re

from tracker_diagnostics import (
    DH_TECH_TRACKER_SHEET_ID,
    CATEGORIES,
    get_diagnostics_store,
    row_content_hash,
)

logger = logging.getLogger(__name__)

# Bounded batch per tick: converges over ticks, and a Cloud Run drain costs at
# most one small batch (mirrors the KB compile's batching).
_DEFAULT_MAX_ROWS = int(os.environ.get("TRACKER_TRIAGE_MAX_ROWS", "5"))

# Conversation id for the read-only agent runs. Kept stable + distinct so these
# turns don't pollute a user's thread.
_TRIAGE_CONVERSATION_ID = "tracker_triage"

# The agent emits this machine-readable trailer the worker parses. The skill
# documents the same contract.
_FIELD_RE = {
    "category": re.compile(r"^CATEGORY:\s*([A-D]|unknown)\b", re.IGNORECASE | re.MULTILINE),
    "suggested_type": re.compile(r"^SUGGESTED_TYPE:\s*(\S+)", re.IGNORECASE | re.MULTILINE),
    "suggested_priority": re.compile(r"^SUGGESTED_PRIORITY:\s*(\S+)", re.IGNORECASE | re.MULTILINE),
    "summary": re.compile(r"^SUMMARY:\s*(.+)$", re.IGNORECASE | re.MULTILINE),
}
_NONE_TOKENS = {"none", "n/a", "na", "-", ""}


def triage_enabled() -> bool:
    """Kill switch — pipeline stays dormant unless explicitly enabled."""
    return os.environ.get("TRACKER_TRIAGE_ENABLED", "").strip().lower() in (
        "1", "true", "on", "yes",
    )


def _triage_prompt(row: dict) -> str:
    """Build the per-row diagnostic instruction for the agent."""
    issue_no = ""
    fields = []
    for k, v in row.items():
        if k.startswith("_"):
            continue
        fields.append(f"- {k}: {v}")
        if "github" in k.lower() and "issue" in k.lower():
            issue_no = str(v)
    fields_block = "\n".join(fields)
    hint = f"\nKnown GitHub issue number for this row: {issue_no}" if issue_no else ""
    return (
        "Run the `tech-issue-triage` skill (call get_skill('tech-issue-triage') "
        "first) to diagnose ONE DH Tech Issue Tracker row. Diagnose only — do not "
        "file anything, change any config, or edit the tracker.\n\n"
        f"Tracker row (sheet {DH_TECH_TRACKER_SHEET_ID}):\n{fields_block}{hint}\n\n"
        "Cross-reference the symptom against Cloud Logging and the code, form a "
        "likely cause and a candidate fix, adversarially pressure-test the fix, "
        "categorize the item, and end your reply with the exact machine-readable "
        "trailer the skill specifies (CATEGORY / SUGGESTED_TYPE / "
        "SUGGESTED_PRIORITY / SUMMARY). Ground every claim in a cited fact."
    )


def _parse_trailer(text: str) -> dict:
    """Extract the machine-readable fields from the agent's reply.

    Missing fields degrade gracefully (category -> 'unknown', others -> None).
    """
    out: dict = {}
    for field, rx in _FIELD_RE.items():
        m = rx.search(text or "")
        out[field] = m.group(1).strip() if m else None

    cat = (out.get("category") or "unknown").upper()
    out["category"] = cat if cat in CATEGORIES else "unknown"

    for key in ("suggested_type", "suggested_priority"):
        val = out.get(key)
        if val is None or val.strip().lower() in _NONE_TOKENS:
            out[key] = None
    out["summary"] = out.get("summary") or ""
    return out


async def run_triage_batch(
    *,
    max_rows: int | None = None,
    run_agent=None,
    fetch_rows=None,
    clear_thread=None,
) -> dict:
    """Diagnose up to ``max_rows`` new/changed tracker rows; park the results.

    ``run_agent``, ``fetch_rows``, and ``clear_thread`` are injectable for
    testing; in production they default to the live agent, the live Smartsheet
    read, and the live checkpoint cleaner.

    Returns ``{processed, diagnosed, skipped, candidates, remaining}``.
    """
    if max_rows is None:
        max_rows = _DEFAULT_MAX_ROWS

    if run_agent is None:
        from agent import run_agent as _live_run_agent

        run_agent = _live_run_agent
    if fetch_rows is None:
        from tools.smartsheet import get_sheet

        async def fetch_rows():
            return await asyncio.to_thread(get_sheet, DH_TECH_TRACKER_SHEET_ID)
    if clear_thread is None:
        from memory import clear_thread as _live_clear_thread

        clear_thread = _live_clear_thread

    store = await get_diagnostics_store()
    sheet = await fetch_rows()
    rows = sheet.get("rows", [])

    # Find rows that are new or whose content changed.
    candidates: list[tuple[dict, str]] = []
    for row in rows:
        row_id = row.get("_row_id")
        if not row_id:
            continue
        h = row_content_hash(row)
        if await store.needs_diagnosis(row_id, h):
            candidates.append((row, h))

    # One-time cleanup of the legacy shared triage conversation. All rows across
    # all runs used to share this single thread_id, so its checkpoint grew
    # unbounded (thousands of messages -> multi-second history trims on every
    # call + cross-row context bleed). Drop it; rows below now use isolated,
    # ephemeral per-row threads. Idempotent — a no-op once it's gone.
    try:
        await clear_thread(_TRIAGE_CONVERSATION_ID)
    except Exception:
        logger.warning("[tracker.triage] legacy thread cleanup skipped", exc_info=True)

    batch = candidates[:max_rows]
    diagnosed = 0
    for row, h in batch:
        row_id = str(row["_row_id"])
        # Diagnose each row in its OWN conversation, then clear it. Triage is
        # per-row and self-contained (the prompt carries the full row), so
        # history must neither accumulate across rows/runs nor leak between rows.
        triage_conv_id = f"{_TRIAGE_CONVERSATION_ID}:{row_id}"
        try:
            reply = await run_agent(
                user_message=_triage_prompt(row),
                conversation_id=triage_conv_id,
                is_background_task=True,
            )
        except Exception as e:  # one bad row must not stall the batch
            logger.error("[tracker.triage] row %s failed: %s: %s", row_id, type(e).__name__, e)
            continue
        finally:
            # Ephemeral: drop this row's checkpoint whether it succeeded or not.
            try:
                await clear_thread(triage_conv_id)
            except Exception:
                logger.warning(
                    "[tracker.triage] could not clear thread %s", triage_conv_id, exc_info=True
                )

        parsed = _parse_trailer(reply)
        issue_no = None
        for k, v in row.items():
            if k.startswith("_"):
                continue
            if "github" in k.lower() and "issue" in k.lower() and str(v).strip():
                issue_no = str(v).strip()
                break

        await store.upsert_diagnosis(
            row_id=row_id,
            sheet_id=DH_TECH_TRACKER_SHEET_ID,
            row_hash=h,
            diagnosis=reply,
            github_issue_no=issue_no,
            summary=parsed["summary"],
            category=parsed["category"],
            suggested_type=parsed["suggested_type"],
            suggested_priority=parsed["suggested_priority"],
            model=os.environ.get("GEMINI_MODEL", ""),
        )
        diagnosed += 1

    result = {
        "processed": len(rows),
        "candidates": len(candidates),
        "diagnosed": diagnosed,
        "skipped": len(rows) - len(candidates),
        "remaining": max(0, len(candidates) - len(batch)),
    }
    logger.info(
        "[tracker.triage] processed=%d diagnosed=%d remaining=%d",
        result["processed"], result["diagnosed"], result["remaining"],
    )
    return result
