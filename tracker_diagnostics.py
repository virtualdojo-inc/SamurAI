"""Store + serving for pre-computed DH Tech Issue Tracker diagnostics.

The tracker-triage pipeline (``tracker_triage.py``) diagnoses each new/changed
DH Tech Issue Tracker row ahead of time — cross-referencing Cloud Logging and
code, pressure-testing a candidate fix, and categorizing it — then **parks** the
fact-grounded result here so that when any team member engages SamurAI, the
analysis is already done and instant to serve.

This is an in-boundary operational cache on the same ``/data`` GCS-FUSE SQLite
database that ``task_store`` uses (the diagnosis cross-references Cloud Logging,
which is in-boundary data, so it must not leave the boundary). It is **not**
curated knowledge — see ``docs/tech_issue_triage_plan.md``.

The pipeline never acts. Diagnostics are read-only recommendations; any filing
or config change happens later, with a human present, through the normal
judge-gated path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time

import aiosqlite
from langchain_core.tools import tool

from task_store import TASK_DB_PATH

logger = logging.getLogger(__name__)

# DH Tech Issue Tracker (Smartsheet). Stringified — 16-digit IDs lose precision
# as JSON numbers (see tools/smartsheet.py).
DH_TECH_TRACKER_SHEET_ID = "1146352141553540"

# Categories the triage assigns each row. Kept here so the store, the worker,
# and the tests share one source of truth.
CATEGORIES = {
    "A": "Tenant config tweak",
    "B": "Config change needing Devin's review",
    "C": "Future feature requirement",
    "D": "Backend code bug",
    "unknown": "Uncategorized",
}

_store: "DiagnosticsStore | None" = None


def row_content_hash(row: dict) -> str:
    """Stable hash of a tracker row's *content* (ignores _row_id/_row_number).

    Used as the watermark: when a row's content changes the hash changes, so the
    worker re-diagnoses it; unchanged rows are skipped. Keys are sorted so order
    is irrelevant.
    """
    content = {k: row[k] for k in sorted(row) if not k.startswith("_")}
    blob = json.dumps(content, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiagnosticsStore:
    """SQLite persistence for parked tracker diagnostics (on the task DB)."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    async def initialize(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """CREATE TABLE IF NOT EXISTS tracker_diagnostics (
                    row_id TEXT PRIMARY KEY,
                    sheet_id TEXT NOT NULL,
                    row_hash TEXT NOT NULL,
                    github_issue_no TEXT,
                    summary TEXT NOT NULL DEFAULT '',
                    category TEXT NOT NULL DEFAULT 'unknown',
                    suggested_type TEXT,
                    suggested_priority TEXT,
                    diagnosis TEXT NOT NULL,
                    model TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'diagnosed',
                    computed_at REAL NOT NULL
                )"""
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_diag_status "
                "ON tracker_diagnostics(status)"
            )
            await db.commit()
        logger.info("Tracker diagnostics store initialized: %s", self.db_path)

    async def needs_diagnosis(self, row_id: str, row_hash: str) -> bool:
        """True if this row has no stored diagnosis, or its content changed."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT row_hash, status FROM tracker_diagnostics WHERE row_id = ?",
                (str(row_id),),
            )
            existing = await cursor.fetchone()
        if existing is None:
            return True
        stored_hash, status = existing
        return stored_hash != row_hash or status == "stale"

    async def upsert_diagnosis(
        self,
        row_id: str,
        sheet_id: str,
        row_hash: str,
        diagnosis: str,
        *,
        github_issue_no: str | None = None,
        summary: str = "",
        category: str = "unknown",
        suggested_type: str | None = None,
        suggested_priority: str | None = None,
        model: str = "",
    ) -> None:
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO tracker_diagnostics
                   (row_id, sheet_id, row_hash, github_issue_no, summary,
                    category, suggested_type, suggested_priority, diagnosis,
                    model, status, computed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'diagnosed', ?)
                   ON CONFLICT(row_id) DO UPDATE SET
                       sheet_id = excluded.sheet_id,
                       row_hash = excluded.row_hash,
                       github_issue_no = excluded.github_issue_no,
                       summary = excluded.summary,
                       category = excluded.category,
                       suggested_type = excluded.suggested_type,
                       suggested_priority = excluded.suggested_priority,
                       diagnosis = excluded.diagnosis,
                       model = excluded.model,
                       status = 'diagnosed',
                       computed_at = excluded.computed_at""",
                (
                    str(row_id),
                    str(sheet_id),
                    row_hash,
                    github_issue_no,
                    summary,
                    category,
                    suggested_type,
                    suggested_priority,
                    diagnosis,
                    model,
                    now,
                ),
            )
            await db.commit()

    async def get(self, row_id: str) -> dict | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM tracker_diagnostics WHERE row_id = ?", (str(row_id),)
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def list_ready(
        self,
        category: str | None = None,
        github_issue_no: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        query = "SELECT * FROM tracker_diagnostics WHERE status = 'diagnosed'"
        params: list = []
        if category:
            query += " AND category = ?"
            params.append(category)
        if github_issue_no:
            query += " AND github_issue_no = ?"
            params.append(str(github_issue_no))
        query += " ORDER BY computed_at DESC LIMIT ?"
        params.append(limit)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def mark_stale(self, row_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE tracker_diagnostics SET status = 'stale' WHERE row_id = ?",
                (str(row_id),),
            )
            await db.commit()


async def get_diagnostics_store() -> DiagnosticsStore:
    """Get or create the singleton diagnostics store."""
    global _store
    if _store is None:
        _store = DiagnosticsStore(TASK_DB_PATH)
        await _store.initialize()
    return _store


def _ready_count_sync() -> int:
    """Synchronous count of ready diagnoses for the prompt index.

    ``_select_prompt_sections`` is sync, so this does a guarded read-only
    sqlite3 query. Any error (DB missing, table not yet created) yields 0 so it
    can never break prompt assembly.
    """
    try:
        conn = sqlite3.connect(TASK_DB_PATH, timeout=1.0)
        try:
            cur = conn.execute(
                "SELECT COUNT(*) FROM tracker_diagnostics WHERE status = 'diagnosed'"
            )
            return int(cur.fetchone()[0])
        finally:
            conn.close()
    except Exception:
        return 0


def tracker_diagnostics_index_text() -> str:
    """One-line prompt index (level-1 disclosure), mirrors knowledge_index_text."""
    count = _ready_count_sync()
    if count <= 0:
        return ""
    return (
        "## DH Tech Issue Tracker — prepared diagnoses\n"
        f"{count} tracker item(s) already have a fact-grounded diagnosis + "
        "recommendation ready (pre-computed, not yet acted on). When asked about "
        "the DH Tech Issue Tracker, a tracker item, or a tracked issue, call "
        "`get_tracker_diagnostics` to serve the prepared analysis instantly "
        "instead of re-diagnosing from scratch. These are recommendations only — "
        "never file an issue or change config without the user's explicit approval."
    )


def _format_diag_brief(d: dict) -> str:
    cat = d.get("category", "unknown")
    cat_label = CATEGORIES.get(cat, cat)
    bits = [f"**Row {d['row_id']}** — {cat} ({cat_label})"]
    if d.get("github_issue_no"):
        bits.append(f"GitHub #{d['github_issue_no']}")
    if d.get("suggested_type"):
        bits.append(f"type={d['suggested_type']}")
    if d.get("suggested_priority"):
        bits.append(f"priority={d['suggested_priority']}")
    header = " · ".join(bits)
    summary = d.get("summary") or ""
    return f"{header}\n  {summary}".rstrip()


@tool
async def get_tracker_diagnostics(
    github_issue_no: str = "",
    category: str = "",
    limit: int = 20,
) -> str:
    """Serve pre-computed DH Tech Issue Tracker diagnoses (read-only).

    The triage pipeline diagnoses tracker rows ahead of time and parks the
    fact-grounded result. Call this when a team member asks about the DH Tech
    Issue Tracker, a specific tracked item, or "what's been diagnosed" — it
    returns the prepared analysis with no diagnostic latency.

    Without filters: a list of ready diagnoses (one-line briefs). Pass
    ``github_issue_no`` to get the FULL diagnosis (cause, log evidence, candidate
    fix, adversarial-check result, recommendation) for that item.

    These are recommendations only. Never file an issue, apply a config change,
    or edit the tracker based on them without the user's explicit approval.

    Args:
        github_issue_no: Optional GitHub issue number to fetch one item's full
            diagnosis.
        category: Optional category filter — A (tenant config tweak),
            B (config change needing Devin's review), C (future feature),
            D (backend code bug).
        limit: Max items in the list view (default 20).
    """
    store = await get_diagnostics_store()

    if github_issue_no:
        matches = await store.list_ready(github_issue_no=github_issue_no, limit=5)
        if not matches:
            return (
                f"No prepared diagnosis for GitHub #{github_issue_no}. It may not "
                "be on the tracker yet, or the triage pipeline hasn't reached it."
            )
        out = []
        for d in matches:
            out.append(_format_diag_brief(d))
            out.append("\n" + d.get("diagnosis", "").strip())
        return "\n\n".join(out)

    cat = category.strip().upper() or None
    items = await store.list_ready(category=cat, limit=limit)
    if not items:
        return "No prepared tracker diagnoses are ready yet."
    lines = [f"**Prepared tracker diagnoses** ({len(items)})\n"]
    lines.extend(_format_diag_brief(d) for d in items)
    lines.append(
        "\nAsk for a specific GitHub issue number to see its full diagnosis. "
        "Recommendations only — get the user's approval before acting."
    )
    return "\n".join(lines)


TRACKER_DIAGNOSTICS_TOOLS = [get_tracker_diagnostics]
