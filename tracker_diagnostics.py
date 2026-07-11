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
import os
import sqlite3
import time

from langchain_core.tools import tool
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.models import Base, TRACKER_DIAGNOSTICS_TABLES, TrackerDiagnostic
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


def _row_to_dict(obj: TrackerDiagnostic) -> dict:
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}


class DiagnosticsStore:
    """Persistence for parked tracker diagnostics (async SQLAlchemy).

    Runs on the Postgres backbone in prod (DATABASE_URL) and on a SQLite file
    for tests/local — the same dual-backend pattern as ``task_store.TaskStore``.
    The previous raw-aiosqlite table on the GCS-FUSE file corrupted under
    concurrent writers ("database disk image is malformed", 2026-07).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._url = db_path if "://" in db_path else f"sqlite+aiosqlite:///{db_path}"
        self._engine = None
        self._sessionmaker: async_sessionmaker | None = None

    async def initialize(self) -> None:
        """Create the engine + the diagnostics table. Idempotent."""
        if self._engine is None:
            self._engine = create_async_engine(self._url, future=True)
            self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)
        async with self._engine.begin() as conn:
            await conn.run_sync(
                lambda c: Base.metadata.create_all(
                    c, tables=TRACKER_DIAGNOSTICS_TABLES, checkfirst=True
                )
            )
        logger.info("Tracker diagnostics store initialized: %s", self.db_path)
        await self._refresh_ready_count()

    async def _refresh_ready_count(self) -> None:
        """Keep the sync prompt-index count cache warm (see tracker_diagnostics_index_text)."""
        global _ready_count_cache, _ready_count_ts
        try:
            async with self._sessionmaker() as session:
                result = await session.execute(
                    select(func.count())
                    .select_from(TrackerDiagnostic)
                    .where(TrackerDiagnostic.status == "diagnosed")
                )
                _ready_count_cache = int(result.scalar_one())
                _ready_count_ts = time.time()
        except Exception as e:  # cache only — never break a write path
            logger.warning("[tracker.diag] ready-count refresh failed: %s", e)

    async def needs_diagnosis(self, row_id: str, row_hash: str) -> bool:
        """True if this row has no stored diagnosis, or its content changed."""
        async with self._sessionmaker() as session:
            obj = await session.get(TrackerDiagnostic, str(row_id))
        if obj is None:
            return True
        return obj.row_hash != row_hash or obj.status == "stale"

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
        fields = dict(
            sheet_id=str(sheet_id),
            row_hash=row_hash,
            github_issue_no=github_issue_no,
            summary=summary,
            category=category,
            suggested_type=suggested_type,
            suggested_priority=suggested_priority,
            diagnosis=diagnosis,
            model=model,
            status="diagnosed",
            computed_at=time.time(),
        )
        async with self._sessionmaker() as session:
            obj = await session.get(TrackerDiagnostic, str(row_id))
            if obj is None:
                session.add(TrackerDiagnostic(row_id=str(row_id), **fields))
            else:
                for k, v in fields.items():
                    setattr(obj, k, v)
            await session.commit()
        await self._refresh_ready_count()

    async def get(self, row_id: str) -> dict | None:
        async with self._sessionmaker() as session:
            obj = await session.get(TrackerDiagnostic, str(row_id))
            return _row_to_dict(obj) if obj else None

    async def list_ready(
        self,
        category: str | None = None,
        github_issue_no: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        stmt = select(TrackerDiagnostic).where(TrackerDiagnostic.status == "diagnosed")
        if category:
            stmt = stmt.where(TrackerDiagnostic.category == category)
        if github_issue_no:
            stmt = stmt.where(TrackerDiagnostic.github_issue_no == str(github_issue_no))
        stmt = stmt.order_by(TrackerDiagnostic.computed_at.desc()).limit(limit)
        async with self._sessionmaker() as session:
            result = await session.execute(stmt)
            return [_row_to_dict(o) for o in result.scalars().all()]

    async def mark_stale(self, row_id: str) -> None:
        async with self._sessionmaker() as session:
            await session.execute(
                update(TrackerDiagnostic)
                .where(TrackerDiagnostic.row_id == str(row_id))
                .values(status="stale")
            )
            await session.commit()
        await self._refresh_ready_count()


async def get_diagnostics_store() -> DiagnosticsStore:
    """Get or create the singleton diagnostics store.

    Uses DATABASE_URL (Postgres) in prod; falls back to the SQLite file when
    DATABASE_URL is unset (tests/local) — mirrors ``task_store.get_task_store``.
    """
    global _store
    if _store is None:
        url = os.environ.get("DATABASE_URL")
        _store = DiagnosticsStore(url or TASK_DB_PATH)
        await _store.initialize()
    return _store


# Prompt-index count cache. ``_select_prompt_sections`` is sync and runs on
# every graph hop, so it must not do per-hop I/O (the old version ran a
# sqlite3 query against the GCS-FUSE file on every hop). The cache is refreshed
# by the store on initialize and after every write.
_ready_count_cache: int | None = None
_ready_count_ts: float = 0.0
_READY_COUNT_TTL = 300  # sqlite fallback re-query interval


def _ready_count_sync() -> int:
    """Synchronous count of ready diagnoses for the prompt index.

    Guarded read-only sqlite3 query — the no-DATABASE_URL fallback only. Any
    error (DB missing, table not yet created) yields 0 so it can never break
    prompt assembly.
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


def _ready_count() -> int:
    global _ready_count_cache, _ready_count_ts
    fresh = _ready_count_cache is not None and (time.time() - _ready_count_ts) < _READY_COUNT_TTL
    if fresh:
        return _ready_count_cache
    if os.environ.get("DATABASE_URL"):
        # Postgres: never issue a blocking sync query from prompt assembly.
        # Serve the (possibly stale) cache; the store refreshes it on writes.
        return _ready_count_cache or 0
    _ready_count_cache = _ready_count_sync()
    _ready_count_ts = time.time()
    return _ready_count_cache


def tracker_diagnostics_index_text() -> str:
    """One-line prompt index (level-1 disclosure), mirrors knowledge_index_text."""
    count = _ready_count()
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
