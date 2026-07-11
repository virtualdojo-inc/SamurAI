"""SQLAlchemy 2.x async declarative models for SamurAI's Postgres backbone.

Mirrors the CMO service's ``db/models.py`` conventions (DeclarativeBase,
``_uuid_pk``/``_timestamps`` helpers, JSONB, ``pgvector.Vector``). First tables:

- ``pending_approvals`` — the durable record behind the reusable Approve/Revise/
  Reject Teams card (see ``docs/teams_approval_card_design.md``). The card carries
  ONLY ``request_id``; the verified clicker is the authority and the action payload
  lives here server-side, so there is nothing capability-bearing in the card to
  forge. Single-use is enforced by an atomic ``UPDATE ... WHERE status='pending'``.
- ``code_runs`` — the sandbox script library: every approved/executed script, so
  the agent can reuse a vetted script instead of regenerating (the Voyager "code
  skill" pattern). ``embedding`` enables similarity-reuse once an in-boundary
  embedder is wired.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Gemini text-embedding-004 is 768-dim. Adjust if the wired embedder differs; the
# column is nullable and unused until similarity-reuse is implemented.
EMBED_DIM = 768


class Base(DeclarativeBase):
    """Shared declarative base for all SamurAI ORM models."""


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _timestamps() -> tuple[Mapped[datetime], Mapped[datetime]]:
    created = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    return created, updated  # type: ignore[return-value]


class PendingApproval(Base):
    """Durable record behind the reusable Approve/Revise/Reject Teams card."""

    __tablename__ = "pending_approvals"

    request_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(500), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    payload_sha256: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    # pending | claimed | approved | rejected | revising | expired
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending", server_default="pending"
    )
    requested_by: Mapped[str] = mapped_column(String(320), nullable=False, default="")
    approver_email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    activity_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    revise_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (Index("ix_pending_approvals_status", "status"),)


class CodeRun(Base):
    """Library of sandbox-executed scripts — reuse a vetted script instead of
    regenerating. Populated only with approved, executed runs."""

    __tablename__ = "code_runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    language: Mapped[str] = mapped_column(String(32), nullable=False, default="python")
    script: Mapped[str] = mapped_column(Text, nullable=False)
    inputs_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    outcome: Mapped[str] = mapped_column(String(16), nullable=False, default="ok")  # ok | fail
    result_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    approved_by: Mapped[str | None] = mapped_column(String(320), nullable=True)
    reusable: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM), nullable=True)
    created_at, updated_at = _timestamps()

    __table_args__ = (Index("ix_code_runs_reusable", "reusable"),)


# ── Background tasks / conversation refs / team roster ───────────────────
# Migrated from the SQLite-on-GCS-FUSE store. Cross-dialect generic types
# (String/Text/Integer/Float) so the SAME models run on SQLite (tests/local,
# no DATABASE_URL) and Postgres (prod). Timestamps stay epoch-float (set in
# task_store) to preserve the existing dict-return contract.


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(8), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(320), nullable=False)
    user_name: Mapped[str] = mapped_column(String(320), nullable=False, default="")
    user_email: Mapped[str] = mapped_column(String(320), nullable=False, default="")
    user_timezone: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    conversation_id: Mapped[str] = mapped_column(String(500), nullable=False)
    task_type: Mapped[str] = mapped_column(String(40), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    cron_expression: Mapped[str | None] = mapped_column(String(120), nullable=True)
    run_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    created_at: Mapped[float] = mapped_column(Float, nullable=False, default=time.time)
    last_run_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    next_run_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    run_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    max_failures: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    locked_until: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    __table_args__ = (
        Index("idx_tasks_user", "user_id"),
        Index("idx_tasks_status", "status"),
    )


class ConversationRef(Base):
    __tablename__ = "conversation_refs"

    conversation_id: Mapped[str] = mapped_column(String(500), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(320), nullable=False)
    ref_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False, default=time.time)


class TeamRoster(Base):
    __tablename__ = "team_roster"

    email: Mapped[str] = mapped_column(String(320), primary_key=True)
    teams_id: Mapped[str] = mapped_column(String(500), nullable=False)
    display_name: Mapped[str] = mapped_column(String(320), nullable=False, default="")
    service_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    updated_at: Mapped[float] = mapped_column(Float, nullable=False, default=time.time)


class TrackerDiagnostic(Base):
    """Parked DH Tech Issue Tracker diagnoses (see tracker_diagnostics.py).

    Migrated off the raw-aiosqlite table on the GCS-FUSE file, which corrupted
    ("database disk image is malformed") under concurrent writers — same failure
    class that moved the task store to Postgres. Generic types so the model runs
    on SQLite (tests/local) and Postgres (prod).
    """

    __tablename__ = "tracker_diagnostics"

    row_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    sheet_id: Mapped[str] = mapped_column(String(64), nullable=False)
    row_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    github_issue_no: Mapped[str | None] = mapped_column(String(20), nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    category: Mapped[str] = mapped_column(String(20), nullable=False, default="unknown")
    suggested_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    suggested_priority: Mapped[str | None] = mapped_column(String(20), nullable=True)
    diagnosis: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="diagnosed")
    computed_at: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (Index("idx_diag_status", "status"),)


# Tables migrated from the old SQLite task store — created via create_all on the
# store's own engine (works on SQLite + Postgres). The pgvector table (code_runs)
# is intentionally excluded so the SQLite fallback path never sees a Vector column.
TASK_STORE_TABLES = [Task.__table__, ConversationRef.__table__, TeamRoster.__table__]
TRACKER_DIAGNOSTICS_TABLES = [TrackerDiagnostic.__table__]


__all__ = [
    "Base", "PendingApproval", "CodeRun", "EMBED_DIM",
    "Task", "ConversationRef", "TeamRoster", "TASK_STORE_TABLES",
    "TrackerDiagnostic", "TRACKER_DIAGNOSTICS_TABLES",
]
