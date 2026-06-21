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

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Index, String, Text, func
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


__all__ = ["Base", "PendingApproval", "CodeRun", "EMBED_DIM"]
