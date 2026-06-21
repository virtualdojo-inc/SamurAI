"""Offline tests for the db/ models + session scaffold (no DB connection)."""
import pytest

import db.models as m
import db.session as s


def test_metadata_has_expected_tables():
    tables = set(m.Base.metadata.tables)
    assert "pending_approvals" in tables
    assert "code_runs" in tables


def test_pending_approval_columns():
    cols = m.PendingApproval.__table__.columns
    assert cols["request_id"].primary_key
    for c in (
        "action_type", "conversation_id", "payload_json", "payload_sha256",
        "status", "requested_by", "approver_email", "activity_id",
        "revise_note", "created_at", "expires_at", "decided_at",
    ):
        assert c in cols, f"missing column {c}"


def test_code_run_has_vector_embedding():
    from pgvector.sqlalchemy import Vector

    cols = m.CodeRun.__table__.columns
    assert "script" in cols
    assert isinstance(cols["embedding"].type, Vector)


def test_init_engine_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    s._engine = None
    s._sessionmaker = None
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        s.init_engine()
