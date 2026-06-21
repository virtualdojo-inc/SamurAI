"""Tests for the secured admin endpoint (admin.py)."""
import json
from unittest.mock import AsyncMock, patch

import pytest

import admin


class _Req:
    """Minimal stand-in for aiohttp.web.Request for handle_admin."""

    def __init__(self, headers=None, body=None, remote="1.2.3.4"):
        self.headers = headers or {}
        self._body = body
        self.remote = remote

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    admin._rate.clear()
    monkeypatch.delenv("SAMURAI_ADMIN_KEY", raising=False)
    yield
    admin._rate.clear()


def _body(resp):
    return json.loads(resp.body)


# ── Fail-closed + auth ─────────────────────────────────────────────────


async def test_disabled_when_no_key():
    """No SAMURAI_ADMIN_KEY → endpoint is disabled (404), even with a token."""
    resp = await admin.handle_admin(
        _Req(headers={"Authorization": "Bearer anything"}, body={"op": "ping"})
    )
    assert resp.status == 404


async def test_unauthorized_wrong_token(monkeypatch):
    monkeypatch.setenv("SAMURAI_ADMIN_KEY", "supersecretkey")
    resp = await admin.handle_admin(
        _Req(headers={"Authorization": "Bearer wrong"}, body={"op": "ping"})
    )
    assert resp.status == 401


async def test_unauthorized_missing_header(monkeypatch):
    monkeypatch.setenv("SAMURAI_ADMIN_KEY", "k")
    resp = await admin.handle_admin(_Req(body={"op": "ping"}))
    assert resp.status == 401


async def test_authorized_ping(monkeypatch):
    monkeypatch.setenv("SAMURAI_ADMIN_KEY", "k")
    resp = await admin.handle_admin(
        _Req(headers={"Authorization": "Bearer k"}, body={"op": "ping"})
    )
    assert resp.status == 200
    assert _body(resp)["result"]["ok"] is True


# ── Op allowlist ────────────────────────────────────────────────────────


async def test_unknown_op_rejected(monkeypatch):
    monkeypatch.setenv("SAMURAI_ADMIN_KEY", "k")
    resp = await admin.handle_admin(
        _Req(headers={"Authorization": "Bearer k"}, body={"op": "rm_rf_slash"})
    )
    assert resp.status == 400
    assert set(_body(resp)["allowed"]) == {"ping", "db_query", "logs", "migrate_data", "chat"}


async def test_chat_op_runs_agent(monkeypatch):
    monkeypatch.setenv("SAMURAI_ADMIN_KEY", "k")
    with patch("agent.run_agent", new_callable=AsyncMock, return_value="pong") as ra:
        resp = await admin.handle_admin(
            _Req(headers={"Authorization": "Bearer k"},
                 body={"op": "chat", "args": {"message": "hi there"}})
        )
    assert resp.status == 200
    assert _body(resp)["result"]["reply"] == "pong"
    assert ra.call_args.kwargs["user_message"] == "hi there"


async def test_chat_requires_message(monkeypatch):
    monkeypatch.setenv("SAMURAI_ADMIN_KEY", "k")
    resp = await admin.handle_admin(
        _Req(headers={"Authorization": "Bearer k"}, body={"op": "chat", "args": {}})
    )
    assert "error" in _body(resp)["result"]


# ── Read-only SQL guard (no DB needed — guard rejects before connecting) ──


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO tasks VALUES (1)",
        "UPDATE tasks SET x=1",
        "DELETE FROM tasks",
        "DROP TABLE tasks",
        "ALTER TABLE tasks ADD COLUMN c int",
        "TRUNCATE tasks",
        "GRANT ALL ON tasks TO public",
        "SELECT 1; DROP TABLE tasks",   # stacked statement
        "",                              # empty
        "EXPLAIN ANALYZE DELETE FROM x", # not a bare SELECT/WITH
    ],
)
async def test_db_query_rejects_non_readonly(sql):
    out = await admin._op_db_query({"sql": sql})
    assert "error" in out


async def test_db_query_accepts_select_shape():
    """A bare SELECT passes the guard (it would then hit the DB — not exercised here)."""
    from admin import _READ_SQL, _FORBIDDEN_SQL

    sql = "SELECT count(*) FROM pending_approvals"
    assert _READ_SQL.match(sql) and not _FORBIDDEN_SQL.search(sql) and ";" not in sql


# ── Rate limiting ─────────────────────────────────────────────────────────


async def test_rate_limit(monkeypatch):
    monkeypatch.setenv("SAMURAI_ADMIN_KEY", "k")
    last = None
    for _ in range(admin._RATE_LIMIT_PER_MIN + 5):
        last = await admin.handle_admin(
            _Req(headers={"Authorization": "Bearer k"}, body={"op": "ping"}, remote="9.9.9.9")
        )
    assert last.status == 429
