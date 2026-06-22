"""Tests for the agent-side code sandbox tools (tools/code_sandbox.py)."""
from unittest.mock import AsyncMock, patch

import pytest

import tools.code_sandbox as cs


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("SAMURAI_SANDBOX_ENABLED", "SANDBOX_URL", "SANDBOX_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    yield


# ── Enable gate ─────────────────────────────────────────────────────────


async def test_run_code_disabled_by_default():
    out = await cs.run_code.ainvoke(
        {"description": "x", "script": "print(1)"}
    )
    assert "disabled" in out.lower()


# ── Happy path / error path (sandbox HTTP mocked) ──────────────────────────


async def test_run_code_happy_path(monkeypatch):
    monkeypatch.setenv("SAMURAI_SANDBOX_ENABLED", "on")
    fake = {"outcome": "ok", "stdout": "hi\n", "stderr": "", "result": {"n": 2},
            "elapsed_ms": 5, "exit_code": 0}
    with (
        patch.object(cs, "_execute", new=AsyncMock(return_value=fake)),
        patch.object(cs, "_record_run", new=AsyncMock(return_value="abc-123")),
    ):
        out = await cs.run_code.ainvoke(
            {"description": "double", "script": "result={'n':2}"}
        )
    assert "outcome: ok" in out
    assert "hi" in out
    assert "abc-123" in out


async def test_run_code_surfaces_sandbox_error(monkeypatch):
    monkeypatch.setenv("SAMURAI_SANDBOX_ENABLED", "on")
    with patch.object(cs, "_execute", new=AsyncMock(return_value={"error": "boom"})):
        out = await cs.run_code.ainvoke({"description": "x", "script": "print(1)"})
    assert "Sandbox error" in out and "boom" in out


async def test_run_code_does_not_persist_on_sqlite(monkeypatch):
    """Off Postgres, _record_run no-ops (returns None) and the run still works."""
    monkeypatch.setenv("SAMURAI_SANDBOX_ENABLED", "on")
    fake = {"outcome": "ok", "stdout": "x", "stderr": "", "result": None,
            "elapsed_ms": 1, "exit_code": 0}
    with patch.object(cs, "_execute", new=AsyncMock(return_value=fake)):
        # real _record_run -> _is_postgres() False in tests -> returns None
        out = await cs.run_code.ainvoke({"description": "x", "script": "print('x')"})
    assert "outcome: ok" in out
    assert "code_run" not in out  # nothing persisted, so no id mentioned


# ── find_prior_script ──────────────────────────────────────────────────────


async def test_find_prior_script_noop_off_postgres(monkeypatch):
    monkeypatch.setattr(cs, "_is_postgres", lambda: False)
    out = await cs.find_prior_script.ainvoke({"query": "anything"})
    assert "No prior-script library" in out


# ── Helpers ────────────────────────────────────────────────────────────────


def test_inputs_hash_is_deterministic_and_order_insensitive():
    assert cs._inputs_hash({"a": 1, "b": 2}) == cs._inputs_hash({"b": 2, "a": 1})
    assert cs._inputs_hash({"a": 1}) != cs._inputs_hash({"a": 2})


def test_run_code_is_judge_gated():
    """run_code must be a write-class tool so the safety judge reviews scripts."""
    import judge

    assert "run_code" in judge.WRITE_TOOL_NAMES
