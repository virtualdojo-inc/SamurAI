"""Tests for the tracker-triage worker (trailer parsing + batch logic)."""

import pytest

import tracker_diagnostics as td
import tracker_triage as tt


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(td, "TASK_DB_PATH", str(tmp_path / "tasks.sqlite"))
    td._store = None
    yield
    td._store = None


@pytest.fixture(autouse=True)
def _stub_clear_thread(monkeypatch):
    """Batch runs now clear per-row checkpoints via memory.clear_thread; stub it
    so tests that don't inject their own never touch the real checkpointer."""
    import memory
    from unittest.mock import AsyncMock

    monkeypatch.setattr(memory, "clear_thread", AsyncMock(return_value=True))


# ── trailer parsing ───────────────────────────────────────────────────────

def test_parse_full_trailer():
    text = (
        "Diagnosis body...\n\n"
        "CATEGORY: D\nSUGGESTED_TYPE: Bug\nSUGGESTED_PRIORITY: P1\n"
        "SUMMARY: NOT NULL violation on quotes.total; add a default.\n"
    )
    p = tt._parse_trailer(text)
    assert p["category"] == "D"
    assert p["suggested_type"] == "Bug"
    assert p["suggested_priority"] == "P1"
    assert "NOT NULL" in p["summary"]


def test_parse_none_tokens_become_null():
    text = "CATEGORY: A\nSUGGESTED_TYPE: none\nSUGGESTED_PRIORITY: n/a\nSUMMARY: tenant tweak"
    p = tt._parse_trailer(text)
    assert p["category"] == "A"
    assert p["suggested_type"] is None
    assert p["suggested_priority"] is None


def test_parse_missing_or_bad_category_is_unknown():
    assert tt._parse_trailer("no trailer here")["category"] == "unknown"
    assert tt._parse_trailer("CATEGORY: Z")["category"] == "unknown"


def test_triage_enabled_env(monkeypatch):
    monkeypatch.delenv("TRACKER_TRIAGE_ENABLED", raising=False)
    assert tt.triage_enabled() is False
    monkeypatch.setenv("TRACKER_TRIAGE_ENABLED", "on")
    assert tt.triage_enabled() is True
    monkeypatch.setenv("TRACKER_TRIAGE_ENABLED", "false")
    assert tt.triage_enabled() is False


# ── batch logic (injected agent + sheet, no live deps) ─────────────────────

def _sheet(rows):
    return {"name": "DH Tech", "total_rows": len(rows), "columns": [], "rows": rows}


def _fake_agent_factory(reply="CATEGORY: D\nSUGGESTED_TYPE: Bug\nSUGGESTED_PRIORITY: P2\nSUMMARY: x"):
    calls = []

    async def fake_run_agent(*, user_message, conversation_id, is_background_task):
        calls.append(user_message)
        return reply

    return fake_run_agent, calls


async def test_batch_diagnoses_new_rows_and_skips_unchanged():
    rows = [
        {"Symptom": "boom A", "Github Issue No": "692", "_row_id": "111"},
        {"Symptom": "boom B", "_row_id": "222"},
    ]

    async def fetch():
        return _sheet(rows)

    agent, calls = _fake_agent_factory()
    res = await tt.run_triage_batch(run_agent=agent, fetch_rows=fetch)
    assert res["diagnosed"] == 2
    assert res["skipped"] == 0
    assert len(calls) == 2

    # The github issue number was extracted and stored.
    store = await td.get_diagnostics_store()
    rec = await store.get("111")
    assert rec["github_issue_no"] == "692"
    assert rec["category"] == "D"

    # Second run, same rows → nothing re-diagnosed.
    agent2, calls2 = _fake_agent_factory()
    res2 = await tt.run_triage_batch(run_agent=agent2, fetch_rows=fetch)
    assert res2["diagnosed"] == 0
    assert res2["candidates"] == 0
    assert len(calls2) == 0


async def test_batch_redoes_changed_row():
    rows = [{"Symptom": "boom", "_row_id": "111"}]

    async def fetch():
        return _sheet(rows)

    agent, _ = _fake_agent_factory()
    await tt.run_triage_batch(run_agent=agent, fetch_rows=fetch)

    rows[0]["Symptom"] = "different boom"  # content changed → re-diagnose
    agent2, calls2 = _fake_agent_factory()
    res = await tt.run_triage_batch(run_agent=agent2, fetch_rows=fetch)
    assert res["diagnosed"] == 1
    assert len(calls2) == 1


async def test_batch_cap_limits_work_and_reports_remaining():
    rows = [{"Symptom": f"s{i}", "_row_id": str(i)} for i in range(5)]

    async def fetch():
        return _sheet(rows)

    agent, calls = _fake_agent_factory()
    res = await tt.run_triage_batch(run_agent=agent, fetch_rows=fetch, max_rows=2)
    assert res["diagnosed"] == 2
    assert res["candidates"] == 5
    assert res["remaining"] == 3
    assert len(calls) == 2


async def test_row_without_id_is_ignored():
    rows = [{"Symptom": "no id row"}]  # missing _row_id

    async def fetch():
        return _sheet(rows)

    agent, calls = _fake_agent_factory()
    res = await tt.run_triage_batch(run_agent=agent, fetch_rows=fetch)
    assert res["diagnosed"] == 0
    assert len(calls) == 0


async def test_each_row_uses_isolated_thread_and_purges_legacy():
    """Regression: triage must NOT reuse one shared conversation_id (that grew
    the checkpoint to thousands of messages). Each row gets its own thread which
    is cleared, and the legacy shared thread is purged once."""
    rows = [
        {"Symptom": "a", "_row_id": "111"},
        {"Symptom": "b", "_row_id": "222"},
    ]

    async def fetch():
        return _sheet(rows)

    convs = []

    async def agent(*, user_message, conversation_id, is_background_task):
        convs.append(conversation_id)
        return "CATEGORY: A\nSUGGESTED_TYPE: none\nSUGGESTED_PRIORITY: none\nSUMMARY: ok"

    cleared = []

    async def clear(cid):
        cleared.append(cid)
        return True

    await tt.run_triage_batch(run_agent=agent, fetch_rows=fetch, clear_thread=clear)

    # Each row diagnosed in its OWN thread — never the bare shared id.
    assert convs == ["tracker_triage:111", "tracker_triage:222"]
    # Legacy shared thread purged once, and every per-row thread cleared.
    assert "tracker_triage" in cleared          # one-time legacy cleanup
    assert "tracker_triage:111" in cleared
    assert "tracker_triage:222" in cleared


async def test_failed_row_still_clears_its_thread():
    """Even a row whose agent call raises must have its ephemeral thread cleared
    (the clear runs in a finally)."""
    rows = [{"Symptom": "bad", "_row_id": "9"}]

    async def fetch():
        return _sheet(rows)

    async def boom(*, user_message, conversation_id, is_background_task):
        raise RuntimeError("model blew up")

    cleared = []

    async def clear(cid):
        cleared.append(cid)
        return True

    res = await tt.run_triage_batch(run_agent=boom, fetch_rows=fetch, clear_thread=clear)
    assert res["diagnosed"] == 0
    assert "tracker_triage:9" in cleared  # cleared despite the failure


async def test_one_failing_row_does_not_stall_batch():
    rows = [
        {"Symptom": "good", "_row_id": "1"},
        {"Symptom": "bad", "_row_id": "2"},
    ]

    async def fetch():
        return _sheet(rows)

    async def flaky_agent(*, user_message, conversation_id, is_background_task):
        if "bad" in user_message:
            raise RuntimeError("model blew up")
        return "CATEGORY: A\nSUGGESTED_TYPE: none\nSUGGESTED_PRIORITY: none\nSUMMARY: ok"

    res = await tt.run_triage_batch(run_agent=flaky_agent, fetch_rows=fetch)
    assert res["diagnosed"] == 1  # the good one still stored
    store = await td.get_diagnostics_store()
    assert await store.get("1") is not None
    assert await store.get("2") is None  # failed row not stored → retried next tick
