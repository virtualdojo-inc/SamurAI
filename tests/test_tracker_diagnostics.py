"""Tests for the tracker-diagnostics store, hash, index, and serving tool."""

import pytest

import tracker_diagnostics as td


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    """Point the store + sync index at a throwaway DB; reset the singleton."""
    db = str(tmp_path / "tasks.sqlite")
    monkeypatch.setattr(td, "TASK_DB_PATH", db)
    td._store = None
    yield
    td._store = None


# ── row_content_hash ────────────────────────────────────────────────────

def test_hash_ignores_internal_fields():
    a = {"Symptom": "boom", "_row_id": "111", "_row_number": 1}
    b = {"Symptom": "boom", "_row_id": "999", "_row_number": 7}
    assert td.row_content_hash(a) == td.row_content_hash(b)


def test_hash_is_order_independent_but_content_sensitive():
    a = {"Symptom": "boom", "Priority": "High"}
    b = {"Priority": "High", "Symptom": "boom"}
    assert td.row_content_hash(a) == td.row_content_hash(b)
    c = {"Symptom": "different", "Priority": "High"}
    assert td.row_content_hash(a) != td.row_content_hash(c)


# ── store CRUD + needs_diagnosis ──────────────────────────────────────────

async def test_needs_diagnosis_lifecycle():
    store = await td.get_diagnostics_store()
    # New row → needs work.
    assert await store.needs_diagnosis("111", "hashA") is True

    await store.upsert_diagnosis(
        row_id="111", sheet_id=td.DH_TECH_TRACKER_SHEET_ID,
        row_hash="hashA", diagnosis="full text", summary="s",
        category="D", suggested_type="Bug", suggested_priority="P2",
        github_issue_no="692",
    )
    # Same hash → already diagnosed.
    assert await store.needs_diagnosis("111", "hashA") is False
    # Changed content → re-diagnose.
    assert await store.needs_diagnosis("111", "hashB") is True

    # Stale → re-diagnose even with same hash.
    await store.mark_stale("111")
    assert await store.needs_diagnosis("111", "hashA") is True


async def test_upsert_overwrites_and_clears_stale():
    store = await td.get_diagnostics_store()
    await store.upsert_diagnosis(
        row_id="1", sheet_id="s", row_hash="h1", diagnosis="v1", category="A",
    )
    await store.mark_stale("1")
    await store.upsert_diagnosis(
        row_id="1", sheet_id="s", row_hash="h2", diagnosis="v2", category="B",
    )
    rec = await store.get("1")
    assert rec["diagnosis"] == "v2"
    assert rec["row_hash"] == "h2"
    assert rec["category"] == "B"
    assert rec["status"] == "diagnosed"


async def test_list_ready_filters():
    store = await td.get_diagnostics_store()
    await store.upsert_diagnosis(row_id="1", sheet_id="s", row_hash="h", diagnosis="d1", category="A")
    await store.upsert_diagnosis(row_id="2", sheet_id="s", row_hash="h", diagnosis="d2", category="D", github_issue_no="692")
    await store.upsert_diagnosis(row_id="3", sheet_id="s", row_hash="h", diagnosis="d3", category="D", github_issue_no="700")
    await store.mark_stale("3")  # stale items are excluded

    assert len(await store.list_ready()) == 2
    assert len(await store.list_ready(category="D")) == 1  # only #692; #700 is stale
    by_issue = await store.list_ready(github_issue_no="692")
    assert len(by_issue) == 1 and by_issue[0]["row_id"] == "2"


# ── prompt index (sync) ───────────────────────────────────────────────────

def test_index_empty_when_no_diagnoses():
    assert td.tracker_diagnostics_index_text() == ""


async def test_index_reports_count_after_upsert():
    store = await td.get_diagnostics_store()
    await store.upsert_diagnosis(row_id="1", sheet_id="s", row_hash="h", diagnosis="d", category="A")
    text = td.tracker_diagnostics_index_text()
    assert "DH Tech Issue Tracker" in text
    assert "1 tracker item" in text
    assert "get_tracker_diagnostics" in text


def test_index_never_raises_on_missing_db(monkeypatch):
    monkeypatch.setattr(td, "TASK_DB_PATH", "/nonexistent/dir/nope.sqlite")
    assert td.tracker_diagnostics_index_text() == ""


# ── serving tool ──────────────────────────────────────────────────────────

async def test_tool_empty_state():
    out = await td.get_tracker_diagnostics.ainvoke({})
    assert "No prepared tracker diagnoses" in out


async def test_tool_list_and_detail():
    store = await td.get_diagnostics_store()
    await store.upsert_diagnosis(
        row_id="2", sheet_id="s", row_hash="h", diagnosis="FULL DIAGNOSIS BODY",
        summary="null deref in quoting", category="D",
        suggested_type="Bug", suggested_priority="P1", github_issue_no="692",
    )
    listed = await td.get_tracker_diagnostics.ainvoke({})
    assert "Prepared tracker diagnoses" in listed
    assert "null deref in quoting" in listed
    assert "FULL DIAGNOSIS BODY" not in listed  # list view stays compact

    detail = await td.get_tracker_diagnostics.ainvoke({"github_issue_no": "692"})
    assert "FULL DIAGNOSIS BODY" in detail
    assert "GitHub #692" in detail


async def test_tool_unknown_issue():
    out = await td.get_tracker_diagnostics.ainvoke({"github_issue_no": "404"})
    assert "No prepared diagnosis for GitHub #404" in out
