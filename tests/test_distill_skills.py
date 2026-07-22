"""Tests for the in-boundary skill distiller (kb/distill_skills.py).

Fully hermetic: Vertex Gemini, GCS, and GitHub are mocked; the sanitizer runs for real.
"""

import json

import pytest

from kb import distill_skills as ds


@pytest.fixture
def bucket(monkeypatch):
    store: dict[str, str] = {}
    monkeypatch.setattr(ds.storage, "read_text", lambda p: store.get(p))
    monkeypatch.setattr(ds.storage, "write_text", lambda p, c, **k: store.__setitem__(p, c))
    monkeypatch.setattr(ds.storage, "list_text", lambda prefix: [
        (k, v) for k, v in store.items() if k.startswith(prefix) and k.endswith(".md")])
    monkeypatch.setattr(ds.storage, "list_paths", lambda prefix: [k for k in store if k.startswith(prefix)])
    monkeypatch.setattr(ds.storage, "acquire_lock", lambda p, ttl_seconds=1800: True)
    monkeypatch.setattr(ds.storage, "release_lock", lambda p: None)
    # neutralize the Vertex client + GitHub dedup/delivery
    monkeypatch.setattr(ds, "get_kb_llm", lambda: object())
    monkeypatch.setattr(ds, "_existing_skill_names_descriptions", lambda: [])
    filed: list = []
    monkeypatch.setattr(ds, "_file_draft_issue",
                        lambda name, md, reason: filed.append(name) or f"https://x/{name}")
    ds._filed = filed
    return store


def _mock_llm(monkeypatch, distill_json, sanitize_clean=True, sanitize_reason="ok"):
    def fake(llm, system, user):
        if system is ds._DISTILL_SYS:
            return json.dumps(distill_json)
        if system is ds._SANITIZE_SYS:
            return json.dumps({"clean": sanitize_clean, "reason": sanitize_reason})
        return "{}"
    monkeypatch.setattr(ds, "_llm_text", fake)


def _convos(monkeypatch, mapping):
    monkeypatch.setattr(ds, "_load_recent_conversations", lambda: mapping)


LESSON_TURN = {
    "ts": "2026-07-22T10:00:00Z", "conversation_id": "c1",
    "user_message": "why is the bulk clone slow?",
    "assistant_response": "it was an N+1 per-row flush; batched it",
    "tools": ["github_get_commit", "run_code"],
}
TRIVIAL_TURN = {
    "ts": "2026-07-22T10:00:00Z", "conversation_id": "c2",
    "user_message": "hi", "assistant_response": "hello!", "tools": [],
}

CLEAN_SKILL = {
    "worth_capturing": True,
    "name": "n-plus-one-batch-flush-fix",
    "description": "When a bulk write does a per-row flush (N+1), batch into one Core insert.",
    "body": "Replace add_all+flush with a single multi-row insert; assert row count.",
    "reason": "recurring pattern",
}


def test_kill_switch_off(monkeypatch, bucket):
    monkeypatch.delenv("SKILLS_DISTILL_ENABLED", raising=False)
    assert ds.run_skill_distill() == {"skipped": True}


def test_distills_clean_skill_and_files_issue(monkeypatch, bucket):
    _convos(monkeypatch, {"c1": [LESSON_TURN]})
    _mock_llm(monkeypatch, CLEAN_SKILL)
    stats = ds.run_skill_distill(force=True)
    assert stats["distilled"] == 1
    assert stats["filed"] == 1
    assert ds._filed == ["n-plus-one-batch-flush-fix"]
    assert "support/skills-drafts/n-plus-one-batch-flush-fix.md" in bucket
    # an audit record was written
    assert any(k.startswith("support/skills-drafts/.audit/") for k in bucket)


def test_trivial_conversation_skipped(monkeypatch, bucket):
    _convos(monkeypatch, {"c2": [TRIVIAL_TURN]})
    _mock_llm(monkeypatch, CLEAN_SKILL)  # even if LLM would produce one, filter skips first
    stats = ds.run_skill_distill(force=True)
    assert stats["distilled"] == 0
    assert stats["conversations"] == 0  # filtered before the LLM


def test_deterministic_gate_blocks_secret(monkeypatch, bucket):
    dirty = dict(CLEAN_SKILL)
    dirty["body"] = "set api_key = 'sk-supersecret12345' then retry"
    _convos(monkeypatch, {"c1": [LESSON_TURN]})
    _mock_llm(monkeypatch, dirty)
    stats = ds.run_skill_distill(force=True)
    assert stats["det_blocked"] == 1
    assert stats["distilled"] == 0
    assert ds._filed == []


def test_llm_sanitize_veto(monkeypatch, bucket):
    _convos(monkeypatch, {"c1": [LESSON_TURN]})
    _mock_llm(monkeypatch, CLEAN_SKILL, sanitize_clean=False, sanitize_reason="tenant-ish")
    stats = ds.run_skill_distill(force=True)
    assert stats["llm_blocked"] == 1
    assert stats["distilled"] == 0


def test_duplicate_skipped(monkeypatch, bucket):
    _convos(monkeypatch, {"c1": [LESSON_TURN]})
    _mock_llm(monkeypatch, CLEAN_SKILL)
    monkeypatch.setattr(ds, "_existing_skill_names_descriptions",
                        lambda: [("n-plus-one-batch-flush-fix", "already have it")])
    stats = ds.run_skill_distill(force=True)
    assert stats["duplicate"] == 1
    assert stats["distilled"] == 0


def test_not_worth_capturing(monkeypatch, bucket):
    _convos(monkeypatch, {"c1": [LESSON_TURN]})
    _mock_llm(monkeypatch, {"worth_capturing": False, "reason": "one-off"})
    stats = ds.run_skill_distill(force=True)
    assert stats["not_worth"] == 1
    assert stats["distilled"] == 0


def test_idempotent_second_run(monkeypatch, bucket):
    _convos(monkeypatch, {"c1": [LESSON_TURN]})
    _mock_llm(monkeypatch, CLEAN_SKILL)
    ds.run_skill_distill(force=True)
    ds._filed.clear()
    # second run: same rollup hash in manifest → conversation not reprocessed
    stats = ds.run_skill_distill(force=True)
    assert stats["conversations"] == 0
    assert stats["distilled"] == 0
    assert ds._filed == []


def test_lock_contention(monkeypatch, bucket):
    monkeypatch.setattr(ds.storage, "acquire_lock", lambda p, ttl_seconds=1800: False)
    _convos(monkeypatch, {"c1": [LESSON_TURN]})
    _mock_llm(monkeypatch, CLEAN_SKILL)
    stats = ds.run_skill_distill(force=True)
    assert stats == {"skipped": True, "reason": "locked"}
