"""Tests for SamurAI skill-usage telemetry emit (skill_usage.py)."""

import pytest

import skill_usage


@pytest.fixture(autouse=True)
def _clear_counter():
    skill_usage._counts.clear()
    yield
    skill_usage._counts.clear()


def test_record_counts_valid_names_only():
    skill_usage.record("reading-cloud-logs")
    skill_usage.record("reading-cloud-logs")
    skill_usage.record("github-issue-triage")
    skill_usage.record("Bad Name!")   # invalid → ignored
    snap = skill_usage._snapshot_and_reset()
    assert snap == {"reading-cloud-logs": 2, "github-issue-triage": 1}
    assert skill_usage._counts == {}  # reset after snapshot


def test_emit_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SKILLS_USAGE_ENABLED", raising=False)
    skill_usage.record("reading-cloud-logs")
    assert skill_usage.emit_usage() == {"skipped": True}
    # counts preserved (not snapshotted) when skipped
    assert skill_usage._counts["reading-cloud-logs"] == 1


def test_emit_empty_is_noop(monkeypatch):
    assert skill_usage.emit_usage(force=True) == {"emitted": 0}


def test_emit_files_issue(monkeypatch):
    created = {}

    class FakeRepo:
        def create_issue(self, title, body, labels):
            created["title"] = title
            created["body"] = body
            created["labels"] = labels

    monkeypatch.setattr("tools.github._github", lambda: type("G", (), {"get_repo": lambda self, r: FakeRepo()})())
    skill_usage.record("reading-cloud-logs")
    skill_usage.record("reading-cloud-logs")
    skill_usage.record("github-issue-triage")
    result = skill_usage.emit_usage(force=True)
    assert result == {"emitted": 3, "skills": 2}
    assert created["labels"] == ["skill-usage"]
    assert "<!-- skill-usage -->" in created["body"]
    assert '"skill": "reading-cloud-logs", "count": 2' in created["body"]
    # counter cleared after successful emit
    assert skill_usage._counts == {}


def test_emit_failure_restores_counts(monkeypatch):
    def boom():
        raise RuntimeError("github down")

    monkeypatch.setattr("tools.github._github", boom)
    skill_usage.record("reading-cloud-logs")
    skill_usage.record("reading-cloud-logs")
    result = skill_usage.emit_usage(force=True)
    assert "error" in result
    # counts restored so they're retried next flush
    assert skill_usage._counts["reading-cloud-logs"] == 2


def test_get_skill_records_usage(monkeypatch):
    """skills.get_skill increments the usage counter for a served skill."""
    import skills
    monkeypatch.setattr(skills, "load_skill_catalog",
                        lambda force=False: [{"name": "reading-cloud-logs",
                                              "description": "d", "body": "B", "dir": "x"}])
    skills.get_skill.func("reading-cloud-logs")
    assert skill_usage._counts["reading-cloud-logs"] == 1
