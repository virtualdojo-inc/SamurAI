"""Tests for the skill-catalog evaluation (kb/evaluate_skills.py)."""

import pytest

from kb import evaluate_skills as ev


CATALOG = [
    ("reading-cloud-logs", "Read Google Cloud logs fast with tight filters and field projection"),
    ("github-issue-triage", "Triage bugs: check duplicates, file with the right issue type"),
    ("cloud-log-reader", "Read Google Cloud logs fast with tight filters and field projection"),  # dup of #1
    ("never-used-skill", "Some niche procedure nobody has invoked yet"),
]
USAGE = {"reading-cloud-logs": 12, "github-issue-triage": 3}


def test_evaluate_flags_dead_valuable_duplicate(monkeypatch):
    monkeypatch.setattr(ev, "VALUABLE_THRESHOLD", 10)
    result = ev.evaluate(CATALOG, USAGE)
    assert result["total"] == 4
    # dead = catalog names with 0 usage
    assert set(result["dead"]) == {"cloud-log-reader", "never-used-skill"}
    # valuable = >= threshold
    assert result["valuable"] == [("reading-cloud-logs", 12)]
    # duplicate = the two identical-description log readers
    pairs = {frozenset((a, b)) for a, b, _ in result["duplicates"]}
    assert frozenset(("reading-cloud-logs", "cloud-log-reader")) in pairs


def test_report_md_contains_sections(monkeypatch):
    from datetime import datetime, timezone
    result = ev.evaluate(CATALOG, USAGE)
    md = ev._report_md(result, datetime(2026, 7, 22, tzinfo=timezone.utc))
    assert "Skill catalog evaluation — 2026-07-22" in md
    assert "never-used-skill" in md          # dead listed
    assert "reading-cloud-logs` — 12" in md   # valuable listed
    assert "Dead" in md and "Valuable" in md and "duplicates" in md.lower()


def test_usage_leaderboard_parse():
    lb = (
        "# Skill usage leaderboard\n\n"
        "| Rank | Skill | Invocations |\n|---:|---|---:|\n"
        "| 1 | `reading-cloud-logs` | 12 |\n"
        "| 2 | `github-issue-triage` | 3 |\n"
    )
    import re
    counts = {}
    for m in re.finditer(r"\|\s*\d+\s*\|\s*`([a-z0-9-]+)`\s*\|\s*(\d+)\s*\|", lb):
        counts[m.group(1)] = int(m.group(2))
    assert counts == {"reading-cloud-logs": 12, "github-issue-triage": 3}


def test_run_gated_off(monkeypatch):
    monkeypatch.delenv("SKILLS_EVAL_ENABLED", raising=False)
    assert ev.run_skill_evaluation() == {"skipped": True}


def test_run_files_report_issue(monkeypatch):
    monkeypatch.setattr(ev.storage, "acquire_lock", lambda p, ttl_seconds=600: True)
    monkeypatch.setattr(ev.storage, "release_lock", lambda p: None)
    monkeypatch.setattr(ev, "_fetch_catalog", lambda: CATALOG)
    monkeypatch.setattr(ev, "_fetch_usage", lambda: USAGE)
    created = {}

    class FakeRepo:
        def create_issue(self, title, body, labels):
            created.update(title=title, body=body, labels=labels)

    monkeypatch.setattr("tools.github._github",
                        lambda: type("G", (), {"get_repo": lambda self, r: FakeRepo()})())
    stats = ev.run_skill_evaluation(force=True)
    assert stats["total"] == 4
    assert stats["dead"] == 2
    assert stats["valuable"] == 1
    assert stats["reported"] is True
    assert created["labels"] == ["skill-eval"]


def test_run_locked(monkeypatch):
    monkeypatch.setattr(ev.storage, "acquire_lock", lambda p, ttl_seconds=600: False)
    assert ev.run_skill_evaluation(force=True) == {"skipped": True, "reason": "locked"}
