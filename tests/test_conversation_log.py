"""Tests for durable per-turn conversation capture (conversation_log.py)."""

import json
from datetime import datetime, timezone

import conversation_log as clog


def test_log_turn_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(clog, "RAW_DIR", tmp_path / "raw")
    ts = datetime(2026, 5, 29, 13, 45, 1, tzinfo=timezone.utc)
    path = clog.log_turn(
        conversation_id="conv-1",
        user_id="user-1",
        user_name="Devin",
        user_email="devin@virtualdojo.com",
        user_message="why is the bot down?",
        assistant_response="it isn't — health is green",
        tools=["query_cloud_logs", "get_skill"],
        is_background_task=False,
        ts=ts,
    )
    assert path is not None
    # Partitioned by date.
    assert "/raw/2026-05-29/" in path.replace("\\", "/")
    rec = json.loads((tmp_path / "raw" / "2026-05-29").glob("*.json").__next__().read_text())
    assert rec["conversation_id"] == "conv-1"
    assert rec["user_message"] == "why is the bot down?"
    assert rec["assistant_response"] == "it isn't — health is green"
    assert rec["tools"] == ["query_cloud_logs", "get_skill"]
    assert rec["ts"] == ts.isoformat()


def test_log_turn_one_file_per_call(tmp_path, monkeypatch):
    monkeypatch.setattr(clog, "RAW_DIR", tmp_path / "raw")
    ts = datetime(2026, 5, 29, 9, 0, 0, tzinfo=timezone.utc)
    for i in range(3):
        clog.log_turn(
            conversation_id=f"c{i}",
            user_id="u",
            user_message="hi",
            assistant_response="hello",
            ts=ts,
        )
    files = list((tmp_path / "raw" / "2026-05-29").glob("*.json"))
    assert len(files) == 3  # no shared-file appends


def test_log_turn_swallows_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(clog, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(
        clog.json, "dumps", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom"))
    )
    # Must not raise — capture can never break a turn.
    assert clog.log_turn(
        conversation_id="c", user_id="u", user_message="x", assistant_response="y"
    ) is None
