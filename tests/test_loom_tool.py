"""Tests for the analyze_loom_video agent tool (tools/loom.py)."""
import pytest

import tools.loom as lt
from kb.ingest_loom import LoomAnalysis

_URL = "https://www.loom.com/share/9614dd0b62e5475985d0b021ee3f33d4"


async def test_rejects_non_loom_url():
    out = await lt.analyze_loom_video.ainvoke({"url": "https://example.com/video"})
    assert "doesn't look like a Loom" in out


async def test_returns_understanding_for_silent_video(monkeypatch):
    res = LoomAnalysis(loom_id="abc", url=_URL, title="Stage bug", duration=12.0,
                       uploader="Rachel", narration_source="none",
                       visual_summary="vs", understanding="WHAT: stage change demo")
    monkeypatch.setattr("kb.ingest_loom.ingest_loom", lambda url, ctx: res)
    out = await lt.analyze_loom_video.ainvoke({"url": _URL})
    assert "Stage bug" in out
    assert "silent screen recording" in out
    assert "WHAT: stage change demo" in out


async def test_includes_narration_when_present(monkeypatch):
    res = LoomAnalysis(loom_id="abc", url=_URL, title="T", duration=5.0,
                       narration_source="transcribed", narration="the bug is here",
                       understanding="note")
    monkeypatch.setattr("kb.ingest_loom.ingest_loom", lambda url, ctx: res)
    out = await lt.analyze_loom_video.ainvoke({"url": _URL})
    assert "Narration (transcribed): the bug is here" in out


async def test_handles_failure_gracefully(monkeypatch):
    def _boom(url, ctx):
        raise RuntimeError("yt-dlp not found")

    monkeypatch.setattr("kb.ingest_loom.ingest_loom", _boom)
    out = await lt.analyze_loom_video.ainvoke({"url": _URL})
    assert "Could not analyze" in out and "yt-dlp not found" in out


def test_loom_tool_is_read_safe():
    """The tool is read-only (no external mutation) — must not be a write tool."""
    import judge

    assert "analyze_loom_video" not in judge.WRITE_TOOL_NAMES
