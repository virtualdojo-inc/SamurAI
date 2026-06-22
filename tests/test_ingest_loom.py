"""Tests for kb/ingest_loom.py — Loom audio+visual ingestion.

Network/LLM/ffmpeg boundaries are mocked; these verify the orchestration logic
(transcript-vs-transcribe-vs-silent selection), the in-boundary guardrail, and
the part builders. The live end-to-end run is the integration verification.
"""
import base64
import os

import pytest

import kb.ingest_loom as il


# ── URL parsing ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("url,expected", [
    ("https://www.loom.com/share/9614dd0b62e5475985d0b021ee3f33d4", "9614dd0b62e5475985d0b021ee3f33d4"),
    ("loom.com/share/abcdef0123456789abcdef", "abcdef0123456789abcdef"),
    ("https://example.com/not-a-loom", None),
    ("", None),
])
def test_loom_id_from_url(url, expected):
    assert il.loom_id_from_url(url) == expected


# ── part builders ───────────────────────────────────────────────────────


def test_text_part():
    assert il._text("hi") == {"text": "hi"}


def test_inline_part_base64s_file(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"\xff\xd8hello")
    part = il._inline(str(p), "image/jpeg")
    assert part["inlineData"]["mimeType"] == "image/jpeg"
    assert base64.b64decode(part["inlineData"]["data"]) == b"\xff\xd8hello"


# ── in-boundary guardrail ─────────────────────────────────────────────────


def test_vertex_refuses_global_endpoint(monkeypatch):
    monkeypatch.setattr(il, "KB_VERTEX_LOCATION", "global")
    with pytest.raises(RuntimeError, match="regional"):
        il._vertex_multimodal([il._text("x")])


# ── orchestration: which narration source wins ────────────────────────────


def _stub_analysis(**kw):
    a = il.LoomAnalysis(loom_id="abc", url="u", duration=12.0, **kw)
    return a


@pytest.fixture
def _mock_visual(monkeypatch):
    """Stub out the visual path + fuse so only audio logic is exercised."""
    monkeypatch.setattr(il, "extract_keyframes", lambda *a, **k: ["f1.jpg"])
    monkeypatch.setattr(il, "analyze_frames", lambda *a, **k: "visual summary")
    monkeypatch.setattr(il, "_fuse", lambda *a, **k: "FUSED NOTE")


def test_uses_loom_transcript_when_present(monkeypatch, _mock_visual):
    a = _stub_analysis(narration="loom says hi", narration_source="loom_transcript")
    a._video = "v.mp4"
    a._audio = None
    monkeypatch.setattr(il, "download_loom", lambda url, wd: a)
    called = {"transcribe": False}
    monkeypatch.setattr(il, "transcribe_audio", lambda p: called.__setitem__("transcribe", True) or "x")

    res = il.ingest_loom("u")
    assert res.narration_source == "loom_transcript"
    assert res.narration == "loom says hi"
    assert called["transcribe"] is False  # not re-transcribed
    assert res.visual_summary == "visual summary"
    assert res.understanding == "FUSED NOTE"


def test_transcribes_when_silent_loom_but_audio_has_speech(monkeypatch, _mock_visual):
    a = _stub_analysis(narration="", narration_source="none")
    a._video = "v.mp4"
    a._audio = "a.mp3"
    monkeypatch.setattr(il, "download_loom", lambda url, wd: a)
    monkeypatch.setattr(il, "_mean_volume_db", lambda p: -20.0)  # loud -> has speech
    monkeypatch.setattr(il, "transcribe_audio", lambda p: "spoken narration here")

    res = il.ingest_loom("u")
    assert res.narration_source == "transcribed"
    assert res.narration == "spoken narration here"


def test_skips_transcription_when_track_is_silent(monkeypatch, _mock_visual):
    a = _stub_analysis(narration="", narration_source="none")
    a._video = "v.mp4"
    a._audio = "a.mp3"
    monkeypatch.setattr(il, "download_loom", lambda url, wd: a)
    monkeypatch.setattr(il, "_mean_volume_db", lambda p: -55.0)  # below threshold
    called = {"transcribe": False}
    monkeypatch.setattr(il, "transcribe_audio", lambda p: called.__setitem__("transcribe", True) or "x")

    res = il.ingest_loom("u")
    assert res.narration_source == "none"
    assert res.narration == ""
    assert called["transcribe"] is False
    # visual path still runs for silent demos
    assert res.visual_summary == "visual summary"


def test_engine_provenance_is_in_boundary(monkeypatch, _mock_visual):
    a = _stub_analysis(narration_source="none")
    a._video = None
    a._audio = None
    monkeypatch.setattr(il, "download_loom", lambda url, wd: a)
    res = il.ingest_loom("u")
    assert res.engine["external_llm"] is False
    assert res.engine["engine"] == "vertex-gemini"
