"""Tests for the proactive Loom-ingest path in kb/ingest_smartsheet.py.

Covers Loom URL detection on rows, the authoritative:false companion note,
dedup, failure isolation, and the enable gate. The Vertex analysis + GCS writes
are mocked.
"""
import pytest

import kb.ingest_smartsheet as ism
from kb.ingest_loom import LoomAnalysis

_URL1 = "https://www.loom.com/share/9614dd0b62e5475985d0b021ee3f33d4"
_URL2 = "https://www.loom.com/share/abcdef0123456789abcdef0123456789"


def _row(cells):
    return {"id": 555, "rowNumber": 3, "cells": cells}


# ── detection ──────────────────────────────────────────────────────────


def test_extract_loom_urls_from_text_and_hyperlink():
    row = _row([
        {"displayValue": f"repro here: {_URL1} thanks"},
        {"hyperlink": {"url": _URL2}},
        {"displayValue": "no link in this cell"},
    ])
    assert ism._extract_loom_urls(row) == sorted([_URL1, _URL2])


def test_extract_loom_urls_ignores_non_loom():
    row = _row([
        {"displayValue": "https://example.com/share/whatever"},
        {"hyperlink": {"url": "https://youtube.com/watch?v=x"}},
    ])
    assert ism._extract_loom_urls(row) == []


# ── note shape ──────────────────────────────────────────────────────────


def test_loom_note_is_marked_non_authoritative():
    a = LoomAnalysis(loom_id="9614dd0b62e5475985d0b021ee3f33d4", url=_URL1,
                     title="Stage bug", duration=12.0, narration_source="none",
                     understanding="user changes the opportunity stage")
    md = ism._loom_note_md("1146352141553540", _row([]), _URL1, a.loom_id, a)
    assert "authoritative: false" in md
    assert "source: loom" in md
    assert a.loom_id in md
    assert "user changes the opportunity stage" in md
    assert _URL1 in md


# ── per-row ingest: write, dedup, failure isolation ──────────────────────


@pytest.fixture
def _capture_storage(monkeypatch):
    writes = {}
    monkeypatch.setattr(ism.storage, "write_text", lambda p, c, **k: writes.__setitem__(p, c))
    return writes


def test_ingest_row_looms_writes_note(monkeypatch, _capture_storage):
    monkeypatch.setattr(ism.storage, "exists", lambda p: False)
    a = LoomAnalysis(loom_id="x", url=_URL1, title="T", duration=5.0,
                     narration_source="none", understanding="note")
    monkeypatch.setattr("kb.ingest_loom.ingest_loom", lambda url, *a2, **k: a)

    n = ism._ingest_row_looms("SHEET", _row([{"displayValue": _URL1}]), "support/raw/smartsheet/")
    assert n == 1
    [(path, body)] = _capture_storage.items()
    assert path.startswith("support/raw/smartsheet/sheet-SHEET-row-555-loom-")
    assert "authoritative: false" in body


def test_ingest_row_looms_dedup_skips_existing(monkeypatch, _capture_storage):
    monkeypatch.setattr(ism.storage, "exists", lambda p: True)  # already analyzed
    called = {"ran": False}
    monkeypatch.setattr("kb.ingest_loom.ingest_loom",
                        lambda *a, **k: called.__setitem__("ran", True))
    n = ism._ingest_row_looms("SHEET", _row([{"displayValue": _URL1}]), "p/")
    assert n == 0
    assert called["ran"] is False
    assert _capture_storage == {}


def test_ingest_row_looms_failure_isolated(monkeypatch, _capture_storage):
    monkeypatch.setattr(ism.storage, "exists", lambda p: False)

    def _boom(*a, **k):
        raise RuntimeError("yt-dlp exploded")

    monkeypatch.setattr("kb.ingest_loom.ingest_loom", _boom)
    n = ism._ingest_row_looms("SHEET", _row([{"displayValue": _URL1}]), "p/")
    assert n == 0           # failure doesn't propagate
    assert _capture_storage == {}


# ── gate ────────────────────────────────────────────────────────────────


def test_loom_ingest_gate(monkeypatch):
    monkeypatch.delenv("KB_LOOM_INGEST_ENABLED", raising=False)
    assert ism._loom_ingest_enabled() is False
    monkeypatch.setenv("KB_LOOM_INGEST_ENABLED", "on")
    assert ism._loom_ingest_enabled() is True
