"""Unit tests for the in-boundary knowledge-base pipeline (kb/).

All tests use a fake in-memory storage and a fake LLM — no real GCS bucket and no
Vertex calls, so nothing touches protected data.
"""

from datetime import datetime, timezone

import pytest

from kb import compile as kb_compile
from kb import ingest_github
from kb import ingest_smartsheet


class FakeStorage:
    def __init__(self):
        self.objs: dict[str, str] = {}

    def read_text(self, p):
        return self.objs.get(p)

    def write_text(self, p, c, content_type="text/markdown"):
        self.objs[p] = c

    def exists(self, p):
        return p in self.objs

    def list_paths(self, prefix):
        return [k for k in self.objs if k.startswith(prefix)]

    def list_text(self, prefix, suffix=".md"):
        return [(k, v) for k, v in self.objs.items() if k.startswith(prefix) and k.endswith(suffix)]


class _Resp:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Returns extraction JSON for extract prompts, an article body otherwise."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        system = messages[0].content.lower()
        if "extract knowledge" in system:
            return _Resp(
                '{"topic_slug":"login-issues","title":"Login Issues",'
                '"one_line":"Users hit login errors","key_facts":["SSO token expiry",'
                '"clock skew","retry guidance"],"sensitive":false,"sensitive_kinds":[]}'
            )
        return _Resp("# Login Issues\n\nGrounded summary.\n\n## Related\n- [[other-topic]]")


# ---- secret scrubbing -------------------------------------------------------

def test_scrub_redacts_secrets():
    text = "token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 here"
    clean, n = ingest_github._scrub(text)
    assert n == 1
    assert "ghp_" not in clean
    assert "[REDACTED-SECRET]" in clean


def test_scrub_clean_text_untouched():
    clean, n = ingest_github._scrub("a normal support note")
    assert n == 0
    assert clean == "a normal support note"


# ---- github ingest ----------------------------------------------------------

class _FakeIssue:
    def __init__(self, number, body="", is_pr=False):
        self.number = number
        self.title = f"Issue {number}"
        self.body = body
        self.state = "open"
        self.html_url = f"https://github.com/virtualdojo-inc/virtualdojo/issues/{number}"
        self.created_at = datetime(2026, 5, 1, tzinfo=timezone.utc)
        self.updated_at = datetime(2026, 5, 28, tzinfo=timezone.utc)
        self.labels = []
        self.pull_request = object() if is_pr else None


class _FakeRepo:
    def __init__(self, issues):
        self._issues = issues

    def get_issues(self, **kwargs):
        return self._issues


def test_refresh_github_issues(monkeypatch):
    fake = FakeStorage()
    monkeypatch.setattr(ingest_github, "storage", fake)
    monkeypatch.setattr(
        ingest_github, "_github",
        lambda: type("G", (), {"get_repo": lambda self, r: _FakeRepo([
            _FakeIssue(1, body="login fails"),
            _FakeIssue(2, body="secret ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
            _FakeIssue(3, is_pr=True),  # PR — skipped
        ])})(),
    )
    stats = ingest_github.refresh_github_issues()
    assert stats["issues_written"] == 2
    assert stats["prs_skipped"] == 1
    assert stats["secrets_redacted"] == 1
    assert "support/raw/github-issues/issue-1.md" in fake.objs
    assert "support/raw/github-issues/issue-2.md" in fake.objs
    assert "[REDACTED-SECRET]" in fake.objs["support/raw/github-issues/issue-2.md"]
    # watermark stored
    assert fake.objs.get("support/raw/.state/github_last_sync.txt")


# ---- smartsheet discovery + routing -----------------------------------------

def test_smartsheet_classify():
    # DH Tech Issue Tracker id → support (known id + name)
    assert ingest_smartsheet._classify("1146352141553540", "DH Tech Issue Tracker") == "support"
    # onboarding by name
    assert ingest_smartsheet._classify("999", "Customer Onboarding 2026") == "onboarding"
    # support by name keyword
    assert ingest_smartsheet._classify("888", "Support Tickets") == "support"
    # unknown → skipped
    assert ingest_smartsheet._classify("777", "Q3 Revenue Forecast") is None


def test_smartsheet_ingest_routes_by_scope(monkeypatch):
    fake = FakeStorage()
    monkeypatch.setattr(ingest_smartsheet, "storage", fake)

    def _fake_get(path, params=None):
        if path == "/sheets":
            return {"data": [
                {"id": 1146352141553540, "name": "DH Tech Issue Tracker"},
                {"id": 222, "name": "Customer Onboarding"},
                {"id": 333, "name": "Finance Forecast"},  # skipped
            ]}
        # sheet detail
        sid = int(path.split("/")[-1])
        return {
            "id": sid, "name": "S",
            "columns": [{"id": 1, "title": "Issue"}],
            "rows": [{"id": 10, "rowNumber": 1, "cells": [{"columnId": 1, "displayValue": "login fails"}]}],
        }

    monkeypatch.setattr(ingest_smartsheet, "_get", _fake_get)
    stats = ingest_smartsheet.ingest_smartsheet()
    assert stats["support_rows"] == 1
    assert stats["onboarding_rows"] == 1
    assert stats["skipped_sheet_ids"] == ["333"]
    assert any(k.startswith("support/raw/smartsheet/sheet-1146352141553540-") for k in fake.objs)
    assert any(k.startswith("customers/onboarding/raw/smartsheet/sheet-222-") for k in fake.objs)


# ---- compile ----------------------------------------------------------------

def test_compile_support_produces_wiki_and_index(monkeypatch):
    fake = FakeStorage()
    fake.objs["support/raw/github-issues/issue-1.md"] = "user cannot log in, sso expired"
    fake.objs["support/raw/github-issues/issue-2.md"] = "login retry after clock fix"
    monkeypatch.setattr(kb_compile, "storage", fake)

    stats = kb_compile.compile_support(llm=FakeLLM())
    assert stats["raw_docs_processed"] == 2
    assert stats["articles_written"] >= 1
    # article written with provenance frontmatter
    art = fake.objs["support/wiki/login-issues.md"]
    assert art.startswith("---\n")
    assert "title:" in art and "sources:" in art and "last_verified:" in art and "confidence:" in art
    assert "[[" in art  # backlinks present
    # index regenerated
    assert "support/index.md" in fake.objs
    assert "[[login-issues]]" in fake.objs["support/index.md"]


def test_compile_is_idempotent(monkeypatch):
    fake = FakeStorage()
    fake.objs["support/raw/github-issues/issue-1.md"] = "same content"
    monkeypatch.setattr(kb_compile, "storage", fake)
    kb_compile.compile_support(llm=FakeLLM())
    stats2 = kb_compile.compile_support(llm=FakeLLM())  # nothing changed
    assert stats2["raw_docs_processed"] == 0


def test_compile_engine_is_in_boundary_gemini(monkeypatch):
    fake = FakeStorage()
    fake.objs["support/raw/github-issues/issue-1.md"] = "x"
    monkeypatch.setattr(kb_compile, "storage", fake)
    stats = kb_compile.compile_support(llm=FakeLLM())
    eng = stats["engine"]
    assert eng["engine"] == "vertex-gemini"
    assert eng["external_llm"] is False
    assert eng["location"] != "global"  # regional, in-boundary


# ---- support chat capture ---------------------------------------------------

def test_log_support_chat_gated_off(monkeypatch):
    import conversation_log
    monkeypatch.setattr(conversation_log, "SUPPORT_CHAT_CAPTURE", False)
    assert conversation_log.log_support_chat(
        conversation_id="c", user_id="u", user_message="m", assistant_response="r"
    ) is None


def test_log_support_chat_writes_when_enabled(monkeypatch):
    import conversation_log
    from kb import storage as kb_storage
    written = {}
    monkeypatch.setattr(conversation_log, "SUPPORT_CHAT_CAPTURE", True)
    monkeypatch.setattr(kb_storage, "write_text", lambda p, c, **k: written.update({p: c}))
    path = conversation_log.log_support_chat(
        conversation_id="c1", user_id="u1", user_message="login broken", assistant_response="try X",
        ts=datetime(2026, 5, 29, 10, 0, 0, tzinfo=timezone.utc),
    )
    assert path and path.startswith("support/conversation-history/2026-05-29/")
    body = next(iter(written.values()))
    assert "authoritative: false" in body  # marked as log, not source
    assert "login broken" in body
