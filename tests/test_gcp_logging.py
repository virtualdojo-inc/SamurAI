"""Tests for the compressed Cloud Logging reader (tools/gcp_logging.py)."""

from datetime import datetime, timezone
from types import SimpleNamespace

import tools.gcp_logging as gl


def test_message_prefers_structured_message():
    assert gl._message({"message": "boom", "severity": "ERROR"}) == "boom"


def test_message_plain_text():
    assert gl._message("just text") == "just text"


def test_message_audit_status():
    payload = {"methodName": "google.run.deploy", "status": {"message": "denied"}}
    assert gl._message(payload) == "google.run.deploy: denied"


def test_message_json_fallback_is_bounded():
    out = gl._message({"a": "x" * 1000})
    assert len(out) <= gl._MAX_MSG_CHARS


def _entry(ts, sev, payload, rev="rev-1"):
    return SimpleNamespace(
        timestamp=datetime(2026, 7, 12, ts, 0, tzinfo=timezone.utc),
        severity=sev,
        payload=payload,
        resource=SimpleNamespace(labels={"revision_name": rev}),
    )


class _FakeClient:
    def __init__(self, entries):
        self._entries = entries

    def list_entries(self, filter_=None, order_by=None, max_results=None):
        # tool asks newest-first; hand back in that order.
        return list(self._entries)[:max_results]


def _patch_client(monkeypatch, entries):
    fake_module = SimpleNamespace(
        Client=lambda project=None: _FakeClient(entries),
        DESCENDING="desc",
    )
    import google.cloud

    monkeypatch.setattr(google.cloud, "logging", fake_module, raising=False)
    monkeypatch.setenv("GCP_PROJECT_ID", "test-proj")


def _run(**kw):
    return gl.query_cloud_logs.invoke({"filter_query": 'severity>=ERROR', **kw})


def test_empty(monkeypatch):
    _patch_client(monkeypatch, [])
    assert "No log entries" in _run()


def test_noise_stripped_by_default(monkeypatch):
    entries = [
        _entry(10, "ERROR", "real failure"),
        _entry(9, "WARNING", "OutOfOrderError on tasks.sqlite-journal"),
    ]
    _patch_client(monkeypatch, entries)
    out = _run()
    assert "real failure" in out
    assert "OutOfOrderError" not in out
    assert "omitted" in out  # dropped count reported


def test_noise_kept_when_disabled(monkeypatch):
    entries = [_entry(9, "WARNING", "OutOfOrderError blah")]
    _patch_client(monkeypatch, entries)
    assert "OutOfOrderError" in _run(exclude_regex="")


def test_include_regex(monkeypatch):
    entries = [_entry(10, "ERROR", "TimeoutError x"), _entry(9, "ERROR", "ValueError y")]
    _patch_client(monkeypatch, entries)
    out = _run(include_regex="TimeoutError")
    assert "TimeoutError" in out and "ValueError" not in out


def test_collapse_repeats(monkeypatch):
    entries = [_entry(10, "ERROR", "same"), _entry(10, "ERROR", "same"), _entry(10, "ERROR", "same")]
    _patch_client(monkeypatch, entries)
    out = _run()
    assert "(x3)" in out


def test_chronological_order(monkeypatch):
    # newest-first from API -> output should read oldest-first.
    entries = [_entry(11, "ERROR", "later"), _entry(10, "ERROR", "earlier")]
    _patch_client(monkeypatch, entries)
    out = _run()
    assert out.index("earlier") < out.index("later")
