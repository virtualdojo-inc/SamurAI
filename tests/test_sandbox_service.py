"""Tests for the sandbox executor service (sandbox/app.py).

These run REAL subprocesses to prove the isolation actually holds: no network,
no inherited secrets, no escaping the timeout, structured I/O works. The service
is deployed on Linux (Cloud Run); these exercise the same code path locally.
"""
import importlib.util
import json
import os
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parent.parent / "sandbox" / "app.py"
_spec = importlib.util.spec_from_file_location("sandbox_app", _PATH)
sapp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sapp)


# ── Real-execution isolation tests ─────────────────────────────────────────


def test_happy_path_stdout():
    out = sapp._run_blocking("print('hello world')", None, 10)
    assert out["outcome"] == "ok"
    assert "hello world" in out["stdout"]
    assert out["exit_code"] == 0


def test_inputs_are_available_to_script():
    out = sapp._run_blocking("result = inputs['x'] * 2", {"x": 21}, 10)
    assert out["outcome"] == "ok"
    assert out["result"] == 42


def test_structured_result_roundtrip():
    out = sapp._run_blocking("result = {'sum': sum(range(5))}", None, 10)
    assert out["result"] == {"sum": 10}


def test_error_is_captured_not_raised():
    out = sapp._run_blocking("raise ValueError('boom')", None, 10)
    assert out["outcome"] == "error"
    assert "boom" in out["stderr"]
    assert out["exit_code"] != 0


def test_network_is_blocked():
    script = (
        "import socket\n"
        "try:\n"
        "    socket.socket()\n"
        "    print('OPENED')\n"
        "except OSError:\n"
        "    print('BLOCKED')\n"
    )
    out = sapp._run_blocking(script, None, 10)
    assert "BLOCKED" in out["stdout"]
    assert "OPENED" not in out["stdout"]


def test_parent_env_secrets_not_inherited(monkeypatch):
    monkeypatch.setenv("SUPER_SECRET_LEAK", "topsecret")
    out = sapp._run_blocking(
        "import os; print(os.environ.get('SUPER_SECRET_LEAK', 'NONE'))", None, 10
    )
    assert "topsecret" not in out["stdout"]
    assert "NONE" in out["stdout"]


def test_timeout_kills_runaway():
    out = sapp._run_blocking("while True:\n    pass", None, 1)
    assert out["outcome"] == "timeout"


def test_output_is_capped(monkeypatch):
    monkeypatch.setattr(sapp, "OUTPUT_CAP", 100)
    out = sapp._run_blocking("print('A' * 10000)", None, 10)
    assert len(out["stdout"]) <= 100


def test_streaming_output_is_bounded_and_blocked(monkeypatch):
    """CODE-1: a child streaming unbounded output must not balloon the parent —
    we cap accumulation at ~2*OUTPUT_CAP, then SIGKILL and report 'blocked'."""
    monkeypatch.setattr(sapp, "OUTPUT_CAP", 1000)
    script = "import sys\nwhile True:\n    sys.stdout.write('X' * 65536)\n    sys.stdout.flush()\n"
    out = sapp._run_blocking(script, None, 10)
    assert out["outcome"] == "blocked"
    assert len(out["stdout"]) <= 1000


def test_forked_grandchild_does_not_hang():
    """CODE-2: a grandchild that escapes the process group and holds the pipe
    open must not wedge the reader forever — the call returns promptly."""
    import time as _t

    script = (
        "import os, sys, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os.setsid()\n"          # escape the sandbox child's process group
        "    time.sleep(5)\n"         # survive the kill, holding inherited stdout
        "else:\n"
        "    print('parent-exiting')\n"
        "    sys.exit(0)\n"
    )
    t0 = _t.monotonic()
    out = sapp._run_blocking(script, None, 3)
    elapsed = _t.monotonic() - t0
    assert elapsed < 12  # bounded (deadline + grace + reap), NOT forever
    assert out["outcome"] in ("ok", "timeout", "error")


# ── Auth on the HTTP handler ────────────────────────────────────────────────


class _Req:
    def __init__(self, headers=None, body=None):
        self.headers = headers or {}
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


async def test_run_requires_token(monkeypatch):
    monkeypatch.setenv("SANDBOX_TOKEN", "sk")
    resp = await sapp.handle_run(_Req(body={"script": "print(1)"}))
    assert resp.status == 401


async def test_run_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("SANDBOX_TOKEN", "sk")
    resp = await sapp.handle_run(
        _Req(headers={"Authorization": "Bearer nope"}, body={"script": "print(1)"})
    )
    assert resp.status == 401


async def test_run_fail_closed_when_no_token(monkeypatch):
    monkeypatch.delenv("SANDBOX_TOKEN", raising=False)
    resp = await sapp.handle_run(
        _Req(headers={"Authorization": "Bearer anything"}, body={"script": "print(1)"})
    )
    assert resp.status == 401


async def test_run_requires_script(monkeypatch):
    monkeypatch.setenv("SANDBOX_TOKEN", "sk")
    resp = await sapp.handle_run(_Req(headers={"Authorization": "Bearer sk"}, body={}))
    assert resp.status == 400


async def test_run_executes_with_valid_token(monkeypatch):
    monkeypatch.setenv("SANDBOX_TOKEN", "sk")
    resp = await sapp.handle_run(
        _Req(headers={"Authorization": "Bearer sk"}, body={"script": "print('ok')"})
    )
    assert resp.status == 200
    data = json.loads(resp.body)
    assert data["outcome"] == "ok" and "ok" in data["stdout"]
