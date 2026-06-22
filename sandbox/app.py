"""samurai-sandbox — an isolated, zero-privilege code executor.

This service runs UNTRUSTED, LLM-generated scripts. It is deployed as its own
Cloud Run service with:
  - a service account holding NO IAM roles (no Secret Manager, no DB, no GCP),
  - --ingress=internal (only the bot, over the private network, can reach it),
  - egress denied (it cannot phone home even if code tries).

So the *infrastructure* is the real boundary. Everything in this file is
defense-in-depth on top of that — the goal is that hostile code can, at worst,
burn its own CPU/memory budget and return text. It must never reach the
network, read a secret, escape /tmp, or outlive its timeout.

Contract:
  POST /run   Authorization: Bearer <SANDBOX_TOKEN>
    body:  {"script": "<python>", "inputs": <json|null>, "timeout_s": <int>}
    reply: {"outcome": "ok|error|timeout|blocked",
            "stdout": "...", "stderr": "...", "result": <json|null>,
            "elapsed_ms": <int>, "exit_code": <int|null>}
  GET  /health  -> 200 (unauthenticated; Cloud Run startup/liveness probe)

The script may read `inputs` (a pre-defined global) and may set a `result`
variable OR write JSON to the path in $SANDBOX_RESULT to return structured data.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import resource
import signal
import sys
import tempfile
import time

from aiohttp import web

# ── Limits (env-tunable; conservative defaults) ───────────────────────────
TOKEN_ENV = "SANDBOX_TOKEN"
MAX_TIMEOUT_S = int(os.environ.get("SANDBOX_MAX_TIMEOUT_S", "30"))
DEFAULT_TIMEOUT_S = int(os.environ.get("SANDBOX_DEFAULT_TIMEOUT_S", "15"))
MEM_LIMIT_BYTES = int(os.environ.get("SANDBOX_MEM_MB", "512")) * 1024 * 1024
FSIZE_LIMIT_BYTES = int(os.environ.get("SANDBOX_FSIZE_MB", "50")) * 1024 * 1024
NOFILE_LIMIT = int(os.environ.get("SANDBOX_NOFILE", "256"))
NPROC_LIMIT = int(os.environ.get("SANDBOX_NPROC", "128"))
OUTPUT_CAP = int(os.environ.get("SANDBOX_OUTPUT_CAP", str(256 * 1024)))  # bytes
SCRIPT_CAP = int(os.environ.get("SANDBOX_SCRIPT_CAP", str(256 * 1024)))

# Preamble prepended to every script. Loads `inputs`, blocks the network as a
# seatbelt (egress is also denied at the infra layer), and exposes $SANDBOX_RESULT.
_HARNESS = r'''
import os as _os, json as _json, socket as _socket
# Seatbelt: neuter the network. Egress is denied at the infra layer too; this
# turns an attempted connection into an immediate, legible error.
def _no_net(*_a, **_k):
    raise OSError("network access is disabled in the sandbox")
_socket.socket = _no_net
_socket.create_connection = _no_net
try:
    inputs = _json.load(open(_os.environ["SANDBOX_INPUTS"])) if _os.environ.get("SANDBOX_INPUTS") else None
except Exception:
    inputs = None
result = None
def _emit_result():
    p = _os.environ.get("SANDBOX_RESULT")
    if p and ("result" in globals()) and globals()["result"] is not None:
        try:
            open(p, "w").write(_json.dumps(globals()["result"], default=str))
        except Exception:
            pass
import atexit as _atexit
_atexit.register(_emit_result)
# ── user script follows ──
'''


def _authorized(request: web.Request) -> bool:
    secret = os.environ.get(TOKEN_ENV, "")
    if not secret:
        return False  # fail closed: no token configured -> nobody gets in
    auth = request.headers.get("Authorization", "")
    prefix = "Bearer "
    if not auth.startswith(prefix):
        return False
    return hmac.compare_digest(auth[len(prefix):], secret)


def _set_rlimits(timeout_s: int):
    """preexec_fn for the child: cap CPU, memory, file size, fds, procs.

    Each limit is best-effort and independent — some platforms refuse a given
    limit (e.g. macOS rejects a low RLIMIT_AS because it counts the virtual
    address space the interpreter mmaps at startup). A refused limit must not
    abort the spawn; the hard guarantees (wall-clock kill, scrubbed env, no
    network, /tmp-only cwd) hold regardless. On Linux (Cloud Run) they all apply.
    """
    cpu = min(timeout_s + 1, MAX_TIMEOUT_S + 1)
    for res, limit in (
        (resource.RLIMIT_CPU, cpu),
        (resource.RLIMIT_AS, MEM_LIMIT_BYTES),
        (resource.RLIMIT_FSIZE, FSIZE_LIMIT_BYTES),
        (resource.RLIMIT_NOFILE, NOFILE_LIMIT),
        (getattr(resource, "RLIMIT_NPROC", None), NPROC_LIMIT),
    ):
        if res is None:
            continue
        try:
            resource.setrlimit(res, (limit, limit))
        except (ValueError, OSError):
            pass
    try:
        os.setsid()  # own process group so we can kill the whole tree on timeout
    except OSError:
        pass


def _run_blocking(script: str, inputs, timeout_s: int) -> dict:
    """Run the composed program in an isolated child process. Blocking — call
    from a thread. Returns the result dict."""
    workdir = tempfile.mkdtemp(prefix="sbx-")
    inputs_path = os.path.join(workdir, "inputs.json")
    result_path = os.path.join(workdir, "result.json")
    with open(inputs_path, "w") as f:
        json.dump(inputs, f, default=str)

    program = _HARNESS + "\n" + script
    # Minimal, scrubbed environment — no inherited secrets. -I = isolated mode
    # (ignore PYTHON* env + user site), -B = no .pyc writes, -S = no site.
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": workdir,
        "TMPDIR": workdir,
        "SANDBOX_INPUTS": inputs_path,
        "SANDBOX_RESULT": result_path,
        "PYTHONUNBUFFERED": "1",
    }

    import subprocess

    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-B", "-S", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=workdir, env=env, preexec_fn=lambda: _set_rlimits(timeout_s),
            text=True,
        )
    except Exception as e:
        return {"outcome": "error", "stdout": "", "stderr": f"spawn failed: {e}",
                "result": None, "elapsed_ms": 0, "exit_code": None}

    outcome = "ok"
    try:
        stdout, stderr = proc.communicate(input=program, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        outcome = "timeout"
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        stdout, stderr = proc.communicate()
    elapsed = int((time.monotonic() - started) * 1000)

    exit_code = proc.returncode
    if outcome == "ok" and exit_code != 0:
        outcome = "error"

    result = None
    try:
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
    except Exception:
        result = None

    # Best-effort cleanup of the workdir.
    try:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception:
        pass

    return {
        "outcome": outcome,
        "stdout": (stdout or "")[:OUTPUT_CAP],
        "stderr": (stderr or "")[:OUTPUT_CAP],
        "result": result,
        "elapsed_ms": elapsed,
        "exit_code": exit_code,
    }


async def handle_run(request: web.Request) -> web.Response:
    if not _authorized(request):
        return web.json_response({"error": "unauthorized"}, status=401)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    script = body.get("script") or ""
    if not isinstance(script, str) or not script.strip():
        return web.json_response({"error": "script is required"}, status=400)
    if len(script) > SCRIPT_CAP:
        return web.json_response({"error": "script too large"}, status=413)

    inputs = body.get("inputs")
    try:
        timeout_s = int(body.get("timeout_s", DEFAULT_TIMEOUT_S))
    except (TypeError, ValueError):
        timeout_s = DEFAULT_TIMEOUT_S
    timeout_s = max(1, min(timeout_s, MAX_TIMEOUT_S))

    out = await asyncio.to_thread(_run_blocking, script, inputs, timeout_s)
    return web.json_response(out)


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


def build_app() -> web.Application:
    app = web.Application(client_max_size=2 * 1024 * 1024)
    app.router.add_post("/run", handle_run)
    app.router.add_get("/health", handle_health)
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    web.run_app(build_app(), host="0.0.0.0", port=port)
