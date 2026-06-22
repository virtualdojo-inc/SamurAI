"""Code sandbox tools — generate-and-run scripts in the isolated `samurai-sandbox`
service, plus reuse of vetted prior scripts.

Security model (the important part):
  - `run_code` is a WRITE-class tool (listed in judge.WRITE_TOOL_NAMES), so the
    safety judge reviews the script BEFORE it executes. Fail-closed.
  - Execution happens in `samurai-sandbox`: a zero-privilege Cloud Run service
    with no credentials and no network egress. It can only compute over the
    `inputs` we hand it and return text — it cannot reach the prod DB, Secret
    Manager, or the internet. Anything that must PERSIST to prod is NOT done
    here; the agent returns a proposed result and the apply goes through the
    existing Approve/Revise/Reject card.
  - Every run is recorded in `code_runs` (with an embedding) so a vetted script
    can be reused via `find_prior_script` instead of regenerated.

Gated by SAMURAI_SANDBOX_ENABLED (off by default — set to "on" to enable).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 15
_RESULT_SUMMARY_CAP = 2000


# ── Config / guards ───────────────────────────────────────────────────────


def _sandbox_enabled() -> bool:
    return os.environ.get("SAMURAI_SANDBOX_ENABLED", "").lower() in {"on", "1", "true", "yes"}


def _sandbox_url() -> str:
    return os.environ.get("SANDBOX_URL", "").rstrip("/")


def _sandbox_token() -> str:
    return os.environ.get("SANDBOX_TOKEN", "")


def _is_postgres() -> bool:
    """code_runs (a pgvector table) only exists on Postgres; the SQLite fallback
    never sees the Vector column. Persistence/search no-op gracefully otherwise."""
    try:
        from db.session import _database_url

        return "postgresql" in _database_url()
    except Exception:
        return False


def _inputs_hash(inputs: Any) -> str:
    try:
        blob = json.dumps(inputs, sort_keys=True, default=str)
    except Exception:
        blob = str(inputs)
    return hashlib.sha256(blob.encode()).hexdigest()


# Defense-in-depth pre-screen (CODE-3): reject scripts using obvious escape /
# network / process primitives BEFORE they reach the executor. The sandbox is
# "pure compute over inputs" and legitimate scripts never need these. This is a
# cheap backstop, NOT the boundary — the real controls are infra egress denial +
# zero-role SA (see docs/code_sandbox_plan.md). The judge does not statically
# inspect script bodies, so this closes the subprocess/raw-socket gap it misses.
_BLOCKED_PRIMITIVES = [
    ("ctypes", re.compile(r"\bctypes\b")),
    ("_socket", re.compile(r"\b_socket\b")),
    ("os.system", re.compile(r"\bos\.system\b")),
    ("subprocess", re.compile(r"\bsubprocess\b")),
    ("os.fork", re.compile(r"\bos\.fork\b")),
    ("os.posix_spawn", re.compile(r"\bos\.posix_spawn\b")),
    ("os.exec*", re.compile(r"\bos\.exec[lv]")),
    ("importlib.reload", re.compile(r"\bimportlib\.reload\b|\breload\s*\(\s*socket\b")),
]


def _prescreen(script: str) -> Optional[str]:
    """Return a reason string if the script must be refused, else None."""
    for name, pat in _BLOCKED_PRIMITIVES:
        if pat.search(script):
            return (f"blocked primitive '{name}' — the sandbox is for pure compute "
                    "over inputs (no process/network escape)")
    return None


_REDACTORS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),                          # emails
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"),                     # bearer tokens
    re.compile(r"AIza[0-9A-Za-z._\-]{10,}"),                          # google api keys
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[=:]\s*\S+"),
]


def _redact(s: str) -> str:
    for r in _REDACTORS:
        s = r.sub("[redacted]", s)
    return s


def _result_summary(out: dict) -> str:
    """What to persist in code_runs.result_summary (CODE-3 hygiene): prefer the
    structured `result`; otherwise a SIZE summary, never raw stdout/stderr
    verbatim. Lightly redact obvious secrets either way."""
    if out.get("result") is not None:
        s = json.dumps(out["result"], default=str)
    else:
        s = (f"(no result var; stdout {len(out.get('stdout') or '')} chars, "
             f"stderr {len(out.get('stderr') or '')} chars)")
    return _redact(s)[:_RESULT_SUMMARY_CAP]


def _embed(text: str) -> Optional[list[float]]:
    """Embed text with the same Vertex embedder the memory store uses. Returns
    None if embedding is unavailable (e.g., no Vertex auth in tests)."""
    try:
        from memory import _create_embed_fn

        vecs = _create_embed_fn()([text])
        return list(vecs[0]) if vecs else None
    except Exception as e:  # pragma: no cover - depends on Vertex auth
        logger.warning("[code_sandbox] embed failed: %s", e)
        return None


async def _execute(script: str, inputs: Any, timeout_s: int) -> dict:
    """Call the sandbox service. Returns its JSON, or an {"error": ...} dict."""
    url, token = _sandbox_url(), _sandbox_token()
    if not url or not token:
        return {"error": "sandbox not configured (SANDBOX_URL / SANDBOX_TOKEN unset)"}
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout_s + 10) as client:
            resp = await client.post(
                f"{url}/run",
                headers={"Authorization": f"Bearer {token}"},
                json={"script": script, "inputs": inputs, "timeout_s": timeout_s},
            )
        if resp.status_code != 200:
            return {"error": f"sandbox HTTP {resp.status_code}: {resp.text[:500]}"}
        return resp.json()
    except Exception as e:
        return {"error": f"sandbox call failed: {type(e).__name__}: {e}"}


async def _record_run(description: str, script: str, inputs: Any, out: dict) -> Optional[str]:
    """Best-effort persist of a CodeRun row (+ embedding). Never raises."""
    if not _is_postgres():
        return None
    try:
        from db.models import CodeRun
        from db.session import get_sessionmaker

        summary = _result_summary(out)
        embedding = _embed(f"{description}\n\n{script}")
        row = CodeRun(
            description=description,
            language="python",
            script=script,
            inputs_hash=_inputs_hash(inputs),
            outcome="ok" if out.get("outcome") == "ok" else "fail",
            result_summary=summary,
            embedding=embedding,
            reusable=False,
        )
        sm = get_sessionmaker()
        async with sm() as session:
            session.add(row)
            await session.commit()
            return str(row.id)
    except Exception as e:
        logger.warning("[code_sandbox] record_run failed: %s", e)
        return None


# ── Tools ───────────────────────────────────────────────────────────────


@tool
async def run_code(
    description: str,
    script: str,
    inputs: Optional[Any] = None,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> str:
    """Run a Python script in the isolated, zero-privilege sandbox and return its output.

    Use this for computation, data analysis, transforms, parsing, or
    codegen-and-test over data you have ALREADY fetched with other tools. The
    sandbox has NO network and NO credentials — it cannot reach the database,
    secrets, or the internet. Pass any data the script needs via `inputs`
    (available inside the script as the `inputs` global); the script can return
    structured data by assigning to a `result` variable.

    Do NOT use this to persist changes to production — it can't, by design.
    Compute the proposed change here, then apply it through the normal
    approval-gated tools.

    Args:
        description: One-line summary of what the script does (used for reuse search).
        script: The Python source to execute.
        inputs: Optional JSON-serializable data exposed to the script as `inputs`.
        timeout_s: Wall-clock limit (capped by the sandbox; default 15s).
    """
    if not _sandbox_enabled():
        return "The code sandbox is disabled (SAMURAI_SANDBOX_ENABLED is off)."

    blocked = _prescreen(script)
    if blocked:
        return f"Refused before execution: {blocked}."

    out = await _execute(script, inputs, int(timeout_s))
    if "error" in out:
        return f"Sandbox error: {out['error']}"

    run_id = await _record_run(description, script, inputs, out)

    parts = [f"outcome: {out.get('outcome')}  ({out.get('elapsed_ms')}ms, exit={out.get('exit_code')})"]
    if out.get("stdout"):
        parts.append(f"stdout:\n{out['stdout']}")
    if out.get("stderr"):
        parts.append(f"stderr:\n{out['stderr']}")
    if out.get("result") is not None:
        parts.append(f"result: {json.dumps(out['result'], default=str)[:_RESULT_SUMMARY_CAP]}")
    if run_id:
        parts.append(f"(recorded as code_run {run_id} — say so if it's worth keeping for reuse)")
    return "\n".join(parts)


@tool
async def find_prior_script(query: str, limit: int = 3) -> str:
    """Search previously-run, successful scripts for one matching `query` (semantic).

    Returns the closest prior `code_runs` (description + script + outcome) so a
    vetted script can be reused instead of regenerated. Only available on the
    Postgres backend.
    """
    if not _is_postgres():
        return "No prior-script library available on this backend."
    try:
        from sqlalchemy import select

        from db.models import CodeRun
        from db.session import get_sessionmaker

        qvec = _embed(query)
        if qvec is None:
            return "Could not embed the query to search prior scripts."

        sm = get_sessionmaker()
        async with sm() as session:
            stmt = (
                select(CodeRun)
                .where(CodeRun.embedding.isnot(None), CodeRun.outcome == "ok")
                .order_by(CodeRun.embedding.cosine_distance(qvec))
                .limit(max(1, min(int(limit), 10)))
            )
            rows = (await session.execute(stmt)).scalars().all()
        if not rows:
            return "No matching prior scripts found."
        out = []
        for r in rows:
            tag = " [reusable]" if r.reusable else ""
            out.append(
                f"• {r.description}{tag} (outcome={r.outcome}, id={r.id})\n"
                f"```python\n{r.script}\n```"
            )
        return "\n\n".join(out)
    except Exception as e:
        logger.warning("[code_sandbox] find_prior_script failed: %s", e)
        return f"Prior-script search failed: {type(e).__name__}: {e}"


CODE_SANDBOX_TOOLS = [run_code, find_prior_script]
