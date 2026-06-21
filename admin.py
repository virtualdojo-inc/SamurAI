"""Secured admin endpoint for operational + upgrade tasks.

POST /admin  with  Authorization: Bearer <SAMURAI_ADMIN_KEY>
Body: {"op": "<name>", "args": {...}}

Security model (the endpoint is on the public Cloud Run URL, so this matters):
  - FAIL CLOSED: disabled (404) unless SAMURAI_ADMIN_KEY is set (Secret Manager).
  - Bearer token, constant-time compare (hmac.compare_digest) — no timing oracle.
  - FIXED OP ALLOWLIST — there is deliberately NO arbitrary-code / arbitrary-SQL
    op. db_query is read-only (single SELECT/WITH, forbidden-keyword + semicolon
    block, read-only transaction, statement timeout, row cap).
  - Per-IP rate limit (best-effort, in-process).
  - Every call is audited to stdout (op, verified, client IP, outcome).

This is for read/inspect/trigger-known-op tasks (pull logs, run a read query,
trigger the one-shot data migration). Mutating/arbitrary work belongs behind the
approval-card flow, not here.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
import time

from aiohttp import web

logger = logging.getLogger(__name__)

ADMIN_KEY_ENV = "SAMURAI_ADMIN_KEY"
_MAX_ROWS = 200
_RATE_LIMIT_PER_MIN = 30
_rate: dict[str, list[float]] = {}

# db_query guard: a single read-only SELECT/WITH, no write/DDL keywords, no `;`.
_READ_SQL = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|call|do|merge|vacuum)\b",
    re.IGNORECASE,
)


def _client_ip(request: web.Request) -> str:
    xff = request.headers.get("X-Forwarded-For", "")
    return (xff.split(",")[0].strip() if xff else (request.remote or "")) or "unknown"


def _rate_ok(ip: str) -> bool:
    now = time.time()
    window = [t for t in _rate.get(ip, []) if now - t < 60]
    window.append(now)
    _rate[ip] = window
    return len(window) <= _RATE_LIMIT_PER_MIN


def _authorized(request: web.Request) -> bool:
    """Constant-time bearer check; fail-closed when no key is configured."""
    secret = os.environ.get(ADMIN_KEY_ENV, "")
    if not secret:
        return False
    auth = request.headers.get("Authorization", "")
    prefix = "Bearer "
    if not auth.startswith(prefix):
        return False
    return hmac.compare_digest(auth[len(prefix):], secret)


def _admin_enabled() -> bool:
    return bool(os.environ.get(ADMIN_KEY_ENV, ""))


# ── Allowlisted ops ──────────────────────────────────────────────────────


async def _op_ping(args: dict) -> dict:
    return {"ok": True, "revision": os.environ.get("K_REVISION", "?")}


async def _op_db_query(args: dict) -> dict:
    sql = (args.get("sql") or "").strip()
    if not sql or not _READ_SQL.match(sql) or _FORBIDDEN_SQL.search(sql) or ";" in sql:
        return {"error": "only a single read-only SELECT/WITH query is allowed"}

    from sqlalchemy import text

    from db.session import init_engine

    engine = init_engine()
    async with engine.connect() as conn:
        ro = await conn.execution_options(postgresql_readonly=True)
        await ro.execute(text("SET LOCAL statement_timeout = '15s'"))
        res = await ro.execute(text(sql))
        rows = [dict(r._mapping) for r in res.fetchmany(_MAX_ROWS)]
    # JSON-safe: stringify anything non-trivial.
    safe = [{k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
             for k, v in row.items()} for row in rows]
    return {"rows": safe, "count": len(safe), "truncated": len(safe) >= _MAX_ROWS}


async def _op_logs(args: dict) -> dict:
    """Recent Cloud Logging entries for the bot (read-only)."""
    limit = min(int(args.get("limit", 30)), 100)
    extra = args.get("filter", "")
    flt = ('resource.type="cloud_run_revision" '
           'resource.labels.service_name="samurai-bot"')
    if extra and all(c not in extra for c in '\n"'):  # tiny injection guard
        flt += f" {extra}"
    from google.cloud import logging as gcl

    client = gcl.Client(project=os.environ.get("GCP_PROJECT_ID", "virtualdojo-samurai"))
    out = []
    for e in client.list_entries(filter_=flt, order_by=gcl.DESCENDING, max_results=limit):
        payload = e.payload if isinstance(e.payload, str) else json.dumps(e.payload)[:500]
        out.append({"ts": str(e.timestamp), "severity": str(e.severity), "text": payload})
    return {"entries": out, "count": len(out)}


async def _op_migrate_data(args: dict) -> dict:
    """Trigger the one-shot /data SQLite -> Postgres migration (idempotent)."""
    from migrate_data import run as migrate_run

    return await migrate_run()


_OPS = {
    "ping": _op_ping,
    "db_query": _op_db_query,
    "logs": _op_logs,
    "migrate_data": _op_migrate_data,
}


async def handle_admin(request: web.Request) -> web.Response:
    ip = _client_ip(request)
    if not _admin_enabled():
        return web.json_response({"error": "admin endpoint disabled"}, status=404)
    if not _authorized(request):
        logger.warning("[admin] unauthorized attempt from %s", ip)
        print(f"[admin] DENIED unauthorized from {ip}", flush=True)
        return web.json_response({"error": "unauthorized"}, status=401)
    if not _rate_ok(ip):
        return web.json_response({"error": "rate limited"}, status=429)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    op = body.get("op")
    args = body.get("args") or {}
    handler = _OPS.get(op)
    if handler is None:
        return web.json_response(
            {"error": f"unknown op {op!r}", "allowed": sorted(_OPS)}, status=400
        )

    print(f"[admin] op={op} ip={ip} args_keys={sorted(args)}", flush=True)
    try:
        result = await handler(args)
        print(f"[admin] op={op} OK", flush=True)
        return web.json_response({"op": op, "result": result})
    except Exception as e:
        logger.exception("[admin] op %s failed", op)
        print(f"[admin] op={op} FAILED {type(e).__name__}: {e}", flush=True)
        return web.json_response({"op": op, "error": f"{type(e).__name__}: {e}"}, status=500)
