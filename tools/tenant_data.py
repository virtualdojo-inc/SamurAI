"""Read-only tenant-data tools via VirtualDojo support grants.

AUTH: reuses SamurAI's EXISTING per-user VirtualDojo SSO sign-in (tools/virtualdojo_mcp:
start_oauth_flow / token store). Reads run AS the signed-in user — their identity and their
``system_administrator`` rights in the support tenant. No service account, no API key.
The SSO token is a standard VirtualDojo JWT (``create_access_token`` -> ``get_current_user``
just decodes it with SECRET_KEY), so the SAME token authenticates ``/api/v1`` REST. That
holds only if the user's SSO session and ``VIRTUALDOJO_API_URL`` target the SAME backend
(shared SECRET_KEY) — see ``_warn_if_sso_env_mismatch``. If the user isn't signed in, the
tool returns the SSO sign-in prompt. Because a background task has no signed-in user,
autonomous / scheduled reads of tenant data are barred for free.

READ-ONLY: data + schema are fetched with GET only; ``confirm_write`` is never sent. The
ONLY non-GET calls are impersonation session start/end (``_vdj_post``) — session lifecycle
that mints a short-lived read token and changes no customer data (the backend audits it
into the customer tenant). There is deliberately no data-mutating path in this module.
Gated by ``SAMURAI_TENANT_DATA_ENABLED`` (off by default).

Flow (Phase 2): list grants -> pick a grant_id -> start a 15-min impersonation session as
the granting customer user -> GET schema/records as that user -> end the session.
  - GET  /api/v1/impersonation/my-grants                  (which tenants granted us)
  - POST /api/v1/impersonation/start/{grant_id}           (mint a read session)
  - GET  /api/v1/schema/objects[/{object}/schema]         (object + field metadata)
  - GET  /api/v1/objects/{object}/records?skip&limit      (the actual rows)
  - POST /api/v1/impersonation/end                        (best-effort cleanup)

⚠ PII / residency: record rows may be customer PII/CUI. Reads are NEVER fed to LangMem
extraction or the KB bucket; only metadata (counts / object names) is logged. The serving
chat model is on the Vertex *global* endpoint — summarizing raw rows through it is a
residency item to resolve / get ATO sign-off on before enabling reads in prod. Because of
that, raw record reads have their OWN gate (``SAMURAI_TENANT_RECORDS_ENABLED``) on top of
``SAMURAI_TENANT_DATA_ENABLED``: list-grants + describe-schema (no row PII) can run while
record reads stay off until residency is signed off.

Per-user factory, like create_virtualdojo_tool. See docs/tenant_data_access_plan.md.
"""
from __future__ import annotations

import logging
import os

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# How many record rows we ever pull in one read (defense-in-depth cap; the backend also
# caps limit<=1000). Keeps a single read from dragging large PII volumes through the model.
_MAX_RECORD_LIMIT = 200


def _enabled() -> bool:
    return os.environ.get("SAMURAI_TENANT_DATA_ENABLED", "").lower() in {"on", "1", "true", "yes"}


def _records_enabled() -> bool:
    """Raw record reads (customer rows, PII/CUI) have a SEPARATE gate from list/schema so
    they stay off until the data-residency question is signed off — even when the main
    tenant-data switch is on. See the PII/residency note in the module docstring."""
    return os.environ.get("SAMURAI_TENANT_RECORDS_ENABLED", "").lower() in {"on", "1", "true", "yes"}


def _api_base() -> str:
    return os.environ.get("VIRTUALDOJO_API_URL", "").rstrip("/")


def _host(url: str) -> str:
    """Bare host of a URL (no scheme/port/path), for env-match comparison."""
    s = url.split("://", 1)[-1]
    return s.split("/", 1)[0].split(":", 1)[0].lower()


def _warn_if_sso_env_mismatch() -> None:
    """The SSO token only validates against /api/v1 if the user signed in against the SAME
    backend that serves VIRTUALDOJO_API_URL (shared SECRET_KEY). If the bot's SSO points at
    a different host than the REST base, every read 403s with a confusing 'Could not validate
    credentials'. Log a loud hint instead of leaving that mystery."""
    try:
        from tools.virtualdojo_mcp import MCP_URL
        sso, api = _host(MCP_URL), _host(_api_base())
        if api and sso and sso != api:
            logger.warning(
                "[tenant_data] SSO host (%s) != VIRTUALDOJO_API_URL host (%s) — the SSO JWT is "
                "signed by a different backend and will 403 on /api/v1. Point the bot's "
                "VirtualDojo SSO (VIRTUALDOJO_MCP_URL) at the same host as VIRTUALDOJO_API_URL.",
                sso, api,
            )
    except Exception:
        pass


async def _vdj_get(token: str, path: str, params: dict | None = None,
                   tenant_id: str | None = None) -> dict:
    """Read-only GET to the VirtualDojo API. Returns {"data": <json>} or {"error": "..."}.
    Never raises. This is the ONLY verb used for data + schema."""
    base = _api_base()
    if not base:
        return {"error": "VIRTUALDOJO_API_URL is not configured"}
    import httpx

    headers = {"Authorization": f"Bearer {token}"}
    if tenant_id:
        headers["X-Tenant-ID"] = tenant_id
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{base}{path}", headers=headers, params=params or {})
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:300]}"}
        return {"data": resp.json()}
    except Exception as e:
        logger.warning("[tenant_data] GET %s failed: %s", path, e)
        return {"error": f"{type(e).__name__}: {e}"}


async def _vdj_post(token: str, path: str, json_body: dict | None = None) -> dict:
    """POST for impersonation SESSION LIFECYCLE ONLY (start/end). Never sends a
    ``confirm_write`` flag — it mints/ends a read token and mutates no customer data. This is
    the sole non-GET verb in the module; it must never be pointed at a data endpoint.
    ``json_body`` carries only session-lifecycle fields (e.g. the ``session_id`` /end requires);
    it is None for start. Returns {"data": <json>} or {"error": "..."}. Never raises."""
    base = _api_base()
    if not base:
        return {"error": "VIRTUALDOJO_API_URL is not configured"}
    assert "/impersonation/" in path, "tenant_data._vdj_post is for impersonation lifecycle only"
    import httpx

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{base}{path}", headers={"Authorization": f"Bearer {token}"},
                                     json=json_body)
        if resp.status_code not in (200, 201):
            return {"error": f"HTTP {resp.status_code}: {resp.text[:300]}"}
        return {"data": resp.json() if resp.text else {}}
    except Exception as e:
        logger.warning("[tenant_data] POST %s failed: %s", path, e)
        return {"error": f"{type(e).__name__}: {e}"}


async def _start_impersonation(token: str, grant_id: str) -> dict:
    """Start a 15-min read session as the granting customer user. Returns the start response
    ({"access_token", "target_user_email", "session_id", ...}) under "data", or {"error":...}.
    Backend-enforced: caller's tenant must equal the grant's target tenant AND have
    system_administrator, and the grant must be active/unexpired (else 403/400)."""
    out = await _vdj_post(token, f"/api/v1/impersonation/start/{grant_id}")
    if "error" in out:
        return out
    data = out.get("data") or {}
    if not data.get("access_token"):
        return {"error": "impersonation/start returned no access token"}
    return {"data": data}


async def _end_impersonation(imp_token: str, session_id: str | None = None) -> None:
    """Best-effort session close. The backend's /end requires the ``session_id`` from the start
    response; without it the call 422s. Failure is non-fatal (the token self-expires in 15 min),
    but passing session_id lets the session close immediately instead of lingering to TTL."""
    try:
        await _vdj_post(imp_token, "/api/v1/impersonation/end",
                        {"session_id": session_id} if session_id else None)
    except Exception:
        pass


def _audit(user_id: str, grant_id: str, target: str | None, obj: str | None,
           op: str, count) -> None:
    """Fine-grained per-read audit — METADATA ONLY (never row contents). The backend audits
    session start into the customer tenant but not per-endpoint reads, so this is our record."""
    logger.info(
        "[samurai.tenant_data_access] user=%s grant=%s target=%s object=%s op=%s count=%s",
        user_id, grant_id, target or "?", obj or "-", op, count,
    )


def _format_grants(items: list) -> str:
    if not items:
        return "No tenants currently have an active support grant authorized to us."
    lines = [f"{len(items)} tenant(s) with an active support grant:"]
    for g in items:
        name = g.get("granting_tenant_name") or g.get("tenant_id") or "unknown tenant"
        lines.append(
            f"• {name} — grant {g.get('id')} "
            f"(by {g.get('granting_user_email', '?')}, expires {g.get('expires_at', '?')})"
        )
    return "\n".join(lines)


def _format_schema(data, object_name: str | None) -> str:
    if object_name:
        fields = data.get("fields", data) if isinstance(data, dict) else data
        if not fields:
            return f"No schema returned for object '{object_name}'."
        rows = fields if isinstance(fields, list) else (
            fields.get("fields", []) if isinstance(fields, dict) else [])
        lines = [f"Schema for '{object_name}' ({len(rows)} fields):"]
        for f in rows:
            if isinstance(f, dict):
                lines.append(f"• {f.get('api_name') or f.get('name')}: "
                             f"{f.get('field_type') or f.get('type', '?')}")
        return "\n".join(lines)
    objs = data if isinstance(data, list) else data.get("items", [])
    if not objs:
        return "No objects found for that tenant."
    lines = [f"{len(objs)} objects in that tenant:"]
    for o in objs:
        if isinstance(o, dict):
            lines.append(f"• {o.get('api_name') or o.get('name')} — {o.get('label', '')}")
    return "\n".join(lines)


def _format_records(data: dict, object_name: str) -> tuple[str, int]:
    records = data.get("records", []) if isinstance(data, dict) else []
    total = (data.get("total_count") if isinstance(data, dict) else None)
    n = len(records)
    header = f"{n} record(s) from '{object_name}'" + (
        f" (of {total} total)" if total is not None else "") + ":"
    lines = [header]
    for r in records:
        if isinstance(r, dict):
            # Compact one-line preview per record — id + a few fields.
            preview = ", ".join(f"{k}={v}" for k, v in list(r.items())[:6])
            lines.append(f"• {preview}")
    return "\n".join(lines), n


class _ListGrantsInput(BaseModel):
    active_only: bool = Field(True, description="Only show currently-active (unexpired) grants.")


class _SchemaInput(BaseModel):
    grant_id: str = Field(description="The grant id (from list_tenant_support_grants) for the "
                                      "tenant to inspect.")
    object_name: str | None = Field(
        None, description="Optional object api_name to get its field schema; omit to list all "
                          "objects in the tenant.")


class _RecordsInput(BaseModel):
    grant_id: str = Field(description="The grant id (from list_tenant_support_grants) for the "
                                      "tenant to read from.")
    object_name: str = Field(description="The object api_name to read records from (e.g. "
                                         "'accounts', 'contacts', 'opportunities').")
    limit: int = Field(50, ge=1, le=_MAX_RECORD_LIMIT,
                       description=f"Max rows to return (1-{_MAX_RECORD_LIMIT}).")
    skip: int = Field(0, ge=0, description="Rows to skip (pagination).")


def create_tenant_data_tools(user_id: str) -> list:
    """Per-user read-only tenant-data tools, authenticated via the user's SSO session."""
    # Imported lazily to avoid a heavy import at module load + circulars.
    from tools.virtualdojo_mcp import (
        _get_access_token,
        is_user_authenticated,
        start_oauth_flow,
    )

    async def _signed_in_token() -> tuple[str | None, str | None]:
        """Return (token, prompt). On success token is set; otherwise prompt carries a real,
        clickable SSO sign-in link (built by start_oauth_flow, which also registers the PKCE
        state the OAuth callback needs). A background task has no signed-in user here, so it
        gets the prompt and never reaches the backend."""
        if not is_user_authenticated(user_id):
            try:
                login_url, _state = await start_oauth_flow(user_id)
            except Exception as e:
                logger.warning("[tenant_data] could not start SSO flow: %s", e)
                return None, ("Please sign in to VirtualDojo (SSO) first — I couldn't build a "
                              "sign-in link just now; try again in a moment.")
            return None, (
                f"You need to sign in to VirtualDojo first. "
                f"[Sign in to VirtualDojo CRM]({login_url})\n\n"
                f"After you sign in, ask me again and I'll pull the data."
            )
        token = await _get_access_token(user_id)
        if not token:
            return None, "Your VirtualDojo session expired — please sign in again."
        return token, None

    async def _list_tenant_support_grants(active_only: bool = True) -> str:
        if not _enabled():
            return "Tenant-data access is disabled (SAMURAI_TENANT_DATA_ENABLED is off)."
        _warn_if_sso_env_mismatch()
        token, prompt = await _signed_in_token()
        if not token:
            return prompt
        out = await _vdj_get(token, "/api/v1/impersonation/my-grants",
                             {"active_only": str(bool(active_only)).lower()})
        if "error" in out:
            return f"Could not list support grants: {out['error']}"
        data = out.get("data") or {}
        items = data.get("items", data if isinstance(data, list) else [])
        return _format_grants(items)

    async def _describe_tenant_schema(grant_id: str, object_name: str | None = None) -> str:
        if not _enabled():
            return "Tenant-data access is disabled (SAMURAI_TENANT_DATA_ENABLED is off)."
        _warn_if_sso_env_mismatch()
        token, prompt = await _signed_in_token()
        if not token:
            return prompt
        started = await _start_impersonation(token, grant_id)
        if "error" in started:
            return f"Could not start a read session for that grant: {started['error']}"
        info = started["data"]
        imp_token, target = info["access_token"], info.get("target_user_email")
        session_id = info.get("session_id")
        try:
            path = (f"/api/v1/schema/objects/{object_name}/schema" if object_name
                    else "/api/v1/schema/objects")
            out = await _vdj_get(imp_token, path)
        finally:
            await _end_impersonation(imp_token, session_id)
        if "error" in out:
            return f"Could not read schema: {out['error']}"
        text = _format_schema(out.get("data"), object_name)
        _audit(user_id, grant_id, target, object_name, "schema", "n/a")
        return text

    async def _read_tenant_records(grant_id: str, object_name: str,
                                   limit: int = 50, skip: int = 0) -> str:
        if not _enabled():
            return "Tenant-data access is disabled (SAMURAI_TENANT_DATA_ENABLED is off)."
        if not _records_enabled():
            return ("Reading raw tenant records is gated off (SAMURAI_TENANT_RECORDS_ENABLED) "
                    "pending data-residency sign-off — I can still list support grants and "
                    "describe a tenant's schema.")
        _warn_if_sso_env_mismatch()
        limit = max(1, min(int(limit), _MAX_RECORD_LIMIT))
        token, prompt = await _signed_in_token()
        if not token:
            return prompt
        started = await _start_impersonation(token, grant_id)
        if "error" in started:
            return f"Could not start a read session for that grant: {started['error']}"
        info = started["data"]
        imp_token, target = info["access_token"], info.get("target_user_email")
        session_id = info.get("session_id")
        try:
            out = await _vdj_get(imp_token, f"/api/v1/objects/{object_name}/records",
                                 {"skip": skip, "limit": limit})
        finally:
            await _end_impersonation(imp_token, session_id)
        if "error" in out:
            return f"Could not read records: {out['error']}"
        text, n = _format_records(out.get("data") or {}, object_name)
        _audit(user_id, grant_id, target, object_name, "records", n)
        return text

    list_grants = StructuredTool.from_function(
        coroutine=_list_tenant_support_grants,
        name="list_tenant_support_grants",
        description=(
            "List the customer tenants that have authorized a support grant to us (read-only, "
            "safe). Runs as YOU (the signed-in VirtualDojo user); if you're not signed in it "
            "returns an SSO sign-in link. Does NOT read any tenant's data — use the grant_id it "
            "returns with describe_tenant_schema / read_tenant_records to do that."
        ),
        args_schema=_ListGrantsInput,
    )
    describe_schema = StructuredTool.from_function(
        coroutine=_describe_tenant_schema,
        name="describe_tenant_schema",
        description=(
            "Read-only: list a granted tenant's objects, or the field schema of one object. "
            "Needs a grant_id from list_tenant_support_grants. Starts a short-lived support "
            "session as the granting user (you must be signed in via SSO). Returns metadata "
            "(object/field names), not customer records."
        ),
        args_schema=_SchemaInput,
    )
    read_records = StructuredTool.from_function(
        coroutine=_read_tenant_records,
        name="read_tenant_records",
        description=(
            "Read-only: fetch records from one object in a granted tenant (needs a grant_id "
            "from list_tenant_support_grants + the object api_name). Starts a short-lived "
            "support session as the granting user (you must be signed in via SSO). Returns "
            "REAL customer data — only do this when the user explicitly asks for that tenant's "
            "records; never autonomously."
        ),
        args_schema=_RecordsInput,
    )
    return [list_grants, describe_schema, read_records]
