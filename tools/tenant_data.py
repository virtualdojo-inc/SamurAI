"""Read-only tenant-data tools via VirtualDojo support grants.

AUTH: reuses SamurAI's EXISTING per-user VirtualDojo SSO sign-in (tools/virtualdojo_mcp:
get_login_url / token store). Reads run AS the signed-in user — their identity and their
``system_administrator`` rights in the support tenant. No service account, no API key.
If the user isn't signed in, the tool returns the SSO sign-in prompt. Because a background
task has no signed-in user, autonomous / scheduled reads of tenant data are barred for free.

READ-ONLY: only GET endpoints (and, later, SELECT-only SQL) — there is deliberately no
write path in this module and no ``confirm_write``. Gated by ``SAMURAI_TENANT_DATA_ENABLED``
(off by default). Endpoints mirror the virtualdojo CLI's read subset:
  - GET /api/v1/impersonation/my-grants            (which tenants granted us)
  - GET /api/v1/impersonation/start/{grant_id}     (Phase 2 — start a read session)
  - GET /api/v1/objects/{object}/records , /schema  (Phase 2 — the actual reads)

Per-user factory, like create_virtualdojo_tool. See docs/tenant_data_access_plan.md.
"""
from __future__ import annotations

import logging
import os

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return os.environ.get("SAMURAI_TENANT_DATA_ENABLED", "").lower() in {"on", "1", "true", "yes"}


def _api_base() -> str:
    return os.environ.get("VIRTUALDOJO_API_URL", "").rstrip("/")


async def _vdj_get(token: str, path: str, params: dict | None = None,
                   tenant_id: str | None = None) -> dict:
    """Read-only GET to the VirtualDojo API as the signed-in user. The ONLY HTTP verb
    this module uses. Returns {"data": <json>} or {"error": "..."}. Never raises."""
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


class _ListGrantsInput(BaseModel):
    active_only: bool = Field(True, description="Only show currently-active (unexpired) grants.")


def create_tenant_data_tools(user_id: str) -> list:
    """Per-user read-only tenant-data tools, authenticated via the user's SSO session."""
    # Imported lazily to avoid a heavy import at module load + circulars.
    from tools.virtualdojo_mcp import (
        _get_access_token,
        get_login_url,
        is_user_authenticated,
    )

    async def _signed_in_token() -> tuple[str | None, str | None]:
        """Return (token, prompt). On success token is set; otherwise prompt explains
        the SSO sign-in the user must complete first."""
        if not is_user_authenticated(user_id):
            url = get_login_url(user_id)
            return None, (f"You need to sign in to VirtualDojo (SSO) first: {url}" if url
                          else "Please sign in to VirtualDojo (SSO) first.")
        token = await _get_access_token(user_id)
        if not token:
            return None, "Your VirtualDojo session expired — please sign in again."
        return token, None

    async def _list_tenant_support_grants(active_only: bool = True) -> str:
        if not _enabled():
            return "Tenant-data access is disabled (SAMURAI_TENANT_DATA_ENABLED is off)."
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

    list_grants = StructuredTool.from_function(
        coroutine=_list_tenant_support_grants,
        name="list_tenant_support_grants",
        description=(
            "List the customer tenants that have authorized a support grant to us (read-only, "
            "safe). Runs as YOU (the signed-in VirtualDojo user); if you're not signed in it "
            "returns an SSO sign-in link. Does NOT read any tenant's data — that's a separate, "
            "explicitly-approved step."
        ),
        args_schema=_ListGrantsInput,
    )
    return [list_grants]
