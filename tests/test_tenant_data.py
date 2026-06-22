"""Tests for read-only tenant-data tools (tools/tenant_data.py) — SSO/per-user model."""
from unittest.mock import AsyncMock

import pytest

import tools.tenant_data as td


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("SAMURAI_TENANT_DATA_ENABLED", "VIRTUALDOJO_API_URL"):
        monkeypatch.delenv(k, raising=False)
    yield


def _tool(user_id="u1"):
    return td.create_tenant_data_tools(user_id)[0]


async def test_disabled_by_default():
    out = await _tool().ainvoke({})
    assert "disabled" in out.lower()


async def test_not_signed_in_returns_sso_link(monkeypatch):
    monkeypatch.setenv("SAMURAI_TENANT_DATA_ENABLED", "on")
    monkeypatch.setattr("tools.virtualdojo_mcp.is_user_authenticated", lambda uid: False)
    monkeypatch.setattr("tools.virtualdojo_mcp.get_login_url", lambda uid: "https://sso.vdj/login")
    out = await _tool().ainvoke({})
    assert "sign in" in out.lower() and "https://sso.vdj/login" in out


async def test_session_expired(monkeypatch):
    monkeypatch.setenv("SAMURAI_TENANT_DATA_ENABLED", "on")
    monkeypatch.setattr("tools.virtualdojo_mcp.is_user_authenticated", lambda uid: True)
    monkeypatch.setattr("tools.virtualdojo_mcp._get_access_token", AsyncMock(return_value=None))
    out = await _tool().ainvoke({})
    assert "expired" in out.lower()


async def test_list_grants_success_uses_user_session(monkeypatch):
    monkeypatch.setenv("SAMURAI_TENANT_DATA_ENABLED", "on")
    monkeypatch.setattr("tools.virtualdojo_mcp.is_user_authenticated", lambda uid: True)
    monkeypatch.setattr("tools.virtualdojo_mcp._get_access_token", AsyncMock(return_value="user-jwt"))
    payload = {"data": {"items": [
        {"id": "g1", "granting_tenant_name": "Acme Corp",
         "granting_user_email": "admin@acme.com", "expires_at": "2026-07-01"}
    ]}}
    monkeypatch.setattr(td, "_vdj_get", AsyncMock(return_value=payload))
    out = await _tool().ainvoke({})
    assert "Acme Corp" in out and "g1" in out


def test_format_grants_empty():
    assert "no tenants" in td._format_grants([]).lower()


async def test_vdj_get_uses_user_bearer_token(monkeypatch):
    monkeypatch.setenv("VIRTUALDOJO_API_URL", "https://api.vdj")
    captured = {}

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"items": []}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None, params=None):
            captured.update(url=url, headers=headers)
            return _Resp()

    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: _Client())
    out = await td._vdj_get("user-jwt", "/api/v1/impersonation/my-grants",
                            {"active_only": "true"}, tenant_id="t1")
    assert out["data"] == {"items": []}
    assert captured["headers"]["Authorization"] == "Bearer user-jwt"  # the user's SSO JWT
    assert captured["headers"]["X-Tenant-ID"] == "t1"


def test_read_only_by_construction():
    # No write HTTP helper exists in this module.
    assert not hasattr(td, "_vdj_post")
