"""Tests for read-only tenant-data tools (tools/tenant_data.py) — SSO/per-user model."""
import inspect
from unittest.mock import AsyncMock

import pytest

import tools.tenant_data as td


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("SAMURAI_TENANT_DATA_ENABLED", "SAMURAI_TENANT_RECORDS_ENABLED", "VIRTUALDOJO_API_URL"):
        monkeypatch.delenv(k, raising=False)
    yield


def _tools(user_id="u1"):
    return td.create_tenant_data_tools(user_id)


def _tool(user_id="u1"):
    return _tools(user_id)[0]


def _by_name(name, user_id="u1"):
    return next(t for t in _tools(user_id) if t.name == name)


def _sign_in(monkeypatch, token="user-jwt", records=True):
    monkeypatch.setenv("SAMURAI_TENANT_DATA_ENABLED", "on")
    if records:
        monkeypatch.setenv("SAMURAI_TENANT_RECORDS_ENABLED", "on")
    monkeypatch.setattr("tools.virtualdojo_mcp.is_user_authenticated", lambda uid: True)
    monkeypatch.setattr("tools.virtualdojo_mcp._get_access_token", AsyncMock(return_value=token))


# --- Phase 1: list grants -------------------------------------------------

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
    _sign_in(monkeypatch)
    payload = {"data": {"items": [
        {"id": "g1", "granting_tenant_name": "Acme Corp",
         "granting_user_email": "admin@acme.com", "expires_at": "2026-07-01"}
    ]}}
    monkeypatch.setattr(td, "_vdj_get", AsyncMock(return_value=payload))
    out = await _tool().ainvoke({})
    assert "Acme Corp" in out and "g1" in out


def test_format_grants_empty():
    assert "no tenants" in td._format_grants([]).lower()


# --- Phase 2: schema + records via impersonation --------------------------

def test_factory_exposes_three_tools():
    names = {t.name for t in _tools()}
    assert names == {"list_tenant_support_grants", "describe_tenant_schema", "read_tenant_records"}


async def test_schema_and_records_disabled_by_default():
    assert "disabled" in (await _by_name("describe_tenant_schema").ainvoke({"grant_id": "g1"})).lower()
    assert "disabled" in (await _by_name("read_tenant_records").ainvoke(
        {"grant_id": "g1", "object_name": "accounts"})).lower()


async def test_records_not_signed_in_is_barred(monkeypatch):
    # A background task has no signed-in user -> SSO prompt, never reaches the backend.
    monkeypatch.setenv("SAMURAI_TENANT_DATA_ENABLED", "on")
    monkeypatch.setenv("SAMURAI_TENANT_RECORDS_ENABLED", "on")
    monkeypatch.setattr("tools.virtualdojo_mcp.is_user_authenticated", lambda uid: False)
    monkeypatch.setattr("tools.virtualdojo_mcp.get_login_url", lambda uid: "https://sso.vdj/login")
    start = AsyncMock()
    monkeypatch.setattr(td, "_start_impersonation", start)
    out = await _by_name("read_tenant_records").ainvoke({"grant_id": "g1", "object_name": "accounts"})
    assert "sign in" in out.lower()
    start.assert_not_called()  # never started an impersonation session


async def test_records_gated_until_residency_flag(monkeypatch):
    # Main tenant-data flag ON, but records flag OFF -> records refuse, never sign in / start.
    monkeypatch.setenv("SAMURAI_TENANT_DATA_ENABLED", "on")  # records flag intentionally unset
    start = AsyncMock()
    monkeypatch.setattr(td, "_start_impersonation", start)
    out = await _by_name("read_tenant_records").ainvoke({"grant_id": "g1", "object_name": "accounts"})
    assert "residency" in out.lower()
    start.assert_not_called()


async def test_schema_works_without_records_flag(monkeypatch):
    # describe_tenant_schema runs on the main flag alone (no row PII, no records flag needed).
    _sign_in(monkeypatch, records=False)
    monkeypatch.setattr(td, "_start_impersonation", AsyncMock(return_value={
        "data": {"access_token": "imp-jwt", "target_user_email": "x@acme.com"}}))
    monkeypatch.setattr(td, "_end_impersonation", AsyncMock())
    monkeypatch.setattr(td, "_vdj_get", AsyncMock(return_value={
        "data": [{"api_name": "accounts", "label": "Accounts"}]}))
    out = await _by_name("describe_tenant_schema").ainvoke({"grant_id": "g1"})
    assert "accounts" in out


async def test_describe_schema_lists_objects(monkeypatch):
    _sign_in(monkeypatch)
    monkeypatch.setattr(td, "_start_impersonation", AsyncMock(return_value={
        "data": {"access_token": "imp-jwt", "target_user_email": "x@acme.com"}}))
    monkeypatch.setattr(td, "_end_impersonation", AsyncMock())
    monkeypatch.setattr(td, "_vdj_get", AsyncMock(return_value={
        "data": [{"api_name": "accounts", "label": "Accounts"}]}))
    out = await _by_name("describe_tenant_schema").ainvoke({"grant_id": "g1"})
    assert "accounts" in out


async def test_read_records_returns_rows(monkeypatch):
    _sign_in(monkeypatch)
    monkeypatch.setattr(td, "_start_impersonation", AsyncMock(return_value={
        "data": {"access_token": "imp-jwt", "target_user_email": "x@acme.com"}}))
    end = AsyncMock()
    monkeypatch.setattr(td, "_end_impersonation", end)
    monkeypatch.setattr(td, "_vdj_get", AsyncMock(return_value={
        "data": {"records": [{"id": "1", "name": "Acme"}], "total_count": 1}}))
    out = await _by_name("read_tenant_records").ainvoke({"grant_id": "g1", "object_name": "accounts"})
    assert "Acme" in out and "1 record" in out
    end.assert_awaited()  # session is always closed


async def test_read_records_surfaces_grant_403(monkeypatch):
    _sign_in(monkeypatch)
    monkeypatch.setattr(td, "_start_impersonation", AsyncMock(return_value={
        "error": "HTTP 403: This grant is not addressed to your tenant"}))
    out = await _by_name("read_tenant_records").ainvoke({"grant_id": "g1", "object_name": "accounts"})
    assert "could not start a read session" in out.lower() and "403" in out


# --- Auth header + read-only guarantees -----------------------------------

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


async def test_vdj_post_is_impersonation_only(monkeypatch):
    # The sole non-GET verb refuses to be aimed at a data endpoint.
    monkeypatch.setenv("VIRTUALDOJO_API_URL", "https://api.vdj")
    with pytest.raises(AssertionError):
        await td._vdj_post("t", "/api/v1/objects/accounts/records")


def test_read_only_by_construction():
    # confirm_write is never used as a code-level key/value (docstring prose may mention it).
    src = inspect.getsource(td)
    assert '"confirm_write"' not in src and "'confirm_write'" not in src
    # Data + schema are GET-only; _vdj_post is guarded to impersonation lifecycle.
    assert "/impersonation/" in inspect.getsource(td._vdj_post)
