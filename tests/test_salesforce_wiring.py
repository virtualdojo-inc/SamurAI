"""Wiring tests for the Salesforce case-management tools (tools/salesforce.py).

These guard the exact gap that shipped once: the tool file existed and deployed,
but was never imported into agent.py, so the agent could not call the tools and
`simple-salesforce` was missing from requirements. These tests fail loudly if the
tools are ever unbound, un-gated, or unreachable again.
"""
import pytest

from tools.salesforce import (
    SALESFORCE_TOOLS,
    query_cases,
    get_case_details,
    add_case_comment,
    update_case_status,
)

_READ_TOOLS = {"query_cases", "get_case_details"}
_WRITE_TOOLS = {"add_case_comment", "update_case_status"}
_ALL_SF_TOOLS = _READ_TOOLS | _WRITE_TOOLS


def test_salesforce_tools_exported():
    """SALESFORCE_TOOLS exports exactly the four case tools."""
    names = {t.name for t in SALESFORCE_TOOLS}
    assert names == _ALL_SF_TOOLS


def test_salesforce_tools_bound_to_agent():
    """The tools must be reachable by the agent's ToolNode (ALL_TOOLS)."""
    import agent

    bound = {t.name for t in agent.ALL_TOOLS if getattr(t, "name", None)}
    missing = _ALL_SF_TOOLS - bound
    assert not missing, f"Salesforce tools not bound to the agent: {sorted(missing)}"


def test_salesforce_group_selected_on_case_keywords():
    """A case-related message activates the salesforce tool group."""
    import agent

    selected = {t.name for t in agent._select_tool_groups("please close Salesforce case 00001009")}
    assert _ALL_SF_TOOLS <= selected


def test_read_tools_are_read_only_in_judge():
    import judge

    for name in _READ_TOOLS:
        assert name in judge.READ_ONLY_TOOL_NAMES
        assert name not in judge.WRITE_TOOL_NAMES


def test_write_tools_are_judge_gated():
    import judge

    for name in _WRITE_TOOLS:
        assert name in judge.WRITE_TOOL_NAMES
        assert name not in judge.READ_ONLY_TOOL_NAMES


def test_lazy_runtime_deps_are_installed():
    """salesforce.py imports simple_salesforce at module load; guard that CI
    (which pip-installs requirements.txt) has it declared — the gap that broke
    query_cases in prod once."""
    import simple_salesforce  # noqa: F401


def test_refresh_token_read_from_env(monkeypatch):
    """The token comes from the injected SF_CLI_REFRESH_TOKEN env var (matching
    every other secret in this service), not a Secret Manager API call."""
    import tools.salesforce as sf

    monkeypatch.setenv("SF_CLI_REFRESH_TOKEN", "tok-abc123")
    assert sf._get_refresh_token() == "tok-abc123"


def test_refresh_token_missing_raises_clear_error(monkeypatch):
    import tools.salesforce as sf

    monkeypatch.delenv("SF_CLI_REFRESH_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="SF_CLI_REFRESH_TOKEN"):
        sf._get_refresh_token()


def test_update_case_status_passes_id_and_data_separately(monkeypatch):
    """Regression: simple_salesforce's SFType.update needs (record_id, data).
    Passing only the dict raised 'missing 1 required positional argument: data'
    in prod when the user tried to close cases. Also: update returns an int HTTP
    status code (not a dict), and the Id belongs in the URL, not the body."""
    import tools.salesforce as sf_mod

    captured = {}

    class FakeCase:
        def update(self, record_id, data):
            captured["record_id"] = record_id
            captured["data"] = data
            return 204  # simple_salesforce returns the HTTP status code

    class FakeSF:
        Case = FakeCase()

    monkeypatch.setattr(sf_mod, "_create_sf_connection", lambda: FakeSF())

    out = update_case_status.invoke({
        "case_id": "500XX0000000abc",  # starts with 500 -> skips CaseNumber lookup
        "new_status": "Closed",
        "close_case": True,
        "closure_notes": "resolved",
    })

    assert captured["record_id"] == "500XX0000000abc"  # Id in the URL
    assert "Id" not in captured["data"]                # not duplicated in the body
    assert captured["data"]["Status"] == "Closed"
    assert captured["data"]["ClosureNotes"] == "resolved"
    assert "updated to status: Closed" in out          # int 204 handled as success


def test_add_case_comment_uses_casecomment_object(monkeypatch):
    """Regression: a case comment must use the CaseComment object
    (ParentId/CommentBody/IsPublished), not the legacy Note object, and
    is_internal must drive IsPublished (internal => not published)."""
    import tools.salesforce as sf_mod

    captured = {}

    class FakeCaseComment:
        def create(self, data):
            captured["data"] = data
            return {"id": "00aXX0000001", "success": True, "errors": []}

    class FakeNote:
        def create(self, *a, **k):
            raise AssertionError("must not use Note for case comments")

    class FakeSF:
        CaseComment = FakeCaseComment()
        Note = FakeNote()

    monkeypatch.setattr(sf_mod, "_create_sf_connection", lambda: FakeSF())

    out = add_case_comment.invoke({
        "case_id": "500XX0000000abc",
        "comment": "Closing per customer request.",
        "is_internal": True,
    })

    assert captured["data"]["ParentId"] == "500XX0000000abc"
    assert captured["data"]["CommentBody"] == "Closing per customer request."
    assert captured["data"]["IsPublished"] is False  # internal => not published
    assert "internal comment" in out
    assert "CaseComment ID: 00aXX0000001" in out


def test_query_cases_description_owns_case_routing():
    """query_cases must clearly own 'Salesforce case' requests. Prod misrouted
    'list the quotely cases from salesforce' to list_tenant_support_grants (the
    CRM SSO flow); the description now names those phrasings explicitly."""
    desc = query_cases.description.lower()
    assert "salesforce" in desc and "case" in desc
    assert "quotely" in desc  # 'quotely cases' must route here, not to the CRM tool


def test_tenant_grant_tool_disclaims_salesforce_cases():
    """The tenant support-grant tool must steer case requests to query_cases so
    the model stops sending users through SSO for a Salesforce query."""
    from tools.tenant_data import create_tenant_data_tools

    tools = create_tenant_data_tools("test-user")
    grants = next(t for t in tools if t.name == "list_tenant_support_grants")
    desc = grants.description.lower()
    assert "not for salesforce cases" in desc
    assert "query_cases" in desc


def test_salesforce_prompt_section_loads_for_case_requests():
    """The system prompt frames SamurAI as a 'CRM assistant' and told the model
    to send un-signed-in users to 'connect to VirtualDojo' for data requests —
    which is why 'list the cases from salesforce' emitted an SSO link instead of
    calling query_cases. A dedicated Salesforce prompt section must load for case
    requests and steer AWAY from the CRM/tenant path."""
    import agent

    prompt = agent._select_prompt_sections("list the cases from salesforce")
    assert "Salesforce support cases" in prompt
    assert "query_cases" in prompt
    assert "connect to virtualdojo" in prompt.lower()  # explicitly tells it NOT to

    # It should also load for a plain 'cases' request with no 'salesforce' word.
    assert "Salesforce support cases" in agent._select_prompt_sections("please list the open cases")
