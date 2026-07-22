"""Tests for agent.py — LangGraph agent graph and run_agent()."""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_llm():
    """Patch ChatGoogleGenerativeAI and memory deps before importing agent."""
    with (
        patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_cls,
        patch("memory.get_checkpointer", new_callable=AsyncMock) as mock_ckpt,
        patch(
            "memory.retrieve_relevant_memories",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("memory.create_memory_tools", return_value=[]),
        patch("memory.get_memory_store", return_value=MagicMock()),
        patch("memory.get_background_extractor", return_value=MagicMock()),
        patch("memory.get_core_extractor", return_value=MagicMock()),
        patch("memory.get_team_extractor", return_value=MagicMock()),
        patch("memory.persist_memories"),
    ):
        mock_instance = MagicMock()
        mock_instance.bind_tools.return_value = mock_instance
        mock_instance.ainvoke = AsyncMock(
            return_value=MagicMock(content="Hello from SamurAI!", tool_calls=[])
        )
        mock_cls.return_value = mock_instance

        from langgraph.checkpoint.memory import MemorySaver

        mock_ckpt.return_value = MemorySaver()

        import agent

        importlib.reload(agent)
        agent._user_graphs.clear()
        yield mock_instance, agent


def test_static_tools_list(mock_llm):
    _, agent = mock_llm
    assert len(agent.STATIC_TOOLS) == len(agent.ALL_TOOLS)
    assert len(agent.ALL_TOOLS) == 101  # Static tools (CRM/memory/tenant-data added per-user)
    tool_names = {t.name for t in agent.STATIC_TOOLS}
    assert "query_cloud_logs" in tool_names
    assert "run_code" in tool_names
    assert "find_prior_script" in tool_names
    assert "analyze_loom_video" in tool_names
    assert "get_skill" in tool_names
    assert "read_knowledge" in tool_names
    assert "search_wiki" in tool_names
    assert "get_tracker_diagnostics" in tool_names
    assert "list_cloud_run_services" in tool_names
    assert "check_gcp_metrics" in tool_names
    assert "github_list_prs" in tool_names
    assert "github_get_pr_details" in tool_names
    assert "github_list_recent_commits" in tool_names
    assert "github_list_issues" in tool_names
    assert "github_search_issues" in tool_names
    assert "github_get_issue_details" in tool_names
    assert "github_create_issue" in tool_names
    assert "github_get_issue_type" in tool_names
    assert "github_set_issue_type" in tool_names
    assert "github_list_workflow_runs" in tool_names
    assert "github_get_workflow_run_details" in tool_names
    # Social media tools
    assert "social_generate_image" in tool_names
    assert "social_preview_post" in tool_names
    assert "social_publish_post" in tool_names
    assert "social_schedule_post" in tool_names
    assert "social_list_scheduled" in tool_names
    assert "social_get_post" in tool_names
    assert "social_update_post" in tool_names
    assert "social_delete_post" in tool_names
    # Google search
    assert "google_search" in tool_names
    # Background task tools
    assert "create_background_task" in tool_names
    assert "list_background_tasks" in tool_names
    assert "pause_background_task" in tool_names
    assert "resume_background_task" in tool_names
    assert "cancel_background_task" in tool_names
    # Teams messaging tools
    assert "send_teams_message" in tool_names
    assert "lookup_team_member" in tool_names
    assert "list_team_members" in tool_names
    # FedRAMP compliance tools
    assert "fedramp_collect_evidence" in tool_names
    assert "fedramp_evidence_summary" in tool_names
    assert "fedramp_daily_log_review" in tool_names
    assert "fedramp_check_scc_findings" in tool_names
    # FedRAMP doc tools
    assert "fedramp_read_document" in tool_names
    assert "fedramp_propose_edit" in tool_names
    assert "fedramp_review_code" in tool_names
    # OSCAL tools
    assert "oscal_generate_ssp" in tool_names
    assert "oscal_validate_package" in tool_names
    assert "oscal_catalog_lookup" in tool_names
    assert "oscal_render_pdf" in tool_names
    # Repo sync tools
    assert "sync_repo" in tool_names
    assert "read_repo_file" in tool_names
    assert "read_repo_file_range" in tool_names
    assert "search_repo_code" in tool_names
    assert "list_repo_files" in tool_names
    # Investigate sub-agent
    assert "investigate" in tool_names
    # Troubleshooting DB tools
    assert "save_troubleshooting_step" in tool_names
    assert "search_troubleshooting" in tool_names
    assert "delete_troubleshooting_step" in tool_names
    # GitHub close issue
    assert "github_close_issue" in tool_names
    # GitHub edit issue
    assert "github_edit_issue" in tool_names


def test_every_static_tool_has_a_friendly_label():
    """Every bound tool needs an explicit Teams status label in run_agent's
    _tool_labels, so a tool call shows a friendly name (not a humanized
    fallback). Add a label to _tool_labels whenever you add a tool."""
    import inspect
    import re

    import agent

    src = inspect.getsource(agent)
    block = src[src.index("_tool_labels = {"):]
    block = block[: block.index("\n    }")]
    labeled = set(re.findall(r'"([a-z0-9_]+)":', block))
    missing = sorted(t.name for t in agent.ALL_TOOLS if t.name not in labeled)
    assert not missing, f"tools missing a friendly label in _tool_labels: {missing}"


def test_build_human_content_multimodal(mock_llm):
    _, agent = mock_llm
    # No images -> plain string (unchanged default path)
    assert agent._build_human_content("hi", None) == "hi"
    # Images -> text block + image data-content-block(s)
    out = agent._build_human_content("describe", [{"data": "QUFB", "mime_type": "image/png"}])
    assert out[0] == {"type": "text", "text": "describe"}
    assert out[1] == {"type": "image", "base64": "QUFB", "mime_type": "image/png"}
    # Malformed image entries (missing data/mime) are dropped
    assert agent._build_human_content("x", [{"mime_type": "image/png"}]) == [
        {"type": "text", "text": "x"}
    ]


def test_loom_share_url_selects_loom_tool(mock_llm):
    """A pasted bare Loom URL makes analyze_loom_video available (no keyword needed)."""
    _, agent = mock_llm
    tools = agent._select_tool_groups(
        "https://www.loom.com/share/9614dd0b62e5475985d0b021ee3f33d4"
    )
    assert "analyze_loom_video" in {t.name for t in tools}


def test_system_prompt_defined(mock_llm):
    _, agent = mock_llm
    assert "SamurAI" in agent.SYSTEM_PROMPT
    assert "DevOps" in agent.SYSTEM_PROMPT
    assert "VirtualDojo CRM" in agent.SYSTEM_PROMPT
    assert "Long-term Memory" in agent.SYSTEM_PROMPT
    assert "manage_memory" in agent.SYSTEM_PROMPT
    # Autonomous agent capabilities
    assert "FULLY AUTONOMOUS" in agent.SYSTEM_PROMPT
    assert "Background Tasks" in agent.SYSTEM_PROMPT or "background_task" in agent.SYSTEM_PROMPT
    assert "AUTONOMY RULES" in agent.SYSTEM_PROMPT
    assert "send_teams_message" in agent.SYSTEM_PROMPT
    # FedRAMP system prompt
    assert "FedRAMP" in agent.SYSTEM_PROMPT
    assert "OSCAL" in agent.SYSTEM_PROMPT
    assert "FR2615441197" in agent.SYSTEM_PROMPT
    assert "virtualdojo-inc/Fedramp" in agent.SYSTEM_PROMPT
    assert "fedramp_collect_evidence" in agent.SYSTEM_PROMPT or "fedramp_evidence_summary" in agent.SYSTEM_PROMPT
    # Step budget guidance (prevents recursion limit exhaustion)
    assert "STEP BUDGET" in agent.SYSTEM_PROMPT
    assert "2-4 tool calls" in agent.SYSTEM_PROMPT
    assert "memory retrieval and extraction happen automatically" in agent.SYSTEM_PROMPT


def test_system_prompt_troubleshooting_hardening(mock_llm):
    """Tier 1 prompt edits: hypothesis discipline, parallel investigate dispatch,
    duplicate-implementation check, no hard cap for troubleshooting."""
    _, agent = mock_llm
    prompt = agent.SYSTEM_PROMPT
    # Hypothesis discipline — state 2-3 hypotheses before investigating
    assert "hypotheses" in prompt.lower()
    # Parallel investigate() dispatch is the speed lever
    assert "investigate(" in prompt
    assert "PARALLEL INVESTIGATION" in prompt
    assert "same turn" in prompt.lower() or "same wall time" in prompt.lower()
    # Duplicate implementation check (the activities bug class)
    assert "DUPLICATE IMPLEMENTATION" in prompt
    # No hard tool-call cap for troubleshooting
    assert "no hard cap" in prompt.lower()
    # Issue-search-first is the Phase 1 knowledge-base retrieval lever
    assert "github_search_issues" in prompt
    assert "ISSUE SEARCH FIRST" in prompt
    # Phase 2: autonomous save at the end of a successful bug hunt
    assert "save_troubleshooting_step" in prompt
    assert "SAVE THE PATTERN" in prompt


def test_select_tool_groups_core_only(mock_llm):
    """Simple query should get ONLY core tools — no fallback groups."""
    _, agent = mock_llm
    tools = agent._select_tool_groups("check the logs for errors")
    names = {t.name for t in tools}
    assert "query_cloud_logs" in names
    assert "list_cloud_run_services" in names
    # No fallback — GitHub should NOT be loaded for simple log queries
    assert "github_list_issues" not in names
    # Background-task/scheduling tools are now ALWAYS available (moved to core)
    # so the agent can schedule recurring jobs regardless of phrasing.
    assert "create_background_task" in names
    # File tools should NOT be loaded for a simple log query
    assert "get_spreadsheet_info" not in names
    # FedRAMP/OSCAL should NOT be loaded
    assert "fedramp_collect_evidence" not in names
    assert "oscal_generate_ssp" not in names
    assert "social_preview_post" not in names


def test_select_tool_groups_fedramp(mock_llm):
    """FedRAMP query should load fedramp tools."""
    _, agent = mock_llm
    tools = agent._select_tool_groups("check the fedramp compliance status")
    names = {t.name for t in tools}
    assert "fedramp_collect_evidence" in names
    assert "fedramp_evidence_summary" in names
    assert "query_cloud_logs" in names  # core always loaded


def test_select_tool_groups_oscal(mock_llm):
    """OSCAL query should load OSCAL tools."""
    _, agent = mock_llm
    tools = agent._select_tool_groups("generate the OSCAL SSP")
    names = {t.name for t in tools}
    assert "oscal_generate_ssp" in names
    assert "oscal_validate_package" in names


def test_select_tool_groups_troubleshoot(mock_llm):
    """Troubleshoot query should load repo sync tools."""
    _, agent = mock_llm
    tools = agent._select_tool_groups("troubleshoot the production errors")
    names = {t.name for t in tools}
    assert "sync_repo" in names
    assert "read_repo_file" in names
    assert "search_repo_code" in names


def test_select_tool_groups_loads_investigate_on_troubleshoot(mock_llm):
    """Troubleshoot / root-cause / why-is queries should also load the investigate
    sub-agent, since it lives in the repo tool group."""
    _, agent = mock_llm
    for msg in [
        "troubleshoot the production errors",
        "why is the activities endpoint broken",
        "investigate the timeout errors",
        "find the root cause of the 500s",
        "this traceback keeps showing up",
    ]:
        names = {t.name for t in agent._select_tool_groups(msg)}
        assert "investigate" in names, f"investigate not loaded for: {msg!r}"
        # github_search_issues is in the repo group so it loads on
        # troubleshooting even when no github keyword is present.
        assert (
            "github_search_issues" in names
        ), f"github_search_issues not loaded for: {msg!r}"


def test_select_tool_groups_investigate_not_loaded_for_simple(mock_llm):
    """Simple non-troubleshoot queries should NOT pull in the investigate tool."""
    _, agent = mock_llm
    for msg in [
        "check the logs",
        "list cloud run services",
        "show me the open PRs",
        "send a message to Cyrus",
    ]:
        names = {t.name for t in agent._select_tool_groups(msg)}
        assert "investigate" not in names, f"investigate wrongly loaded for: {msg!r}"


def test_select_tool_groups_fix_issue_loads_localization(mock_llm):
    """The controlled issue-fix flow ('fix issue' / 'attempt a fix' / 'fix plan')
    must load the read-only localization tools so SamurAI can build a brief."""
    _, agent = mock_llm
    for msg in [
        "fix issue 123 in the data service",
        "attempt a fix for issue #5",
        "create the fix plan for issue 9",
    ]:
        names = {t.name for t in agent._select_tool_groups(msg)}
        assert "sync_repo" in names, f"sync_repo not loaded for: {msg!r}"
        assert "search_repo_code" in names, f"search_repo_code not loaded for: {msg!r}"
        assert "investigate" in names, f"investigate not loaded for: {msg!r}"
        assert (
            "github_search_issues" in names
        ), f"github_search_issues not loaded for: {msg!r}"


def test_select_tool_groups_social(mock_llm):
    """Social media query should load social tools."""
    _, agent = mock_llm
    tools = agent._select_tool_groups("draft a linkedin post")
    names = {t.name for t in tools}
    assert "social_preview_post" in names
    # Should NOT load repo/fedramp
    assert "sync_repo" not in names
    assert "fedramp_collect_evidence" not in names


def test_select_tool_groups_multiple(mock_llm):
    """Query touching multiple groups should load all relevant groups."""
    _, agent = mock_llm
    tools = agent._select_tool_groups("check github issues and fedramp compliance")
    names = {t.name for t in tools}
    assert "github_list_issues" in names
    assert "fedramp_collect_evidence" in names
    assert "query_cloud_logs" in names  # core


def test_select_tool_groups_dedupes_overlapping_tools(mock_llm):
    """Tools registered in multiple groups must appear only once when both
    groups activate. Gemini rejects duplicate function declarations with
    400 INVALID_ARGUMENT, which previously broke any message hitting both
    the github and repo groups (e.g. "investigate the github issue ...")."""
    _, agent = mock_llm
    tools = agent._select_tool_groups("investigate the github issue with the data service")
    names = [t.name for t in tools]
    assert names.count("github_search_issues") == 1


def test_select_tool_groups_much_smaller_than_all(mock_llm):
    """Dynamic selection should return significantly fewer tools than ALL_TOOLS."""
    _, agent = mock_llm
    simple_tools = agent._select_tool_groups("check the logs")
    assert len(simple_tools) < len(agent.ALL_TOOLS) / 2


def test_select_tool_groups_background_tasks(mock_llm):
    """Background task keywords should load task tools."""
    _, agent = mock_llm
    tools = agent._select_tool_groups("create a background task to remind me")
    names = {t.name for t in tools}
    assert "create_background_task" in names
    assert "list_background_tasks" in names
    assert "pause_background_task" in names


def test_select_tool_groups_files(mock_llm):
    """File keywords should load file handler tools."""
    _, agent = mock_llm
    tools = agent._select_tool_groups("fill the spreadsheet column")
    names = {t.name for t in tools}
    assert "get_spreadsheet_info" in names
    assert "fill_spreadsheet_column" in names
    assert "edit_spreadsheet" in names


def test_select_tool_groups_file_deployed_no_file_tools(mock_llm):
    """'file deployed' should NOT trigger file handler tools — 'file' is too generic."""
    _, agent = mock_llm
    tools = agent._select_tool_groups("was the config file deployed to main")
    names = {t.name for t in tools}
    assert "get_spreadsheet_info" not in names
    assert "fill_spreadsheet_column" not in names


def test_select_tool_groups_memory_without_tools(mock_llm):
    """Memory keywords without memory_tools arg should not crash."""
    _, agent = mock_llm
    tools = agent._select_tool_groups("do you remember my preferences")
    names = {t.name for t in tools}
    # Core tools should still be there
    assert "query_cloud_logs" in names


def test_select_tool_groups_memory_with_tools(mock_llm):
    """Memory keywords with memory_tools should include them."""
    _, agent = mock_llm
    mock_mem_tool = MagicMock()
    mock_mem_tool.name = "manage_memory"
    tools = agent._select_tool_groups(
        "do you remember what I told you last time",
        memory_tools=[mock_mem_tool],
    )
    names = {t.name for t in tools}
    assert "manage_memory" in names
    assert "query_cloud_logs" in names


def test_select_tool_groups_no_memory_for_logs(mock_llm):
    """GCP log queries should NOT get memory tools even if passed."""
    _, agent = mock_llm
    mock_mem_tool = MagicMock()
    mock_mem_tool.name = "search_memory"
    tools = agent._select_tool_groups(
        "check the production logs for errors",
        memory_tools=[mock_mem_tool],
    )
    names = {t.name for t in tools}
    assert "query_cloud_logs" in names
    # Memory tools should NOT be included — no memory keywords
    assert "search_memory" not in names


def test_select_tool_groups_github_only_when_keyword_matches(mock_llm):
    """GitHub tools should only load when github keywords match."""
    _, agent = mock_llm
    # No github keywords
    tools_no_gh = agent._select_tool_groups("check the logs")
    names_no_gh = {t.name for t in tools_no_gh}
    assert "github_list_prs" not in names_no_gh

    # With github keywords
    tools_gh = agent._select_tool_groups("show me the pull requests")
    names_gh = {t.name for t in tools_gh}
    assert "github_list_prs" in names_gh


def test_needs_pro_model_for_oscal(mock_llm):
    _, agent = mock_llm
    from langchain_core.messages import HumanMessage

    assert agent._needs_pro_model([HumanMessage(content="generate the OSCAL SSP")])
    assert agent._needs_pro_model([HumanMessage(content="review code in main.py")])
    assert agent._needs_pro_model([HumanMessage(content="update the SSP control AC-2")])
    assert agent._needs_pro_model([HumanMessage(content="migrate the ConMon SOP to OSCAL")])
    assert agent._needs_pro_model([HumanMessage(content="validate package")])
    assert agent._needs_pro_model([HumanMessage(content="render PDF of the SSP")])
    assert agent._needs_pro_model([HumanMessage(content="propose edit to the IR plan")])
    assert agent._needs_pro_model([HumanMessage(content="look up control AC-2")])


def test_needs_pro_model_handles_multimodal_content(mock_llm):
    """Image turns give HumanMessage a list content (text + image blocks). The
    keyword check must read the text block, not crash on `.lower()` of a list.
    Regression for the paste-a-screenshot AttributeError."""
    _, agent = mock_llm
    from langchain_core.messages import HumanMessage

    multimodal = [
        {"type": "text", "text": "review code in main.py"},
        {"type": "image", "base64": "AAAA", "mime_type": "image/png"},
    ]
    assert agent._needs_pro_model([HumanMessage(content=multimodal)]) is True
    plain_img = [{"type": "image", "base64": "AAAA", "mime_type": "image/png"}]
    assert agent._needs_pro_model([HumanMessage(content=plain_img)]) is False
    # _text_of extracts the text block(s) and ignores images
    assert agent._text_of(multimodal) == "review code in main.py"
    assert agent._text_of("plain string") == "plain string"


def test_needs_pro_model_for_fix_issue(mock_llm):
    """Controlled issue-fix phrasings should route to the Pro model."""
    _, agent = mock_llm
    from langchain_core.messages import HumanMessage

    assert agent._needs_pro_model([HumanMessage(content="fix issue 123")])
    assert agent._needs_pro_model([HumanMessage(content="attempt a fix for issue 5")])


def test_needs_pro_model_for_troubleshooting(mock_llm):
    _, agent = mock_llm
    from langchain_core.messages import HumanMessage

    assert agent._needs_pro_model([HumanMessage(content="troubleshoot the 500 errors on prod")])
    assert agent._needs_pro_model([HumanMessage(content="debug the auth failure in the API")])
    assert agent._needs_pro_model([HumanMessage(content="why is the deployment failing?")])
    assert agent._needs_pro_model([HumanMessage(content="investigate the timeout errors")])
    assert agent._needs_pro_model([HumanMessage(content="diagnose the memory leak")])
    assert agent._needs_pro_model([HumanMessage(content="the API is broken, what's wrong?")])
    assert agent._needs_pro_model([HumanMessage(content="analyze code in config.py for the bug")])
    assert agent._needs_pro_model([HumanMessage(content="I see a traceback in the logs")])
    # Operational / log-analysis phrasings — added 2026-05 because Flash
    # was being chosen for these and quality suffered.
    assert agent._needs_pro_model([HumanMessage(content="check the logs for errors")])
    assert agent._needs_pro_model([HumanMessage(content="review the samurai gcloud logs")])
    assert agent._needs_pro_model([HumanMessage(content="list cloud run services")])
    assert agent._needs_pro_model([HumanMessage(content="any errors lately?")])
    assert agent._needs_pro_model([HumanMessage(content="what happened during the outage?")])


def test_needs_flash_model_for_simple_queries(mock_llm):
    _, agent = mock_llm
    from langchain_core.messages import HumanMessage

    assert not agent._needs_pro_model([HumanMessage(content="show open PRs")])
    assert not agent._needs_pro_model([HumanMessage(content="send a message to Cyrus")])
    assert not agent._needs_pro_model([HumanMessage(content="list my background tasks")])
    assert not agent._needs_pro_model([HumanMessage(content="what's the fedramp status")])
    assert not agent._needs_pro_model([HumanMessage(content="collect evidence for AC")])


def test_needs_pro_model_empty_messages(mock_llm):
    _, agent = mock_llm
    assert not agent._needs_pro_model([])


def test_pro_model_keywords_defined(mock_llm):
    _, agent = mock_llm
    assert isinstance(agent.PRO_MODEL_KEYWORDS, list)
    assert len(agent.PRO_MODEL_KEYWORDS) > 0
    assert "oscal" in agent.PRO_MODEL_KEYWORDS


def test_drop_empty_messages_strips_zero_parts_only():
    """A persisted empty AIMessage (or whitespace human) 400s the whole turn
    (Gemini: 'must include at least one parts field'). _drop_empty_messages must
    remove exactly those, while keeping tool-call / tool-result / system / real
    messages — even when their string content is empty."""
    import agent
    from langchain_core.messages import (
        AIMessage, HumanMessage, SystemMessage, ToolMessage,
    )

    msgs = [
        SystemMessage(content=""),                        # keep (system, handled separately)
        HumanMessage(content="hello"),                    # keep
        AIMessage(content=""),                            # DROP (empty str, no tool_calls)
        HumanMessage(content="   \n "),                   # DROP (whitespace-only)
        AIMessage(content=[]),                            # DROP (empty list / multimodal)
        HumanMessage(content=[{"type": "text", "text": ""}]),  # DROP (all-empty-text list)
        HumanMessage(content=[                            # keep (has an image part)
            {"type": "text", "text": ""},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]),
        AIMessage(content="", tool_calls=[
            {"name": "t", "args": {}, "id": "1"}]),       # keep (function_call parts)
        ToolMessage(content="", tool_call_id="1"),        # keep (function_response part)
        AIMessage(content="done"),                        # keep
    ]
    out = agent._drop_empty_messages(msgs)

    kept = {(type(m).__name__, str(m.content)[:20]) for m in out}
    assert len(out) == 6  # 5 dropped (empty str/ws/[]/empty-text-list/None)
    # every surviving message has at least one renderable part
    for m in out:
        assert isinstance(m, (SystemMessage, ToolMessage)) or agent._content_has_parts(
            m.content, bool(getattr(m, "tool_calls", None))
        )
    # the multimodal message with a real image part survived
    assert any(isinstance(m.content, list) and len(m.content) == 2 for m in out)
    assert any(getattr(m, "tool_calls", None) for m in out)
    assert any(isinstance(m, ToolMessage) for m in out)


def test_content_has_parts_shapes():
    import agent
    assert agent._content_has_parts("hi", False) is True
    assert agent._content_has_parts("  ", False) is False
    assert agent._content_has_parts("", True) is True                       # tool_calls win
    assert agent._content_has_parts([], False) is False
    assert agent._content_has_parts([{"type": "text", "text": ""}], False) is False
    assert agent._content_has_parts([{"type": "image_url", "image_url": {}}], False) is True
    assert agent._content_has_parts(None, False) is False


@pytest.mark.asyncio
async def test_build_graph_creates_two_llms(mock_llm):
    mock_cls_instance, agent = mock_llm
    # ChatGoogleGenerativeAI should be called at least three times:
    # flash (tool-deciding), pro (complex reasoning), synth (fast final draft).
    from langchain_google_genai import ChatGoogleGenerativeAI

    agent._user_graphs.clear()
    await agent._build_graph("test-user")
    assert ChatGoogleGenerativeAI.call_count >= 3
    calls = ChatGoogleGenerativeAI.call_args_list
    models = [c.kwargs.get("model") for c in calls]
    # Model ids come from vertex_config (serve + lite tiers), which are env-driven
    # so they can point at the US REP endpoint or global — assert against the
    # config, not hardcoded ids.
    import vertex_config
    assert vertex_config.SERVE_MODEL in models  # flash + pro (tool-deciding / reasoning)
    assert vertex_config.LITE_MODEL in models   # synth (fast final draft)


def _mock_astream(final_content="ok"):
    """Create a mock astream that yields agent events."""

    async def astream(*args, **kwargs):
        # Store call args for assertions
        astream.last_call_args = args
        astream.last_call_kwargs = kwargs
        # Yield a final agent response
        yield {"agent": {"messages": [MagicMock(content=final_content, tool_calls=[])]}}

    astream.last_call_args = None
    astream.last_call_kwargs = None
    return astream


@pytest.mark.asyncio
async def test_run_agent_returns_final_message(mock_llm):
    _, agent = mock_llm
    mock_graph = MagicMock()
    mock_graph.astream = _mock_astream("Here are your logs.")
    agent._get_graph = AsyncMock(return_value=mock_graph)

    result = await agent.run_agent("show me recent errors")
    assert result == "Here are your logs."


@pytest.mark.asyncio
async def test_run_agent_passes_human_message(mock_llm):
    _, agent = mock_llm
    stream_fn = _mock_astream("ok")
    mock_graph = MagicMock()
    mock_graph.astream = stream_fn
    agent._get_graph = AsyncMock(return_value=mock_graph)

    await agent.run_agent("check cloud run services")

    call_args = stream_fn.last_call_args[0]
    messages = call_args["messages"]
    assert len(messages) == 1
    assert "check cloud run services" in messages[0].content


@pytest.mark.asyncio
async def test_run_agent_includes_user_context(mock_llm):
    _, agent = mock_llm
    stream_fn = _mock_astream("ok")
    mock_graph = MagicMock()
    mock_graph.astream = stream_fn
    agent._get_graph = AsyncMock(return_value=mock_graph)

    await agent.run_agent(
        "hello",
        conversation_id="conv-1",
        user_id="u-1",
        user_name="Alice",
        user_email="alice@test.com",
    )

    call_args = stream_fn.last_call_args[0]
    message_text = call_args["messages"][0].content
    assert "User: Alice" in message_text
    assert "Email: alice@test.com" in message_text
    assert "conversation_id: conv-1" in message_text


@pytest.mark.asyncio
async def test_run_agent_graph_routes_to_end_without_tools(mock_llm):
    """Full integration: LLM returns no tool_calls → graph goes straight to END."""
    llm_mock, agent = mock_llm

    from langchain_core.messages import AIMessage

    llm_mock.ainvoke.return_value = AIMessage(content="All good, no tools needed.")

    result = await agent.run_agent("how are things?")
    assert result == "All good, no tools needed."


@pytest.mark.asyncio
async def test_inject_auth_message(mock_llm):
    _, agent = mock_llm
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={"messages": [MagicMock(content="ok")]}
    )
    agent._get_graph = AsyncMock(return_value=mock_graph)

    await agent.inject_auth_message("user-1", "conv-1")

    call_args = mock_graph.ainvoke.call_args[0][0]
    msg = call_args["messages"][0].content
    assert "authenticated with VirtualDojo CRM" in msg


@pytest.mark.asyncio
async def test_get_graph_caches_per_user(mock_llm):
    _, agent = mock_llm
    agent._user_graphs.clear()

    graph1 = await agent._get_graph("user-a")
    graph2 = await agent._get_graph("user-a")
    graph3 = await agent._get_graph("user-b")

    assert graph1 is graph2  # Same user → same graph
    assert graph1 is not graph3  # Different user → different graph


def test_reset_user_graph(mock_llm):
    _, agent = mock_llm
    agent._user_graphs["user-x"] = "some_graph"
    agent.reset_user_graph("user-x")
    assert "user-x" not in agent._user_graphs


def test_extract_text_from_string(mock_llm):
    _, agent = mock_llm
    assert agent._extract_text("hello") == "hello"


def test_extract_text_from_content_blocks(mock_llm):
    _, agent = mock_llm
    blocks = [
        {"type": "text", "text": "line 1"},
        {"type": "text", "text": "line 2"},
    ]
    assert agent._extract_text(blocks) == "line 1\nline 2"


# ── per-message prompt-assembly cache ─────────────────────────────────────


def test_select_prompt_sections_cached_per_message(mock_llm, monkeypatch):
    _, agent = mock_llm
    agent._prompt_cache.clear()
    calls = []

    def _catalog():
        calls.append(1)
        return "## Available skills\n- **x** — y"

    monkeypatch.setattr(agent, "skills_catalog_text", _catalog)
    first = agent._select_prompt_sections("show me the logs")
    second = agent._select_prompt_sections("show me the logs")
    assert first == second
    assert len(calls) == 1  # loaders ran once; second hop hit the cache


def test_select_prompt_sections_hints_override_bypasses_cache(mock_llm, monkeypatch):
    """The selftune eval passes hints_override and must always see a fresh
    assembly — never a cached production prompt."""
    _, agent = mock_llm
    agent._prompt_cache.clear()
    out_a = agent._select_prompt_sections("hello", hints_override="CANDIDATE-A")
    out_b = agent._select_prompt_sections("hello", hints_override="CANDIDATE-B")
    assert "CANDIDATE-A" in out_a
    assert "CANDIDATE-B" in out_b
    assert "hello" not in agent._prompt_cache  # override runs are not cached


# ── 429 backoff ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_backoff_survives_extended_quota_storm(mock_llm, monkeypatch):
    """5 consecutive 429s then success — the July quota storms exhausted the
    previous 3-retry budget and killed background tasks outright."""
    _, agent = mock_llm
    monkeypatch.setattr(agent.asyncio, "sleep", AsyncMock())
    attempts = []

    class _LLM:
        async def ainvoke(self, messages):
            attempts.append(1)
            if len(attempts) <= 5:
                raise agent.ChatGoogleGenerativeAIError("429 RESOURCE_EXHAUSTED: quota")
            return "ok"

    assert await agent._ainvoke_with_backoff(_LLM(), []) == "ok"
    assert len(attempts) == 6


@pytest.mark.asyncio
async def test_backoff_propagates_non_quota_errors(mock_llm, monkeypatch):
    _, agent = mock_llm
    monkeypatch.setattr(agent.asyncio, "sleep", AsyncMock())

    class _LLM:
        async def ainvoke(self, messages):
            raise agent.ChatGoogleGenerativeAIError("400 INVALID_ARGUMENT")

    with pytest.raises(agent.ChatGoogleGenerativeAIError):
        await agent._ainvoke_with_backoff(_LLM(), [])


# ── fire-and-forget background helper ─────────────────────────────────────


@pytest.mark.asyncio
async def test_spawn_background_runs_and_swallows_errors(mock_llm):
    import asyncio as aio

    _, agent = mock_llm
    ran = []

    def _ok(**kwargs):
        ran.append(kwargs)

    def _boom(**kwargs):
        raise RuntimeError("boom")

    agent._spawn_background(_ok, a=1)
    agent._spawn_background(_boom)  # must not surface
    for _ in range(50):
        if ran and not agent._background_tasks:
            break
        await aio.sleep(0.01)
    assert ran == [{"a": 1}]
    assert not agent._background_tasks  # done tasks are discarded


# --- Cache telemetry (_log_cache_stats) ---


def test_log_cache_stats_prints_to_stdout(mock_llm, capsys):
    """[cache] telemetry must go to stdout via print(), not logger.info.

    The app never configures a logging handler, so logger.info records are
    dropped — the line was invisible in Cloud Logging when it first shipped.
    logger.warning is also wrong: stderr is ingested at error severity and
    would pollute severity>=WARNING filters.
    """
    import agent

    response = MagicMock()
    response.usage_metadata = {
        "input_tokens": 1000,
        "output_tokens": 50,
        "input_token_details": {"cache_read": 750},
    }

    agent._log_cache_stats(response)

    out = capsys.readouterr().out
    assert "[cache] input_tokens=1000 cache_read=750 (75% cached) output_tokens=50" in out


def test_log_cache_stats_silent_without_usage_metadata(mock_llm, capsys):
    import agent

    response = MagicMock()
    response.usage_metadata = None
    agent._log_cache_stats(response)

    no_input = MagicMock()
    no_input.usage_metadata = {"input_tokens": 0, "output_tokens": 5}
    agent._log_cache_stats(no_input)

    assert "[cache]" not in capsys.readouterr().out


def test_log_cache_stats_never_raises(mock_llm):
    """Telemetry must never break a turn, even on malformed metadata."""
    import agent

    bad = MagicMock()
    bad.usage_metadata = {"input_tokens": "garbage", "input_token_details": []}
    agent._log_cache_stats(bad)  # must not raise
