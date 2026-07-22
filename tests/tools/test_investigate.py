"""Tests for tools/investigate.py — investigate sub-agent tool."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture(autouse=True)
def reset_graph():
    """Clear the cached sub-graph before and after each test so the mocked
    ChatGoogleGenerativeAI is picked up."""
    import tools.investigate as mod

    mod._reset_graph()
    yield
    mod._reset_graph()


@pytest.fixture
def mock_llm():
    """Patch ChatGoogleGenerativeAI so graph construction doesn't auth to Vertex."""
    with patch("tools.investigate.ChatGoogleGenerativeAI") as mock_cls:
        instance = MagicMock()
        instance.bind_tools.return_value = instance
        instance.ainvoke = AsyncMock(
            return_value=AIMessage(content="default mocked answer")
        )
        mock_cls.return_value = instance
        yield instance


# --- Registration & shape ---


def test_investigate_tool_exposed():
    from tools.investigate import INVESTIGATE_TOOLS, investigate

    assert investigate in INVESTIGATE_TOOLS
    assert investigate.name == "investigate"


def test_investigator_has_only_read_only_tools():
    """The sub-agent must not be able to modify anything or send comms."""
    from tools.investigate import INVESTIGATOR_TOOLS

    names = {t.name for t in INVESTIGATOR_TOOLS}
    # Required read-only tools
    assert "sync_repo" in names
    assert "read_repo_file" in names
    assert "read_repo_file_range" in names
    assert "search_repo_code" in names
    assert "list_repo_files" in names
    assert "query_cloud_logs" in names
    assert "github_list_issues" in names
    assert "github_search_issues" in names
    # Forbidden — modification / comms / memory tools
    forbidden = {
        "send_teams_message",
        "github_create_issue",
        "github_close_issue",
        "manage_memory",
        "manage_core_memory",
        "manage_team_memory",
        "social_publish_post",
        "social_schedule_post",
        "create_background_task",
        "edit_spreadsheet",
        "fill_spreadsheet_column",
        "fedramp_commit_document",
        "oscal_update_control",
    }
    assert not (forbidden & names), f"investigator has forbidden tools: {forbidden & names}"


def test_investigator_system_prompt_is_read_only_and_bounded():
    from tools.investigate import INVESTIGATOR_SYSTEM_PROMPT

    assert "read-only" in INVESTIGATOR_SYSTEM_PROMPT.lower()
    assert "file:line" in INVESTIGATOR_SYSTEM_PROMPT
    assert "200 words" in INVESTIGATOR_SYSTEM_PROMPT


def test_investigator_config_constants():
    """Recursion limit and timeout are bounded so investigators can't hang."""
    from tools.investigate import (
        INVESTIGATOR_RECURSION_LIMIT,
        INVESTIGATOR_TIMEOUT_SECONDS,
    )

    # Raised after first prod test showed 30 steps / 60s was too tight for Flash.
    assert 30 <= INVESTIGATOR_RECURSION_LIMIT <= 75
    assert 30 <= INVESTIGATOR_TIMEOUT_SECONDS <= 300


# --- _extract_text helper ---


def test_extract_text_from_string():
    from tools.investigate import _extract_text

    assert _extract_text("plain text") == "plain text"


def test_extract_text_from_content_blocks():
    from tools.investigate import _extract_text

    blocks = [
        {"type": "text", "text": "line 1"},
        {"type": "text", "text": "line 2"},
    ]
    assert _extract_text(blocks) == "line 1\nline 2"


def test_extract_text_ignores_non_text_blocks():
    from tools.investigate import _extract_text

    blocks = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "url": "..."},
    ]
    assert _extract_text(blocks) == "hello"


def test_extract_text_handles_none():
    from tools.investigate import _extract_text

    assert _extract_text(None) == ""


# --- Happy path ---


@pytest.mark.asyncio
async def test_investigate_returns_sub_agent_answer(mock_llm):
    """When the sub-graph completes with an AIMessage, investigate returns its text."""
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="Found it at activities.py:12")
    )

    from tools.investigate import investigate

    result = await investigate.ainvoke({"question": "Find the auth dep"})
    assert result == "Found it at activities.py:12"


@pytest.mark.asyncio
async def test_investigate_caches_graph(mock_llm):
    """The sub-graph is built once and reused across calls."""
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

    import tools.investigate as mod

    await mod.investigate.ainvoke({"question": "q1"})
    await mod.investigate.ainvoke({"question": "q2"})
    # ChatGoogleGenerativeAI is only instantiated inside _build_investigator_graph,
    # which should fire exactly once across both calls.
    # (The patch fixture's mock_cls is the class constructor.)
    # We access it via the patched module attribute.
    assert mod.ChatGoogleGenerativeAI.call_count == 1


# --- Error handling: investigators never raise, they return error strings ---


@pytest.mark.asyncio
async def test_investigate_catches_sub_agent_exception(mock_llm):
    """An exception inside the sub-graph is converted to a human-readable error string."""
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("llm exploded"))

    from tools.investigate import investigate

    result = await investigate.ainvoke({"question": "anything"})
    assert result.startswith("Investigator failed:")
    assert "RuntimeError" in result
    assert "llm exploded" in result


@pytest.mark.asyncio
async def test_investigate_times_out_cleanly(mock_llm):
    """If the sub-graph hangs longer than the timeout, investigate returns a timeout string."""

    async def hang(_messages):
        await asyncio.sleep(5)
        return AIMessage(content="never")

    mock_llm.ainvoke = hang

    import tools.investigate as mod

    with patch.object(mod, "INVESTIGATOR_TIMEOUT_SECONDS", 0.1):
        mod._reset_graph()  # rebuild graph so the patched LLM mock is bound
        result = await mod.investigate.ainvoke({"question": "hang forever"})

    assert result.startswith("Investigator failed:")
    assert "timed out" in result


@pytest.mark.asyncio
async def test_investigate_handles_empty_response(mock_llm):
    """If the sub-graph returns an empty content message, investigate returns an error string."""
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=""))

    from tools.investigate import investigate

    result = await investigate.ainvoke({"question": "anything"})
    assert result.startswith("Investigator failed:")
    assert "empty response" in result


# --- Parallelism contract ---


@pytest.mark.asyncio
async def test_investigate_routes_through_tool_node_when_tool_calls_present(mock_llm):
    """When the sub-agent emits tool_calls, the graph must route to the tool node,
    execute the tool, then loop back. Covers the 'tools' branch of should_continue."""
    # First turn: LLM asks to call sync_repo. Second turn: LLM returns final answer.
    first = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "sync_repo",
                "args": {
                    "repo": "virtualdojo-inc/virtualdojo",
                    "branch": "main",
                },
                "id": "call-1",
            }
        ],
    )
    final = AIMessage(content="Synced and done.")
    mock_llm.ainvoke = AsyncMock(side_effect=[first, final])

    # Stub repo_sync internals so sync_repo short-circuits with "up to date"
    # instead of actually cloning.
    with (
        patch("tools.repo_sync._get_remote_sha", return_value="abc123"),
        patch("tools.repo_sync._get_local_sha", return_value="abc123"),
        patch("tools.github._github_token", return_value="fake-token"),
    ):
        from tools.investigate import investigate

        result = await investigate.ainvoke({"question": "sync and report"})

    assert result == "Synced and done."
    # Both LLM turns should have fired — confirming the graph looped tools → agent.
    assert mock_llm.ainvoke.call_count == 2


@pytest.mark.asyncio
async def test_investigate_logs_sub_tool_activity(mock_llm, capsys):
    """Sub-agent should emit [investigate] sub_tool_calls and sub_tool_result
    lines so failures are diagnosable in Cloud Logging."""
    first = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "sync_repo",
                "args": {
                    "repo": "virtualdojo-inc/virtualdojo",
                    "branch": "main",
                },
                "id": "call-log-1",
            }
        ],
    )
    final = AIMessage(content="Logged and done.")
    mock_llm.ainvoke = AsyncMock(side_effect=[first, final])

    with (
        patch("tools.repo_sync._get_remote_sha", return_value="abc"),
        patch("tools.repo_sync._get_local_sha", return_value="abc"),
        patch("tools.github._github_token", return_value="fake"),
    ):
        from tools.investigate import investigate

        result = await investigate.ainvoke({"question": "log me"})

    assert result == "Logged and done."
    out = capsys.readouterr().out
    # Start and done markers
    assert "[investigate] start" in out
    assert "[investigate] done" in out
    # Sub-agent tool dispatch and result lines with the key observability fields
    assert "[investigate] sub_tool_calls:" in out
    assert "sync_repo" in out
    assert "[investigate] sub_tool_result:" in out
    assert "size=" in out
    assert "batch_elapsed=" in out


@pytest.mark.asyncio
async def test_investigate_logs_timeout_with_tool_count(mock_llm, capsys):
    """On timeout, the log line should include elapsed and tools count so we
    can see how far the sub-agent got before running out."""

    async def hang(_messages):
        await asyncio.sleep(5)
        return AIMessage(content="never")

    mock_llm.ainvoke = hang

    import tools.investigate as mod

    with patch.object(mod, "INVESTIGATOR_TIMEOUT_SECONDS", 0.1):
        mod._reset_graph()
        result = await mod.investigate.ainvoke({"question": "hang"})

    assert result.startswith("Investigator failed:")
    out = capsys.readouterr().out
    assert "[investigate] timeout" in out
    assert "elapsed=" in out
    assert "tools=" in out


@pytest.mark.asyncio
async def test_investigate_runs_in_parallel_from_asyncio_gather(mock_llm):
    """Multiple investigate() calls via asyncio.gather share one compiled graph
    and complete concurrently — mirrors how LangGraph's ToolNode dispatches
    parallel tool_calls in a single turn."""

    call_order = []

    async def slow_then_fast(_messages):
        # Simulate two concurrent calls finishing in reverse order to prove
        # they weren't serialized.
        idx = len(call_order)
        call_order.append(idx)
        delay = 0.2 if idx == 0 else 0.05
        await asyncio.sleep(delay)
        return AIMessage(content=f"answer-{idx}")

    mock_llm.ainvoke = slow_then_fast

    from tools.investigate import investigate

    results = await asyncio.gather(
        investigate.ainvoke({"question": "q1"}),
        investigate.ainvoke({"question": "q2"}),
    )

    assert len(results) == 2
    # Both calls returned concrete answers (no timeouts or errors).
    assert all(r.startswith("answer-") for r in results), results


# --- Self-contained question guard ---


def test_unresolved_referent_flags_dangling_reference():
    """The exact prod flail case: 'these fields' with no concrete identifier."""
    from tools.investigate import _unresolved_referent

    ref = _unresolved_referent(
        "Are there any Alembic migrations adding these fields? Cite file:line."
    )
    assert ref is not None
    assert "these fields" in ref.lower()


def test_unresolved_referent_allows_anchored_references():
    from tools.investigate import _unresolved_referent

    # A dangling phrase is fine when the question also names the referent.
    assert _unresolved_referent(
        "Does this file app/services/sales_order_service.py handle null "
        "order managers?"
    ) is None
    assert _unresolved_referent(
        "Are there migrations adding these fields: open_market, cta_member_id?"
    ) is None
    assert _unresolved_referent(
        "Is that function `get_current_user` imported anywhere else?"
    ) is None
    assert _unresolved_referent(
        "Are these changes related to issue #1067?"
    ) is None


def test_unresolved_referent_allows_plain_questions():
    from tools.investigate import _unresolved_referent

    assert _unresolved_referent("Find the auth dependency used by route X.") is None
    assert _unresolved_referent(
        "Trace where config value KB_PIPELINE_CRON is set and read."
    ) is None


@pytest.mark.asyncio
async def test_investigate_rejects_unresolved_referent_without_running(mock_llm):
    """A non-self-contained question must be rejected before any model call."""
    from tools.investigate import investigate

    result = await investigate.ainvoke(
        {"question": "Are there any validators for those columns? Cite evidence."}
    )
    assert "rejected" in result
    assert "those columns" in result
    assert "restate" in result.lower()
    mock_llm.ainvoke.assert_not_called()


# --- Hard tool-call budget ---


@pytest.mark.asyncio
async def test_investigator_tool_budget_forces_final_answer(mock_llm):
    """A sub-agent that keeps requesting tools must be cut off at
    INVESTIGATOR_MAX_TOOL_CALLS and forced to answer from gathered evidence
    (invoked without tools), instead of looping to the recursion limit."""
    from tools.investigate import INVESTIGATOR_MAX_TOOL_CALLS, investigate

    llm_calls = 0

    async def always_wants_tools(messages):
        nonlocal llm_calls
        llm_calls += 1
        # The budget branch appends a "TOOL BUDGET EXHAUSTED" message and
        # invokes the tool-free model; answer plainly when we see it.
        if any(
            "TOOL BUDGET EXHAUSTED" in str(getattr(m, "content", ""))
            for m in messages
        ):
            return AIMessage(content="Forced answer from partial evidence.")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "sync_repo",
                    "args": {
                        "repo": "virtualdojo-inc/virtualdojo",
                        "branch": "main",
                    },
                    "id": f"call-{llm_calls}",
                }
            ],
        )

    mock_llm.ainvoke = always_wants_tools

    with (
        patch("tools.repo_sync._get_remote_sha", return_value="abc"),
        patch("tools.repo_sync._get_local_sha", return_value="abc"),
        patch("tools.github._github_token", return_value="fake"),
    ):
        result = await investigate.ainvoke({"question": "loop forever please"})

    assert result == "Forced answer from partial evidence."
    # Exactly budget-many tool rounds + the final forced answer.
    assert llm_calls == INVESTIGATOR_MAX_TOOL_CALLS + 1


@pytest.mark.asyncio
async def test_investigate_logs_tool_call_args(mock_llm, capsys):
    """sub_tool_calls lines must include (truncated) args, not just names —
    without them, failed searches are undiagnosable from Cloud Logging."""
    first = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_repo_code",
                "args": {"query": "open_market", "file_pattern": "alembic/*.py"},
                "id": "call-args-1",
            }
        ],
    )
    final = AIMessage(content="done")
    mock_llm.ainvoke = AsyncMock(side_effect=[first, final])

    from tools.investigate import investigate

    await investigate.ainvoke({"question": "search for open_market usages"})

    out = capsys.readouterr().out
    assert "[investigate] sub_tool_calls:" in out
    assert "open_market" in out
    assert "alembic/*.py" in out
