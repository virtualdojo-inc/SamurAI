"""Tests for conversation compaction in agent.py.

thread_id == conversation_id, so each Teams conversation is one persistent
checkpoint that grows forever. The `compact` graph node summarizes the oldest
turns into a rolling `summary` and RemoveMessages them so the stored checkpoint
stays bounded. These tests cover the pure cut/render helpers and the kill switch;
the node itself is a graph closure exercised end-to-end by the graph tests.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent as agent_module


def _human(text, i):
    return HumanMessage(content=text, id=f"h{i}")


def _ai(text, i, tool_calls=None):
    return AIMessage(content=text, id=f"a{i}", tool_calls=tool_calls or [])


def _tool(text, i, call_id):
    return ToolMessage(content=text, id=f"t{i}", tool_call_id=call_id, name="do_thing")


# ── _choose_compaction_cut ────────────────────────────────────────────────


def test_no_cut_for_short_history():
    msgs = [_human("hi", 0), _ai("hello", 0)]
    cut = agent_module._choose_compaction_cut(
        msgs, trigger_tokens=10, keep_tokens=5, trigger_msgs=100
    )
    assert cut == 0


def test_no_cut_when_under_both_thresholds():
    msgs = [_human("hi", 0), _ai("hello", 0), _human("more", 1), _ai("ok", 1)]
    # Well under token and message triggers → no compaction.
    cut = agent_module._choose_compaction_cut(
        msgs, trigger_tokens=1_000_000, keep_tokens=100, trigger_msgs=1000
    )
    assert cut == 0


def test_cut_lands_on_human_boundary():
    # A conversation of several assistant turns, each: Human, AI(tool_call),
    # Tool, AI(text). Force compaction with tiny thresholds.
    msgs = []
    for i in range(6):
        msgs.append(_human(f"question {i} " * 20, i))
        msgs.append(_ai("", i, tool_calls=[{"name": "do_thing", "args": {}, "id": f"c{i}"}]))
        msgs.append(_tool("result " * 20, i, f"c{i}"))
        msgs.append(_ai(f"answer {i} " * 20, i))

    cut = agent_module._choose_compaction_cut(
        msgs, trigger_tokens=50, keep_tokens=80, trigger_msgs=4
    )
    assert cut > 0, "expected compaction to trigger"
    # The kept window must start on a HumanMessage so no tool-call/result pair
    # is split across the boundary (Gemini 400s on an orphaned function part).
    assert isinstance(msgs[cut], HumanMessage)
    # Something is dropped and something is kept.
    assert 0 < cut < len(msgs)


def test_cut_never_orphans_a_tool_message():
    msgs = [
        _human("q0", 0),
        _ai("", 0, tool_calls=[{"name": "do_thing", "args": {}, "id": "c0"}]),
        _tool("r0 " * 50, 0, "c0"),
        _ai("a0 " * 50, 0),
        _human("q1", 1),
        _ai("a1 " * 50, 1),
    ]
    cut = agent_module._choose_compaction_cut(
        msgs, trigger_tokens=10, keep_tokens=20, trigger_msgs=1
    )
    # The message at the cut boundary is a human turn — never a bare ToolMessage.
    if cut > 0:
        assert not isinstance(msgs[cut], ToolMessage)
        assert isinstance(msgs[cut], HumanMessage)


# ── _render_messages_for_summary ──────────────────────────────────────────


def test_render_includes_tool_call_names_and_skips_empty():
    msgs = [
        _human("hello", 0),
        _ai("", 0, tool_calls=[{"name": "query_cloud_logs", "args": {}, "id": "c0"}]),
        _ai("", 99),  # empty, no tool calls → skipped
        _ai("here are the logs", 1),
    ]
    rendered = agent_module._render_messages_for_summary(msgs)
    assert "Human: hello" in rendered
    assert "called tools: query_cloud_logs" in rendered
    assert "here are the logs" in rendered
    # The empty AI message contributes no line.
    assert rendered.count("\n") == 2


# ── kill switch ───────────────────────────────────────────────────────────


def test_compaction_enabled_default_on(monkeypatch):
    monkeypatch.delenv("SAMURAI_COMPACT_MODE", raising=False)
    assert agent_module._compaction_enabled() is True


def test_compaction_kill_switch(monkeypatch):
    monkeypatch.setenv("SAMURAI_COMPACT_MODE", "off")
    assert agent_module._compaction_enabled() is False
    monkeypatch.setenv("SAMURAI_COMPACT_MODE", "on")
    assert agent_module._compaction_enabled() is True


def test_compaction_settings_env_override(monkeypatch):
    monkeypatch.setenv("SAMURAI_COMPACT_TRIGGER_TOKENS", "5000")
    monkeypatch.setenv("SAMURAI_COMPACT_KEEP_TOKENS", "2000")
    monkeypatch.setenv("SAMURAI_COMPACT_TRIGGER_MSGS", "12")
    assert agent_module._compaction_settings() == (5000, 2000, 12)
