"""Tests for conversation compaction in agent.py.

thread_id == conversation_id, so each Teams conversation is one persistent
checkpoint that grows forever. The `compact` graph node summarizes the oldest
turns into a rolling `summary` and RemoveMessages them so the stored checkpoint
stays bounded. These tests cover the pure cut/render helpers and the kill switch;
the node itself is a graph closure exercised end-to-end by the graph tests.
"""

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)

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
    monkeypatch.setenv("SAMURAI_COMPACT_SUMMARY_MAX_CHARS", "1234")
    cfg = agent_module._compaction_settings()
    assert (cfg.trigger_tokens, cfg.keep_tokens, cfg.trigger_msgs) == (5000, 2000, 12)
    assert cfg.summary_max_chars == 1234


def test_compacted_window_stays_under_inference_trim_budget(monkeypatch):
    """The whole point of the default thresholds: a compacted thread must fit
    inside _MAX_INPUT_TOKENS, so call_model's trim_messages never has to
    silently truncate it. If someone tunes keep/trigger above the inference
    budget, compaction stops being accuracy-neutral."""
    for var in (
        "SAMURAI_COMPACT_TRIGGER_TOKENS",
        "SAMURAI_COMPACT_KEEP_TOKENS",
        "SAMURAI_COMPACT_TRIGGER_MSGS",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = agent_module._compaction_settings()
    assert cfg.keep_tokens < cfg.trigger_tokens, "keep must be below trigger"
    assert cfg.trigger_tokens < agent_module._MAX_INPUT_TOKENS


# ── _select_summarizable_tail (bounded summarization payload) ─────────────


def _fat_thread(turns, chars):
    msgs = []
    for i in range(turns):
        msgs.append(_human("x" * chars, i))
        msgs.append(_ai("y" * chars, i))
    return msgs


def test_steady_state_summarizes_the_whole_drop_region():
    """When the drop region fits the per-pass budget (normal organic growth),
    nothing may be discarded unsummarized."""
    to_drop = _fat_thread(20, 300)
    split = agent_module._select_summarizable_tail(to_drop, 100_000)
    assert split == 0


def test_catch_up_bounds_the_summarization_payload():
    """A runaway thread's first compaction must NOT build one giant call. Before
    this bound, a 2.5M-token thread produced a ~2M-token summarization request —
    past the model's context window, so it raised, the error was swallowed, and
    the thread never compacted at all."""
    to_drop = _fat_thread(1500, 2340)  # ~2.3M est tokens, like the logged thread
    budget = 60_000
    split = agent_module._select_summarizable_tail(to_drop, budget)

    assert split > 0, "expected the ancient tail to be discarded unsummarized"
    to_summarize = to_drop[split:]
    est = agent_module._approx_tokens(to_summarize)
    assert est <= budget * 1.1, f"payload {est} exceeded budget {budget}"
    # And the slice kept is the NEWEST one, nearest the retained window.
    assert to_summarize[-1] is to_drop[-1]


def test_always_summarizes_at_least_one_message():
    """A single message larger than the whole budget must still be summarized,
    not silently skipped into a no-op."""
    huge = [_human("z" * 500_000, 0)]
    assert agent_module._select_summarizable_tail(huge, 1000) == 0


def test_summarizable_tail_handles_empty_region():
    assert agent_module._select_summarizable_tail([], 1000) == 0


@pytest.mark.asyncio
async def test_compact_state_bounds_the_prompt_on_a_runaway_thread(monkeypatch):
    """End-to-end: the transcript actually handed to the model is bounded, and
    the removal still covers the ENTIRE drop region (so the thread converges in
    one pass rather than stalling)."""
    monkeypatch.delenv("SAMURAI_COMPACT_MODE", raising=False)
    monkeypatch.setenv("SAMURAI_COMPACT_MAX_SUMMARIZE_TOKENS", "20000")

    msgs = _fat_thread(1200, 2340)  # far past any threshold
    llm = _FakeLLM("caught up")
    update = await agent_module._compact_state({"messages": msgs}, llm)

    assert update, "runaway thread must compact, not silently no-op"
    prompt = llm.calls[0][-1].content
    assert len(prompt) < 400_000, f"summarization prompt too big: {len(prompt)} chars"

    removed = {m.id for m in update["messages"]}
    kept = [m for m in msgs if m.id not in removed]
    cfg = agent_module._compaction_settings()
    # Converged in ONE pass: what remains is at/below the keep window.
    assert agent_module._approx_tokens(kept) <= cfg.trigger_tokens


# ── _cap_summary ──────────────────────────────────────────────────────────


def test_cap_summary_leaves_short_text_untouched():
    assert agent_module._cap_summary("short summary", 100) == "short summary"


def test_cap_summary_enforces_hard_bound():
    text = "word " * 5000
    capped = agent_module._cap_summary(text, 200)
    assert len(capped) <= 200 + len(
        "\n[summary truncated to stay within the compaction budget]"
    )
    assert "truncated" in capped


def test_cap_summary_cuts_on_a_boundary_not_mid_word():
    text = "First line of the summary.\n" + ("second line padding. " * 50)
    capped = agent_module._cap_summary(text, 120)
    body = capped.split("\n[summary truncated")[0]
    # Cut lands on a sentence/line boundary, so the body doesn't end mid-word.
    assert body.endswith(".") or body.endswith("summary.")


def test_cap_summary_handles_none_and_empty():
    assert agent_module._cap_summary(None, 100) == ""
    assert agent_module._cap_summary("   ", 100) == ""


# ── _compact_state (integration: mocked summarization model) ──────────────


class _FakeLLM:
    """Stands in for the Flash-Lite summarizer used by the compact node."""

    def __init__(self, text="ROLLING SUMMARY", exc=None):
        self.text = text
        self.exc = exc
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self.exc:
            raise self.exc
        return AIMessage(content=self.text, id="summary-1")


def _long_thread(turns=8, pad=400):
    msgs = []
    for i in range(turns):
        msgs.append(_human(f"question {i} " * pad, i))
        msgs.append(_ai(f"answer {i} " * pad, i))
    return msgs


@pytest.mark.asyncio
async def test_compact_state_removes_old_messages_and_stores_summary(monkeypatch):
    monkeypatch.delenv("SAMURAI_COMPACT_MODE", raising=False)
    monkeypatch.setenv("SAMURAI_COMPACT_TRIGGER_TOKENS", "1000")
    monkeypatch.setenv("SAMURAI_COMPACT_KEEP_TOKENS", "2000")
    monkeypatch.setenv("SAMURAI_COMPACT_TRIGGER_MSGS", "4")

    msgs = _long_thread()
    llm = _FakeLLM("Earlier: user asked about deploys; bot checked logs.")
    update = await agent_module._compact_state({"messages": msgs}, llm)

    assert update, "expected a compaction update"
    assert update["summary"].startswith("Earlier:")
    removals = update["messages"]
    assert removals and all(isinstance(m, RemoveMessage) for m in removals)
    # Only the oldest turns are removed — the recent window survives.
    assert len(removals) < len(msgs)
    removed_ids = {m.id for m in removals}
    kept_ids = [m.id for m in msgs if m.id not in removed_ids]
    assert kept_ids, "compaction must keep a recent window"
    assert llm.calls, "summarizer should have been invoked"


@pytest.mark.asyncio
async def test_compact_state_folds_in_previous_summary(monkeypatch):
    monkeypatch.setenv("SAMURAI_COMPACT_TRIGGER_TOKENS", "1000")
    monkeypatch.setenv("SAMURAI_COMPACT_KEEP_TOKENS", "2000")
    monkeypatch.setenv("SAMURAI_COMPACT_TRIGGER_MSGS", "4")

    llm = _FakeLLM("merged summary")
    await agent_module._compact_state(
        {"messages": _long_thread(), "summary": "PRIOR FACTS: PR #42 is open"}, llm
    )

    prompt = llm.calls[0][-1].content
    assert "PRIOR FACTS: PR #42 is open" in prompt, "prior summary must be carried in"


@pytest.mark.asyncio
async def test_compact_state_caps_the_stored_summary(monkeypatch):
    monkeypatch.setenv("SAMURAI_COMPACT_TRIGGER_TOKENS", "1000")
    monkeypatch.setenv("SAMURAI_COMPACT_KEEP_TOKENS", "2000")
    monkeypatch.setenv("SAMURAI_COMPACT_TRIGGER_MSGS", "4")
    monkeypatch.setenv("SAMURAI_COMPACT_SUMMARY_MAX_CHARS", "300")

    # A summarizer that ignores the instructed budget entirely.
    llm = _FakeLLM("blah " * 5000)
    update = await agent_module._compact_state({"messages": _long_thread()}, llm)

    assert len(update["summary"]) < 400, "summary must be hard-capped, not just asked"


@pytest.mark.asyncio
async def test_compact_state_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("SAMURAI_COMPACT_MODE", "off")
    llm = _FakeLLM()
    assert await agent_module._compact_state({"messages": _long_thread()}, llm) == {}
    assert not llm.calls, "kill switch must prevent the summarization call"


@pytest.mark.asyncio
async def test_compact_state_noop_for_short_thread(monkeypatch):
    monkeypatch.delenv("SAMURAI_COMPACT_MODE", raising=False)
    for var in (
        "SAMURAI_COMPACT_TRIGGER_TOKENS",
        "SAMURAI_COMPACT_KEEP_TOKENS",
        "SAMURAI_COMPACT_TRIGGER_MSGS",
    ):
        monkeypatch.delenv(var, raising=False)
    llm = _FakeLLM()
    state = {"messages": [_human("hi", 0), _ai("hello", 0)]}
    assert await agent_module._compact_state(state, llm) == {}
    assert not llm.calls, "short threads must not pay for a summarization call"


@pytest.mark.asyncio
async def test_compact_state_swallows_summarizer_failure(monkeypatch):
    """A failing summarizer must degrade to 'keep full history', never raise —
    the user's reply is already produced by the time this node runs."""
    monkeypatch.setenv("SAMURAI_COMPACT_TRIGGER_TOKENS", "1000")
    monkeypatch.setenv("SAMURAI_COMPACT_KEEP_TOKENS", "2000")
    monkeypatch.setenv("SAMURAI_COMPACT_TRIGGER_MSGS", "4")

    llm = _FakeLLM(exc=RuntimeError("vertex exploded"))
    update = await agent_module._compact_state({"messages": _long_thread()}, llm)

    assert update == {}, "no messages may be dropped when the summary failed"


@pytest.mark.asyncio
async def test_compact_state_does_not_drop_without_a_summary(monkeypatch):
    """An empty summary must not result in silent history loss."""
    monkeypatch.setenv("SAMURAI_COMPACT_TRIGGER_TOKENS", "1000")
    monkeypatch.setenv("SAMURAI_COMPACT_KEEP_TOKENS", "2000")
    monkeypatch.setenv("SAMURAI_COMPACT_TRIGGER_MSGS", "4")

    llm = _FakeLLM("   ")
    assert await agent_module._compact_state({"messages": _long_thread()}, llm) == {}
