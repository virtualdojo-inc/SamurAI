"""Tests for tools/progress.py — conversation-scoped progress tracking."""

import threading

import pytest


@pytest.fixture(autouse=True)
def _reset_progress_store():
    """Clear the module-level store between tests so they don't bleed state."""
    from tools import progress

    progress._progress.clear()
    yield
    progress._progress.clear()


def _invoke_update(conv_id: str, **kwargs):
    """Invoke the update_progress tool with a synthetic RunnableConfig."""
    from tools.progress import update_progress

    config = {"configurable": {"thread_id": conv_id}}
    return update_progress.invoke(kwargs, config=config)


# --- update_progress tool ---


def test_update_progress_writes_to_conversation_scoped_store():
    from tools.progress import get_progress

    _invoke_update(
        "conv-1",
        summary="Linking GH issues to tracker rows",
        completed=["matched #711 to row 4"],
        in_progress="matching #687 to row 12",
        pending=["match #625"],
    )

    entry = get_progress("conv-1")
    assert entry is not None
    assert entry["summary"] == "Linking GH issues to tracker rows"
    assert entry["completed"] == ["matched #711 to row 4"]
    assert entry["in_progress"] == "matching #687 to row 12"
    assert entry["pending"] == ["match #625"]
    assert "updated_at" in entry


def test_update_progress_returns_short_confirmation_to_model():
    """The tool result string goes into the LLM context — keep it tiny so
    repeated calls don't bloat the prompt."""
    result = _invoke_update("conv-x", summary="Doing work")
    assert "Progress saved" in result
    # Sanity: not gigantic
    assert len(result) < 200


def test_update_progress_overwrites_previous_entry():
    """Each call replaces the previous progress — we want the latest snapshot,
    not a journal. Otherwise the prompt grows unbounded as work continues."""
    from tools.progress import get_progress

    _invoke_update(
        "conv-2",
        summary="First pass",
        completed=["step A"],
        pending=["step B", "step C"],
    )
    _invoke_update(
        "conv-2",
        summary="Second pass",
        completed=["step A", "step B"],
        in_progress="step C",
        pending=[],
    )

    entry = get_progress("conv-2")
    assert entry["summary"] == "Second pass"
    assert entry["completed"] == ["step A", "step B"]
    assert entry["in_progress"] == "step C"
    assert entry["pending"] == []


def test_update_progress_isolates_conversations():
    """Two simultaneous conversations must not see each other's progress —
    multiple users hit the bot in parallel."""
    from tools.progress import get_progress

    _invoke_update("conv-A", summary="A's task", completed=["a1"])
    _invoke_update("conv-B", summary="B's task", completed=["b1"])

    a = get_progress("conv-A")
    b = get_progress("conv-B")
    assert a["summary"] == "A's task"
    assert b["summary"] == "B's task"
    assert a["completed"] == ["a1"]
    assert b["completed"] == ["b1"]


def test_update_progress_handles_missing_thread_id():
    """Defensive: if the wiring breaks and config has no thread_id, the tool
    returns an error string instead of crashing the agent loop."""
    from tools.progress import update_progress, get_progress

    result = update_progress.invoke(
        {"summary": "no conv"},
        config={"configurable": {}},
    )
    assert "Error" in result
    assert get_progress("") is None


def test_update_progress_accepts_minimal_args():
    """Summary alone is enough — completed/in_progress/pending are optional."""
    from tools.progress import get_progress

    _invoke_update("conv-min", summary="Just starting")
    entry = get_progress("conv-min")
    assert entry["summary"] == "Just starting"
    assert entry["completed"] == []
    assert entry["pending"] == []
    assert entry["in_progress"] == ""


def test_update_progress_is_thread_safe():
    """Multiple threads can update different conversations concurrently
    without losing writes (LangGraph runs sync tools in worker threads)."""
    from tools.progress import get_progress

    def write(i):
        _invoke_update(f"conv-{i}", summary=f"Task {i}", completed=[f"step{i}"])

    threads = [threading.Thread(target=write, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i in range(20):
        entry = get_progress(f"conv-{i}")
        assert entry is not None
        assert entry["summary"] == f"Task {i}"


# --- get_progress / clear_progress helpers ---


def test_get_progress_returns_none_when_absent():
    from tools.progress import get_progress

    assert get_progress("never-existed") is None


def test_get_progress_returns_copy_not_live_reference():
    """Callers must not be able to mutate the store through the returned dict —
    that would let one consumer corrupt state another consumer reads."""
    from tools.progress import get_progress

    _invoke_update("conv-c", summary="initial", completed=["x"])
    entry = get_progress("conv-c")
    entry["summary"] = "mutated"
    entry["completed"].append("y")

    fresh = get_progress("conv-c")
    assert fresh["summary"] == "initial"
    # Note: shallow copy — nested lists are still shared. Document this in
    # the helper; the synthesizer only reads, never writes.


def test_clear_progress_removes_entry():
    from tools.progress import clear_progress, get_progress

    _invoke_update("conv-d", summary="work in progress")
    assert get_progress("conv-d") is not None

    clear_progress("conv-d")
    assert get_progress("conv-d") is None


def test_clear_progress_no_op_when_absent():
    from tools.progress import clear_progress

    # Must not raise
    clear_progress("never-existed")


# --- render_progress_markdown ---


def test_render_includes_all_sections_when_populated():
    from tools.progress import render_progress_markdown

    entry = {
        "summary": "Linking GH issues",
        "completed": ["matched #711"],
        "in_progress": "matching #687",
        "pending": ["match #625", "verify each update"],
        "updated_at": 0,
    }
    md = render_progress_markdown(entry)
    assert "**Linking GH issues**" in md
    assert "Done:" in md
    assert "- [x] matched #711" in md
    assert "Now: matching #687" in md
    assert "Next:" in md
    assert "- [ ] match #625" in md
    assert "- [ ] verify each update" in md


def test_render_omits_empty_sections():
    """If only summary is set, the rendered output shouldn't include empty
    'Done:' / 'Next:' headers — those would look broken in Teams."""
    from tools.progress import render_progress_markdown

    md = render_progress_markdown(
        {"summary": "Just started", "completed": [], "in_progress": "", "pending": []}
    )
    assert "**Just started**" in md
    assert "Done:" not in md
    assert "Now:" not in md
    assert "Next:" not in md


def test_render_handles_missing_keys():
    """Defensive against partial dicts (e.g. from a future schema change)."""
    from tools.progress import render_progress_markdown

    md = render_progress_markdown({"summary": "x"})
    assert "**x**" in md
    assert "Done:" not in md
