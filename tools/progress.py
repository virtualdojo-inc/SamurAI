"""Conversation-scoped progress tracking for multi-step agent tasks.

The agent calls `update_progress` as it works through a multi-step task.
Three things consume the stored progress:

1. The Teams status callback renders it live so the user sees what's
   actually happening instead of generic "Searching codebase..." chatter.
2. When the agent hits the recursion limit, the synthesizer in
   `agent.py` uses the progress doc (plus the tool log as supporting
   evidence) to write a real recovery message — citing the agent's own
   intent, not just its tool mechanics.
3. When the user says "continue" / "resume" on a subsequent turn,
   `agent.py` reads the progress doc and injects it into the system
   prompt so the agent resumes from where it left off.

Storage is in-process only. Progress survives across turns of the same
conversation (Cloud Run keeps an instance warm with min_instances=1) but
not across instance restarts. That's acceptable — if state is gone, the
"continue" path degrades gracefully to "I don't have prior context."
"""

import threading
import time
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

# Conversation-scoped store: conversation_id -> progress dict.
_progress: dict[str, dict[str, Any]] = {}
_progress_lock = threading.Lock()


def _store(
    conversation_id: str,
    summary: str,
    completed: list[str],
    in_progress: str,
    pending: list[str],
) -> dict[str, Any]:
    entry = {
        "summary": summary,
        "completed": list(completed),
        "in_progress": in_progress,
        "pending": list(pending),
        "updated_at": time.time(),
    }
    with _progress_lock:
        _progress[conversation_id] = entry
    return entry


def get_progress(conversation_id: str) -> Optional[dict[str, Any]]:
    """Return the latest progress entry for a conversation, or None."""
    with _progress_lock:
        entry = _progress.get(conversation_id)
        return dict(entry) if entry else None


def clear_progress(conversation_id: str) -> None:
    """Remove the progress entry for a conversation."""
    with _progress_lock:
        _progress.pop(conversation_id, None)


def render_progress_markdown(entry: dict[str, Any]) -> str:
    """Format a progress entry as user-readable markdown.

    Used by the Teams status callback, the recursion-limit synthesizer
    prompt, and the resume-from-plan context injection.
    """
    lines: list[str] = []
    summary = (entry.get("summary") or "").strip()
    if summary:
        lines.append(f"**{summary}**")

    completed = entry.get("completed") or []
    if completed:
        lines.append("Done:")
        lines.extend(f"- [x] {item}" for item in completed)

    in_progress = (entry.get("in_progress") or "").strip()
    if in_progress:
        lines.append(f"Now: {in_progress}")

    pending = entry.get("pending") or []
    if pending:
        lines.append("Next:")
        lines.extend(f"- [ ] {item}" for item in pending)

    return "\n".join(lines)


@tool
def update_progress(
    summary: str,
    completed: Optional[list[str]] = None,
    in_progress: str = "",
    pending: Optional[list[str]] = None,
    config: RunnableConfig = None,
) -> str:
    """Track and surface progress on a multi-step task.

    Call this AT THE START of a task once you understand what the work
    entails, and AGAIN whenever a major step finishes or your plan
    changes. The user sees the latest state live in Teams as you work;
    if you hit a tool-call limit before finishing, this becomes your
    recovery message; and if the user later says "continue" on a new
    turn, your plan is restored so you can resume cleanly.

    Use this for any task involving more than ~3 sequential tool calls
    or any task you can break into discrete steps. Skip for trivial
    one-shot queries (single log query, listing PRs, sending a message).

    Args:
        summary: One-line description of what you're working on.
        completed: Items already done (past tense, concrete — "matched
            issue #687 to row 12", not "did stuff").
        in_progress: The one thing you're working on right now.
        pending: Items still to do, in the order you plan to tackle them.
    """
    thread_id = None
    if config:
        configurable = config.get("configurable") or {}
        thread_id = configurable.get("thread_id")
    if not thread_id:
        return (
            "Error: update_progress could not determine the conversation. "
            "This is a wiring bug — report it; the call did nothing."
        )

    _store(
        conversation_id=thread_id,
        summary=summary,
        completed=completed or [],
        in_progress=in_progress,
        pending=pending or [],
    )
    return "Progress saved."


PROGRESS_TOOLS = [update_progress]
