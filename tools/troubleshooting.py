"""Troubleshooting knowledge base — embeddings-indexed patterns from prior bug hunts.

Stored in the shared LangMem InMemoryStore under the ("troubleshooting", "virtualdojo")
namespace, so retrieval reuses the existing embedding + persistence infrastructure.
Entries are saved explicitly by the main agent at the end of a successful bug hunt
(NOT auto-extracted in the background — too noisy).

Retrieval is automatic via retrieve_relevant_memories() — no tool call needed.
The search_troubleshooting tool is for explicit, manual lookups.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


TROUBLESHOOTING_NAMESPACE: tuple[str, ...] = ("troubleshooting", "virtualdojo")


def _embedding_text(
    symptom: str,
    winning_hypothesis: str,
    discriminating_evidence: str,
    fix_description: str,
    fix_location: str,
) -> str:
    """Build the text used for vector embedding of a troubleshooting step.

    Combines the fields the next user's question is most likely to overlap with:
    the symptom and the hypothesis. Evidence and fix help disambiguate between
    near-duplicate symptoms.
    """
    return (
        f"Symptom: {symptom}\n"
        f"Hypothesis: {winning_hypothesis}\n"
        f"Evidence: {discriminating_evidence}\n"
        f"Fix: {fix_description} (at {fix_location})"
    )


def _format_step(step: dict) -> str:
    """Render a stored step as a readable bullet for system-prompt injection."""
    symptom = step.get("symptom", "(no symptom)")
    hyp = step.get("winning_hypothesis", "(no hypothesis)")
    fix = step.get("fix_description", "")
    loc = step.get("fix_location", "")
    issue = step.get("github_issue")
    created = step.get("created_at")
    ruled_out = step.get("hypotheses_ruled_out") or []

    age_hint = ""
    if created:
        try:
            days = max(0, int((time.time() - float(created)) // 86400))
            age_hint = f" [saved {days}d ago]"
        except (TypeError, ValueError):
            pass

    issue_hint = f" (issue #{issue})" if issue else ""

    lines = [f"- {symptom}{issue_hint}{age_hint}"]
    lines.append(f"    hypothesis: {hyp}")
    if fix or loc:
        lines.append(f"    fix: {fix} @ {loc}")
    if ruled_out:
        lines.append(f"    ruled out: {'; '.join(ruled_out[:3])}")
    return "\n".join(lines)


async def _save_step(
    *,
    symptom: str,
    winning_hypothesis: str,
    discriminating_evidence: str,
    fix_location: str,
    fix_description: str,
    hypotheses_ruled_out: Optional[list[str]] = None,
    repo: Optional[str] = None,
    github_issue: Optional[int] = None,
    source: str = "manual",
    created_git_sha: Optional[str] = None,
) -> str:
    """Internal implementation — shared between the @tool and the backfill script."""
    from memory import get_memory_store

    step_id = str(uuid.uuid4())
    now = time.time()
    value = {
        "content": _embedding_text(
            symptom=symptom,
            winning_hypothesis=winning_hypothesis,
            discriminating_evidence=discriminating_evidence,
            fix_description=fix_description,
            fix_location=fix_location,
        ),
        "symptom": symptom,
        "winning_hypothesis": winning_hypothesis,
        "discriminating_evidence": discriminating_evidence,
        "fix_location": fix_location,
        "fix_description": fix_description,
        "hypotheses_ruled_out": hypotheses_ruled_out or [],
        "repo": repo,
        "github_issue": github_issue,
        "source": source,
        "created_at": now,
        "created_git_sha": created_git_sha,
        "retrieval_count": 0,
    }

    store = await get_memory_store()
    await store.aput(TROUBLESHOOTING_NAMESPACE, step_id, value)
    logger.info(
        "[troubleshooting] saved step id=%s symptom=%r source=%s",
        step_id,
        symptom[:80],
        source,
    )
    return step_id


@tool
async def save_troubleshooting_step(
    symptom: str,
    winning_hypothesis: str,
    discriminating_evidence: str,
    fix_location: str,
    fix_description: str,
    hypotheses_ruled_out: Optional[list[str]] = None,
    repo: Optional[str] = None,
    github_issue: Optional[int] = None,
) -> str:
    """Save a troubleshooting pattern to the shared VirtualDojo troubleshooting DB.

    Call this ONCE at the end of a successful bug hunt when you have a concrete
    root cause. Future sessions with similar symptoms will retrieve this pattern
    automatically via semantic search.

    ONLY save when ALL of:
    - A concrete root cause is confirmed (not speculation).
    - You have file:line evidence or an explicit config location.
    - The pattern could plausibly recur — skip one-off typos and trivial bugs.

    Prefer populating hypotheses_ruled_out — dead-end paths save future
    investigators from retracing them.

    Args:
        symptom: One-line user-facing description of the problem
            ("API key rejected on POST /activities").
        winning_hypothesis: The actual diagnosis
            ("activities.py imports get_current_user from app.core.deps which is JWT-only").
        discriminating_evidence: The tool call or evidence that proved it
            ("search_repo_code found two get_current_user definitions at
            app.core.deps:39 and app.api.deps:30; activities.py:12 imports from the wrong one").
        fix_location: file:line, config setting, or infrastructure location
            ("app/api/v1/endpoints/activities.py:12").
        fix_description: One-line description of the change
            ("change import to from app.api.deps import get_current_user").
        hypotheses_ruled_out: Dead-end hypotheses investigated and ruled out.
        repo: 'owner/repo' if the fix is repo-specific; None for cross-repo patterns.
        github_issue: GitHub issue number if one tracks this bug.
    """
    try:
        step_id = await _save_step(
            symptom=symptom,
            winning_hypothesis=winning_hypothesis,
            discriminating_evidence=discriminating_evidence,
            fix_location=fix_location,
            fix_description=fix_description,
            hypotheses_ruled_out=hypotheses_ruled_out,
            repo=repo,
            github_issue=github_issue,
            source="manual",
        )
        return f"Saved troubleshooting step {step_id[:8]}."
    except Exception as e:
        logger.exception("[troubleshooting] save failed")
        return f"Save failed: {type(e).__name__}: {e}"


@tool
async def search_troubleshooting(query: str, limit: int = 5) -> str:
    """Search the troubleshooting DB for patterns matching a symptom.

    Retrieval also happens automatically in the background on every message —
    relevant patterns are injected into the system prompt. Call this tool only
    when you want more results, a specific query, or to verify a match.

    Args:
        query: Free-text description of the symptom or the question you're asking.
        limit: Max results to return (default 5).
    """
    from memory import get_memory_store

    try:
        store = await get_memory_store()
        results = await store.asearch(
            TROUBLESHOOTING_NAMESPACE, query=query, limit=limit
        )
    except Exception as e:
        logger.exception("[troubleshooting] search failed")
        return f"Search failed: {type(e).__name__}: {e}"

    if not results:
        return f"No troubleshooting patterns matched '{query}'."

    formatted = [_format_step(r.value) for r in results]
    return f"Top {len(formatted)} troubleshooting patterns for '{query}':\n" + "\n".join(
        formatted
    )


@tool
async def delete_troubleshooting_step(step_id: str) -> str:
    """Delete a troubleshooting step by its full UUID.

    Use for cleanup of stale, incorrect, or low-value patterns. The step_id
    is returned by save_troubleshooting_step and appears in audit logs.

    Args:
        step_id: The full UUID of the step to delete.
    """
    from memory import get_memory_store

    try:
        store = await get_memory_store()
        await store.adelete(TROUBLESHOOTING_NAMESPACE, step_id)
        logger.info("[troubleshooting] deleted step id=%s", step_id)
        return f"Deleted troubleshooting step {step_id[:8]}."
    except Exception as e:
        logger.exception("[troubleshooting] delete failed")
        return f"Delete failed: {type(e).__name__}: {e}"


TROUBLESHOOTING_TOOLS = [
    save_troubleshooting_step,
    search_troubleshooting,
    delete_troubleshooting_step,
]


# --- Retrieval helper used by memory.retrieve_relevant_memories ---


async def retrieve_troubleshooting_patterns(query: str, limit: int = 3) -> Optional[str]:
    """Return a formatted string of top-K troubleshooting patterns for injection
    into the system prompt, or None if no matches.

    Increments the retrieval_count on surfaced steps so we can see which
    patterns are paying rent.
    """
    from memory import get_memory_store

    try:
        store = await get_memory_store()
        results = await store.asearch(
            TROUBLESHOOTING_NAMESPACE, query=query, limit=limit
        )
    except Exception as e:
        logger.debug("[troubleshooting] retrieval failed: %s", e)
        return None

    if not results:
        return None

    # Bump retrieval_count on each surfaced step (best-effort; ignore failures)
    for r in results:
        try:
            val = dict(r.value)
            val["retrieval_count"] = int(val.get("retrieval_count", 0)) + 1
            await store.aput(TROUBLESHOOTING_NAMESPACE, r.key, val)
        except Exception:
            pass

    lines = [_format_step(r.value) for r in results]
    return "Prior troubleshooting patterns:\n" + "\n".join(lines)
