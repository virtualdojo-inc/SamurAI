"""Tool to manually trigger SamurAI's in-boundary knowledge-base compile.

Lets a user ask SamurAI in Teams to "learn from today's chats" / "update your
knowledge". This runs the SAME in-boundary pipeline as the daily schedule
(ingest GitHub + Smartsheet → compile with regional Vertex Gemini → wiki), in
the background on samurai-bot — inside the Assured Workloads boundary. It does
NOT dispatch any GitHub Actions workflow (a GitHub runner is out-of-boundary)
and never calls an external LLM.

It can ship knowledge updates — only trigger with explicit Devin/Cyrus approval.
"""

from __future__ import annotations

import threading

from langchain_core.tools import tool


def _run_pipeline_background() -> None:
    from kb.run import run_support_pipeline

    # force=True: a deliberate human trigger bypasses the daily kill switch.
    run_support_pipeline(force=True)


@tool
def trigger_wiki_compile(reason: str = "") -> str:
    """Manually run SamurAI's in-boundary knowledge-base compile now.

    Kicks off the in-boundary pipeline (ingest recent GitHub issues + Smartsheet,
    then compile the support wiki via regional Vertex Gemini) in the background.
    Runs entirely inside the FedRAMP boundary; no external LLM, no GitHub runner.
    This can update the knowledge base — only trigger with Devin or Cyrus's
    explicit approval.

    Args:
        reason: Optional note on why it's being triggered (for the run log).
    """
    try:
        t = threading.Thread(target=_run_pipeline_background, daemon=True)
        t.start()
        return (
            "Started the in-boundary knowledge-base compile (ingest + Gemini "
            "compile of the support wiki) in the background. It runs inside the "
            f"FedRAMP boundary on Vertex Gemini. Reason: {reason or '(none)'}. "
            "Results land in gs://virtualdojo-knowledge/support/wiki/."
        )
    except Exception as e:
        return f"Could not start the knowledge-base compile: {type(e).__name__}: {e}"


SELF_IMPROVE_TOOLS = [trigger_wiki_compile]
