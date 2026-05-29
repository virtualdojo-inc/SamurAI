"""Tool to manually trigger SamurAI's self-improvement (wiki-compile) pipeline.

Lets a user ask SamurAI in Teams to "learn from today's chats" / "update your
knowledge", which dispatches the nightly wiki-compile GitHub Actions workflow on
demand. It opens a PR with skill/knowledge updates; merge rides the blue/green
deploy. This is an action that can ship changes — only trigger with explicit
Devin/Cyrus approval (autonomy rules).
"""

from __future__ import annotations

from langchain_core.tools import tool

from tools.github import _github

_REPO = "virtualdojo-inc/SamurAI"
_WORKFLOW_FILE = "nightly-wiki-compile.yml"


@tool
def trigger_wiki_compile(reason: str = "") -> str:
    """Manually trigger SamurAI's self-improvement run (the wiki-compile pipeline).

    Dispatches the wiki-compile GitHub Actions workflow now, so SamurAI reviews
    recent conversations and proposes updates to its skills/knowledge wiki via a
    PR. This can ship changes to production — only trigger with Devin or Cyrus's
    explicit approval.

    Args:
        reason: Optional note on why it's being triggered (shown to the team).
    """
    try:
        repo = _github().get_repo(_REPO)
        workflow = repo.get_workflow(_WORKFLOW_FILE)
        ok = workflow.create_dispatch(ref="main")
        if ok:
            return (
                f"Triggered the wiki-compile self-improvement workflow on "
                f"{_REPO}@main. It will review recent conversations and open a PR "
                f"with any skill/knowledge updates. Reason: {reason or '(none)'}"
            )
        return "GitHub did not confirm the dispatch — the workflow may not exist yet."
    except Exception as e:
        return f"Could not trigger the self-improvement workflow: {type(e).__name__}: {e}"


SELF_IMPROVE_TOOLS = [trigger_wiki_compile]
