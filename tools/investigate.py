"""Investigate sub-agent — dispatches focused, read-only troubleshooting queries.

The main agent calls `investigate(question)` to delegate a focused investigation
to a Flash-powered sub-graph with a narrow, read-only tool set. Multiple
investigate() calls in the same turn run concurrently via LangGraph's ToolNode,
so wall time = slowest investigator, not the sum.
"""

import asyncio
import json
import logging
import os
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from tools.gcp_logging import query_cloud_logs
from tools.github import (
    github_get_commit_diff,
    github_get_issue_details,
    github_list_issues,
    github_search_issues,
)
from tools.repo_sync import (
    list_repo_files,
    read_repo_file,
    read_repo_file_range,
    search_repo_code,
    sync_repo,
)

logger = logging.getLogger(__name__)


INVESTIGATOR_TOOLS = [
    sync_repo,
    read_repo_file,
    read_repo_file_range,
    search_repo_code,
    list_repo_files,
    query_cloud_logs,
    github_list_issues,
    github_search_issues,
    github_get_issue_details,
    github_get_commit_diff,
]


INVESTIGATOR_SYSTEM_PROMPT = (
    "You are a focused troubleshooting investigator running as a read-only sub-agent.\n"
    "Your job: answer ONE specific question using the tools provided.\n\n"
    "RULES:\n"
    "- Read-only investigation — the tools you have cannot modify anything.\n"
    "- Cite every code claim with a file:line reference.\n"
    "- Keep your final answer to 200 words or less.\n"
    "- If you can't reach a conclusion after 6 tool calls, report what you ruled "
    "out and stop. Don't speculate past the evidence. There is a HARD budget of "
    "12 tool calls — after that you must answer with what you have.\n"
    "- For code questions: call sync_repo first (its result includes a repo map — "
    "use it to pick the right directory instead of exploring with list_repo_files), "
    "then search_repo_code to locate the symbol, then "
    "read_repo_file_range(file, start_line, end_line) for just the lines that "
    "matter. Only fall back to read_repo_file when you need the whole file.\n"
    "- search_repo_code file_pattern accepts a basename glob ('*.py') or a "
    "directory-scoped glob ('alembic/versions/*.py'). If a search for a term you "
    "expected to exist returns no matches, retry ONCE with a shorter/simpler "
    "substring and no file_pattern before concluding it's absent.\n"
    "- search_repo_code is hard-capped at ~50 KB per call. For broad scans, start "
    "with output_mode='files_with_matches' to get paths only, then re-run "
    "search_repo_code on a tighter file_pattern, or jump straight to "
    "read_repo_file_range. Use offset to paginate when a result note tells you to.\n"
    "- Default to branch='main' on virtualdojo-inc/virtualdojo unless the question "
    "mentions dev/development.\n"
    "- Repos you can access: virtualdojo-inc/virtualdojo, virtualdojo-inc/virtualdojo_cli, "
    "virtualdojo-inc/SamurAI, virtualdojo-inc/Fedramp.\n"
    "- If the answer is 'I don't know from this data', say that plainly."
)


INVESTIGATOR_RECURSION_LIMIT = 50
INVESTIGATOR_TIMEOUT_SECONDS = 120

# Hard tool-call budget. The prompt asks investigators to stop at 6 calls, but
# prompt-only guidance is routinely blown past (prod logs showed one question
# burning 15+ calls, each re-sending the sub-agent's full context). Once the
# budget is spent, the model is invoked WITHOUT tools so it must answer.
INVESTIGATOR_MAX_TOOL_CALLS = 12


# --- Self-contained question guard ---
# Investigators start with a FRESH context: a question like "Are there any
# migrations adding these fields?" has no antecedent for "these fields", so the
# sub-agent guesses search terms and flails (observed in prod: 7 consecutive
# no-match greps). Reject dangling referents unless the question also carries a
# concrete anchor (path, issue #, identifier, or backticked term) that resolves
# the reference.
_DANGLING_REFERENT_RE = re.compile(
    r"\b(?:these|those|this|that|the above|the same|the aforementioned|the existing)\s+"
    r"(?:fields?|files?|issues?|functions?|methods?|errors?|changes?|columns?|"
    r"values?|tables?|bugs?|lines?|migrations?|records?|endpoints?|settings?|"
    r"models?|tests?|components?|items?|ones?)\b",
    re.IGNORECASE,
)
_CONCRETE_ANCHOR_RE = re.compile(
    r"[\w./-]+\.(?:py|js|ts|tsx|vue|md|yml|yaml|json|toml|txt|sql|sh|html|css)\b"  # file path
    r"|#\d+"                 # issue/PR number
    r"|`[^`]+`"              # backticked term
    r"|\b\w+_\w+\b"          # snake_case identifier
    r"|\b[a-z]+[A-Z]\w*\b"   # camelCase identifier
)


def _unresolved_referent(question: str) -> str | None:
    """Return the dangling phrase if the question references context the
    sub-agent can't see, else None."""
    m = _DANGLING_REFERENT_RE.search(question)
    if m and not _CONCRETE_ANCHOR_RE.search(question):
        return m.group(0)
    return None


_investigator_graph = None


def _build_investigator_graph():
    """Build a mini LangGraph with Flash and the investigator tool set."""
    import vertex_config
    llm = ChatGoogleGenerativeAI(
        model=vertex_config.SERVE_MODEL,
        **vertex_config.vertex_kwargs(),
    )
    llm_with_tools = llm.bind_tools(INVESTIGATOR_TOOLS)
    tool_node = ToolNode(INVESTIGATOR_TOOLS, handle_tool_errors=True)

    async def call_model(state: MessagesState):
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=INVESTIGATOR_SYSTEM_PROMPT)] + messages
        n_tool_results = sum(isinstance(m, ToolMessage) for m in messages)
        if n_tool_results >= INVESTIGATOR_MAX_TOOL_CALLS:
            # Budget exhausted: invoke WITHOUT tools bound so the model cannot
            # keep searching and must synthesize an answer from what it has.
            messages = messages + [
                HumanMessage(
                    content=(
                        f"TOOL BUDGET EXHAUSTED ({n_tool_results} tool calls used, "
                        f"limit {INVESTIGATOR_MAX_TOOL_CALLS}). Do not request more "
                        "tools. Answer NOW from the evidence gathered: state what "
                        "you confirmed (with file:line) and what you ruled out."
                    )
                )
            ]
            return {"messages": [await llm.ainvoke(messages)]}
        return {"messages": [await llm_with_tools.ainvoke(messages)]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END
        return "tools"

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()


def _get_graph():
    global _investigator_graph
    if _investigator_graph is None:
        _investigator_graph = _build_investigator_graph()
    return _investigator_graph


def _reset_graph():
    """Test helper — clear the cached graph so a new one is built on next call."""
    global _investigator_graph
    _investigator_graph = None


def _extract_text(content) -> str:
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content) if content is not None else ""


@tool
async def investigate(question: str) -> str:
    """Dispatch a focused troubleshooting sub-agent with read-only tools.

    Use this for non-trivial code or infrastructure questions where you want to
    delegate investigation to a focused sub-agent. Call this MULTIPLE times in
    the SAME turn to dispatch parallel investigators — LangGraph runs them
    concurrently, so 3 parallel calls take the same wall time as 1.

    The sub-agent has access to: sync_repo, read_repo_file, search_repo_code,
    list_repo_files, query_cloud_logs, github_list_issues, github_get_issue_details,
    github_get_commit_diff. It has no modification tools and no Teams/CRM/memory
    access.

    Best for:
    - Narrow, focused questions ("Find the auth dependency used by route X. Cite file:line.")
    - Comparisons ("Compare auth on route X vs route Y, report differences.")
    - Tracing ("Trace where config value Z is set and read.")

    Not for:
    - Broad open-ended questions ("Find all the bugs in the codebase")
    - Actions (it cannot modify anything)
    - Things you can answer with a single search_repo_code or read_repo_file call

    Args:
        question: A specific, SELF-CONTAINED question. The sub-agent starts with
            a fresh context and cannot see this conversation — spell out actual
            field/file/function/issue names instead of "these fields" / "that
            file". Questions with unresolved references are rejected without
            running. Include "Cite file:line" if you want line-level evidence.

    Returns:
        A written summary (up to ~200 words) from the sub-agent, or an error
        string starting with "Investigator failed:" on timeout or failure.
    """
    ref = _unresolved_referent(question)
    if ref:
        return (
            f"Investigator rejected the question: it contains the unresolved "
            f"reference '{ref}' but no concrete identifier. Investigators start "
            f"with a FRESH context and cannot see this conversation — restate "
            f"the question with the actual names (field/file/function/issue #) "
            f"the reference points to, then call investigate again."
        )

    start = time.time()
    q_tag = question[:80]
    logger.info("[investigate] q=%r", question[:200])
    print(f"[investigate] start q={q_tag!r}", flush=True)

    tool_count = 0
    final_text: str | None = None

    async def _run():
        nonlocal tool_count, final_text
        graph = _get_graph()
        last_call_start = time.time()
        async for event in graph.astream(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": INVESTIGATOR_RECURSION_LIMIT},
            stream_mode="updates",
        ):
            if "agent" in event:
                msgs = event["agent"].get("messages", [])
                if msgs:
                    last = msgs[-1]
                    tc = getattr(last, "tool_calls", None)
                    if tc:
                        # Log truncated args, not just names — without them,
                        # failed searches (wrong query/file_pattern) are
                        # undiagnosable from Cloud Logging.
                        calls = []
                        for t in tc:
                            try:
                                args_str = json.dumps(
                                    t.get("args") or {}, default=str
                                )[:200]
                            except (TypeError, ValueError):
                                args_str = "<unserializable>"
                            calls.append(f"{t.get('name', '')}({args_str})")
                        last_call_start = time.time()
                        print(
                            f"[investigate] sub_tool_calls: {calls} q={q_tag!r}",
                            flush=True,
                        )
                    else:
                        final_text = _extract_text(last.content)
            elif "tools" in event:
                batch_elapsed = time.time() - last_call_start
                for msg in event["tools"].get("messages", []):
                    name = getattr(msg, "name", None)
                    if not name:
                        continue
                    tool_count += 1
                    status = (
                        "error" if getattr(msg, "status", None) == "error" else "ok"
                    )
                    content_str = str(msg.content) if msg.content is not None else ""
                    size = len(content_str)
                    preview = content_str[:150]
                    print(
                        f"[investigate] sub_tool_result: {name} ({status}) "
                        f"size={size} batch_elapsed={batch_elapsed:.2f}s "
                        f"q={q_tag!r} -> {preview}",
                        flush=True,
                    )

    try:
        await asyncio.wait_for(_run(), timeout=INVESTIGATOR_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("[investigate] timeout q=%r", question[:100])
        print(
            f"[investigate] timeout elapsed={time.time() - start:.2f}s "
            f"tools={tool_count} q={q_tag!r}",
            flush=True,
        )
        return (
            f"Investigator failed: timed out after {INVESTIGATOR_TIMEOUT_SECONDS}s. "
            f"Try a more focused question."
        )
    except Exception as e:
        logger.exception("[investigate] error q=%r", question[:100])
        print(
            f"[investigate] error elapsed={time.time() - start:.2f}s "
            f"tools={tool_count} q={q_tag!r}: {type(e).__name__}: {e}",
            flush=True,
        )
        return f"Investigator failed: {type(e).__name__}: {e}"

    elapsed = time.time() - start
    print(
        f"[investigate] done elapsed={elapsed:.2f}s tools={tool_count} q={q_tag!r}",
        flush=True,
    )
    logger.info(
        "[investigate] done elapsed=%.2fs tools=%d q=%r",
        elapsed,
        tool_count,
        question[:100],
    )
    answer = final_text.strip() if final_text else ""
    return answer or "Investigator failed: empty response."


INVESTIGATE_TOOLS = [investigate]
