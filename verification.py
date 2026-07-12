"""Chain-of-Verification (CoVe) node for SamurAI's LangGraph agent.

Purpose: reduce hallucinated specifics (line numbers, counts, file paths,
API names) in agent responses by running an independent verification pass
against the tool-call log before shipping the response.

Research basis:
  - Dhuliawala et al., "Chain-of-Verification Reduces Hallucination"
    (arXiv:2309.11495, ACL Findings 2024). +23% F1 on closed-book QA
    when verification runs in a FRESH context — not as self-critique.
  - Google has a known Gemini bug (python-genai #813) where the model
    claims function-call output that was never returned. An independent
    verifier catches this because it sees only the draft + the actual
    tool log, not the model's internal narrative.

Design:
  - Verifier runs as a separate Flash call. Fresh context.
  - It sees ONLY the draft response and the serialized tool trace from
    this turn. No system prompt, no history, no user goals.
  - Its single job: for every specific claim (number, line, file, API),
    decide grounded / ungrounded / unverifiable.
  - Ungrounded claims cause the graph to route back to the main agent
    with a structured directive to either verify via tool call or drop
    the claim.

Modes (env var SAMURAI_VERIFY_MODE):
  - "off"     : disabled. Verification node is skipped.
  - "shadow"  : runs the verifier and logs what it would have rejected,
                but passes the draft through unchanged. Use this to
                collect data before switching to enforce.
  - "enforce" : runs the verifier and routes back to the agent when
                claims are ungrounded.

Cost: one extra Flash call per agent turn that produces a non-tool
response. Typical latency ~200-400ms. Token cost ~1-2k.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)


VERIFY_MODE_OFF = "off"
VERIFY_MODE_SHADOW = "shadow"
VERIFY_MODE_ENFORCE = "enforce"


def get_verify_mode() -> str:
    """Read SAMURAI_VERIFY_MODE; default off (opt-in rollout)."""
    mode = os.environ.get("SAMURAI_VERIFY_MODE", VERIFY_MODE_OFF).lower().strip()
    if mode not in (VERIFY_MODE_OFF, VERIFY_MODE_SHADOW, VERIFY_MODE_ENFORCE):
        logger.warning(
            f"Unknown SAMURAI_VERIFY_MODE={mode!r}; falling back to off"
        )
        return VERIFY_MODE_OFF
    return mode


VERIFIER_SYSTEM_PROMPT = (
    "You are a claim-verifier. Your ONLY job is to compare specific "
    "factual claims in a DRAFT response against a TOOL LOG of what was "
    "actually observed during this turn.\n\n"
    "Extract every quantitative or locational claim from the draft. For "
    "each, decide whether the tool log supports it.\n\n"
    "CLAIMS YOU MUST CHECK:\n"
    "- Specific counts (e.g., '5,300 instances', '15 services', '4 PRs')\n"
    "- Line numbers and file paths (e.g., 'line 136 of foo.py')\n"
    "- Named functions, classes, APIs, or config keys\n"
    "- Specific dates, timestamps, revision names\n"
    "- Percentages and quantitative comparisons\n"
    "- Scale or scope language tied to specifics ('widespread', "
    "'systemic', '40% reduction') — unless the tool log supports it\n\n"
    "CLAIMS YOU DO NOT CHECK:\n"
    "- Recommendations and opinions ('this should be fixed soon')\n"
    "- Qualitative descriptions ('looks like a caching issue')\n"
    "- Common knowledge unrelated to this environment\n\n"
    "OUTPUT: a JSON object with a 'claims' array. Each element has:\n"
    "  - 'claim': the exact phrase from the draft\n"
    "  - 'grounded': true / false / 'unverifiable'\n"
    "  - 'evidence': if grounded, a short quote from the tool log; "
    "if not, a one-sentence reason\n\n"
    "Return ONLY valid JSON. No prose wrapper. No markdown fences."
)


def _serialize_tool_trace(messages: list) -> str:
    """Build a compact string log of every tool call + tool result in the
    conversation turn. Older messages are included too so multi-turn
    claims still get evidence.

    We preserve tool call IDs and truncate overly long tool outputs to
    keep the verifier prompt manageable. Long outputs (>4k chars) are
    truncated with a notice so the verifier knows evidence may exist
    beyond the cutoff."""
    TRUNC = 4000
    entries: list[str] = []
    for msg in messages:
        # Tool CALL (model invoking a tool)
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                name = tc.get("name", "?")
                args = tc.get("args", {})
                tc_id = tc.get("id", "?")
                try:
                    args_str = json.dumps(args, default=str)[:500]
                except Exception:
                    args_str = str(args)[:500]
                entries.append(
                    f"[CALL id={tc_id}] {name}({args_str})"
                )
        # Tool RESULT
        elif isinstance(msg, ToolMessage):
            tc_id = getattr(msg, "tool_call_id", "?")
            content = str(msg.content) if msg.content is not None else ""
            if len(content) > TRUNC:
                content = content[:TRUNC] + f"\n…[truncated {len(content) - TRUNC} chars]"
            entries.append(f"[RESULT id={tc_id}] {content}")
    if not entries:
        return "(no tool calls were made this turn)"
    return "\n\n".join(entries)


def _extract_draft_text(msg: AIMessage) -> str:
    """Gemini content can be a string or list of blocks. Flatten to text."""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _format_nudge(ungrounded: list[dict[str, Any]]) -> str:
    """Build the instruction that goes back to the main agent when
    verification fails. Framed as a user-turn instruction so the main
    agent treats it as a correction, not ambient context."""
    lines = [
        "[VERIFICATION FAILED — the following specific claims in your "
        "draft are not supported by your tool calls from this turn. "
        "Either verify each one with an additional tool call (grep, "
        "file read, log query) or remove the claim from your response. "
        "Do NOT ship the response until every specific claim traces to "
        "a tool result.]",
        "",
    ]
    for item in ungrounded[:10]:  # cap at 10 to avoid runaway prompts
        claim = item.get("claim", "").strip()[:200]
        reason = item.get("evidence", "").strip()[:200]
        lines.append(f"- UNGROUNDED: {claim!r}")
        if reason:
            lines.append(f"  Why: {reason}")
    if len(ungrounded) > 10:
        lines.append(f"... and {len(ungrounded) - 10} more.")
    return "\n".join(lines)


_verifier_llm = None


def _get_verifier_llm():
    """Cache the verifier LLM so repeated turns don't re-instantiate."""
    global _verifier_llm
    if _verifier_llm is None:
        # Use Flash — this is a discriminative task, not a reasoning one.
        # A cheaper model is fine and keeps the overhead low.
        import vertex_config
        _verifier_llm = ChatGoogleGenerativeAI(
            model=vertex_config.SERVE_MODEL,
            # verifier is the one place low temp is probably fine
            **vertex_config.vertex_kwargs(temperature=0.0),
        )
    return _verifier_llm


async def verify_response(draft: AIMessage, messages: list) -> dict[str, Any]:
    """Run the verifier. Returns a dict:
        {
            "mode": "off" | "shadow" | "enforce",
            "ungrounded": [ {claim, evidence}, ... ],
            "raw": the verifier's raw JSON output (or None on failure),
            "error": error message if the verifier call failed,
        }

    Never raises. If the verifier itself fails, returns an empty
    ungrounded list so the draft is not penalized for verifier errors.
    """
    mode = get_verify_mode()
    result: dict[str, Any] = {"mode": mode, "ungrounded": [], "raw": None}
    if mode == VERIFY_MODE_OFF:
        return result

    draft_text = _extract_draft_text(draft).strip()
    if not draft_text:
        # Nothing to verify — no text claims in the draft.
        return result

    tool_log = _serialize_tool_trace(messages)
    if tool_log == "(no tool calls were made this turn)":
        # Skip verification when no tools ran — nothing to ground against.
        # This avoids penalizing greetings, clarifying questions, and
        # simple acknowledgments.
        return result

    try:
        llm = _get_verifier_llm()
        verifier_input = (
            f"DRAFT:\n{draft_text}\n\n"
            f"---\n\n"
            f"TOOL LOG:\n{tool_log}"
        )
        response = await llm.ainvoke(
            [
                SystemMessage(content=VERIFIER_SYSTEM_PROMPT),
                HumanMessage(content=verifier_input),
            ]
        )
        raw = _extract_draft_text(response)
        result["raw"] = raw
        # Strip common markdown fences defensively.
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            # ```json\n...\n```
            raw_clean = raw_clean.strip("`")
            if raw_clean.lower().startswith("json"):
                raw_clean = raw_clean[4:]
            raw_clean = raw_clean.strip()
        parsed = json.loads(raw_clean)
        claims = parsed.get("claims", [])
        ungrounded = [
            c for c in claims
            if c.get("grounded") is False
        ]
        result["ungrounded"] = ungrounded
    except json.JSONDecodeError as e:
        logger.warning(
            f"Verifier returned non-JSON output; skipping this turn: {e}"
        )
        result["error"] = f"json_decode: {e}"
    except Exception as e:
        logger.warning(
            f"Verifier call failed; skipping this turn: {e}",
            exc_info=True,
        )
        result["error"] = str(e)
    return result


async def verification_node(state):
    """LangGraph node that runs verification on the agent's latest
    response. Called after the agent produces a response with no
    pending tool calls.

    Behavior per mode:
      - off:      unreachable (the conditional edge skips this node).
      - shadow:   log the ungrounded claims, return no state change.
      - enforce:  if ungrounded, append a correction message and route
                  the graph back to the agent.

    This function does NOT rewrite the draft — only the main agent is
    allowed to produce user-facing content. Verification can only flag
    and route back."""
    messages = state["messages"]
    if not messages:
        return {}

    last = messages[-1]
    if not isinstance(last, AIMessage):
        return {}
    # If the agent still has pending tool calls, skip verification —
    # the draft isn't final yet.
    if getattr(last, "tool_calls", None):
        return {}

    result = await verify_response(last, messages)
    mode = result["mode"]
    ungrounded = result["ungrounded"]

    if mode == VERIFY_MODE_SHADOW:
        if ungrounded:
            print(
                f"[verification.shadow] would have flagged {len(ungrounded)} "
                f"claim(s): {json.dumps(ungrounded)[:2000]}",
                flush=True,
            )
        else:
            print(
                f"[verification.shadow] draft verified clean "
                f"(raw={'yes' if result.get('raw') else 'no'})",
                flush=True,
            )
        # Shadow mode never alters state.
        return {}

    if mode == VERIFY_MODE_ENFORCE:
        if ungrounded:
            print(
                f"[verification.enforce] routing back with {len(ungrounded)} "
                f"ungrounded claim(s)",
                flush=True,
            )
            return {"messages": [HumanMessage(content=_format_nudge(ungrounded))]}
        # All claims grounded — nothing to do.
        return {}

    return {}


def should_verify(state) -> str:
    """Conditional edge: decide whether to run verification or end.

    Returns:
      - "verify" if mode is shadow/enforce and the last message is a
        non-tool AI response.
      - "end" otherwise.
    """
    mode = get_verify_mode()
    if mode == VERIFY_MODE_OFF:
        return "end"
    messages = state.get("messages") or []
    if not messages:
        return "end"
    last = messages[-1]
    if not isinstance(last, AIMessage):
        return "end"
    if getattr(last, "tool_calls", None):
        # Agent is still calling tools. Not a final draft.
        return "end"
    return "verify"


def should_route_from_verification(state) -> str:
    """Conditional edge from verification node back to agent or to end.

    If the verification node appended a correction message (HumanMessage
    with a [VERIFICATION FAILED ...] marker), route back to agent.
    Otherwise end.
    """
    messages = state.get("messages") or []
    if not messages:
        return "end"
    last = messages[-1]
    if isinstance(last, HumanMessage) and "[VERIFICATION FAILED" in str(last.content):
        return "agent"
    return "end"
