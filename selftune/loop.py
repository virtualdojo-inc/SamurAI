"""Step 3: the autonomous propose → evaluate → promote loop for learned_hints.md.

One cycle:
  1. PROPOSE (in-boundary regional Gemini): read recent failures + good exemplars
     + the current hints → draft a small edit to the hints doc. Generic
     operational/tool-routing guidance only — no PII/customer specifics.
  2. EVALUATE (production model, choice A): score the eval set under current vs.
     candidate hints via selftune.score (objective tool-selection, no self-judging).
  3. PROMOTE: selftune.score.gate decides — promote only on no safety regression
     AND (better steering OR fewer tokens). save_hints versions it for rollback.

No human in the loop: the eval gate + the frozen core + versioned rollback are
the safety. Adaptive cadence: runs daily while it's still accepting changes; after
a convergence streak it self-gates to weekly (no scheduler gymnastics). Won't run
until there are enough eval cases to be meaningful. Kill switch: KB_TUNE_ENABLED.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from selftune import evalset, score
from selftune.hints import load_hints, save_hints

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("SAMURAI_DATA_DIR", "/data")
STATE_PATH = Path(DATA_DIR) / "selftune" / "tune_state.json"
MIN_CASES = int(os.environ.get("KB_TUNE_MIN_CASES", "15"))
CONVERGED_STREAK = int(os.environ.get("KB_TUNE_CONVERGED_STREAK", "5"))
MAX_EVAL_CASES = int(os.environ.get("KB_TUNE_MAX_CASES", "60"))
EVAL_MODEL = os.environ.get("KB_TUNE_EVAL_MODEL", "gemini-3.5-flash")  # choice A: production

_PROPOSE_SYS = (
    "You improve a SUPPORT AGENT's 'learned operational guidance' — a short doc "
    "appended to its system prompt that steers WHICH TOOLS it calls and WHEN, and "
    "keeps it from stalling. You will see recent FAILURES (turns where a tool "
    "errored or the agent gave up) and GOOD examples (messages + the tools that "
    "worked), plus the CURRENT guidance. Propose the FULL revised guidance doc.\n"
    "RULES: make ONE small, focused improvement; keep it concise (every token is "
    "injected every turn); write generic tool-routing/usage guidance only — NO "
    "customer names, data, or specifics; NEVER weaken safety/autonomy rules; do "
    "not restate the core prompt. Output ONLY the markdown guidance, nothing else."
)


def tune_enabled() -> bool:
    return os.environ.get("KB_TUNE_ENABLED", "off").lower() != "off"


def _load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.is_file() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
    except OSError as e:  # pragma: no cover
        logger.warning("[selftune] state save failed: %s", e)


def _should_run(state: dict, force: bool) -> bool:
    """Adaptive cadence: always while improving; after convergence, weekly only."""
    if force:
        return True
    converged = state.get("streak", 0) >= CONVERGED_STREAK
    is_weekly_slot = datetime.now(timezone.utc).weekday() == 0  # Monday
    return (not converged) or is_weekly_slot


def _bump(state: dict, accepted: bool) -> None:
    state["runs"] = state.get("runs", 0) + 1
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    if accepted:
        state["streak"] = 0
        state["accepts"] = state.get("accepts", 0) + 1
        state["last_accepted"] = state["last_run"]
    else:
        state["streak"] = state.get("streak", 0) + 1
    _save_state(state)


def _make_select_fn():
    """Production-model tool-selection predictor (choice A). Reuses the agent's
    exact prompt + tool gating so the eval reflects real behavior."""
    import agent  # lazy: avoids import cycle (agent imports selftune.hints)
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = ChatGoogleGenerativeAI(model=EVAL_MODEL, **agent._GCP_KWARGS)

    def select(hints_text: str, message: str) -> list:
        system = agent._select_prompt_sections(message, hints_override=hints_text)
        tools = agent._select_tool_groups(message)
        try:
            resp = model.bind_tools(tools).invoke(
                [SystemMessage(content=system), HumanMessage(content=message)]
            )
            return [tc.get("name") for tc in (getattr(resp, "tool_calls", None) or [])]
        except Exception as e:  # a flaky eval call shouldn't crash the cycle
            logger.warning("[selftune] select_fn error: %s", e)
            return []

    return select


def _make_propose_fn():
    """In-boundary regional Gemini that drafts the candidate hints doc."""
    from kb.gemini import get_kb_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_kb_llm()

    def propose(current_hints: str, good_cases: list[dict], failures: list[dict]) -> str:
        ctx = (
            "CURRENT GUIDANCE:\n" + (current_hints or "(empty)") + "\n\n"
            "RECENT FAILURES (fix these):\n"
            + "\n".join(f"- {f['message']!r} errored={f['errored_tools']} gave_up={f['gave_up']}"
                        for f in failures[:20])
            + "\n\nGOOD EXAMPLES (message → tools that worked):\n"
            + "\n".join(f"- {c['message']!r} → {c['expected_tools']}"
                        for c in good_cases[:20] if c.get("source") == "mined")
        )
        resp = llm.invoke([SystemMessage(content=_PROPOSE_SYS), HumanMessage(content=ctx)])
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    return propose


def run_tuning_cycle(force: bool = False, propose_fn=None, select_fn=None, cases=None) -> dict:
    """One propose→evaluate→promote cycle. Content-free stats. Never raises out."""
    if not force and not tune_enabled():
        return {"skipped": "disabled"}
    state = _load_state()
    if not _should_run(state, force):
        print(f"[selftune] converged (streak={state.get('streak')}) — weekly-gated, skipping.", flush=True)
        return {"skipped": "converged-weekly-gate"}

    cases = cases if cases is not None else evalset.build_eval_set(max_mined=MAX_EVAL_CASES)
    if len(cases) < MIN_CASES:
        print(f"[selftune] only {len(cases)} eval cases (< {MIN_CASES}) — skipping.", flush=True)
        return {"skipped": "insufficient_cases", "n": len(cases)}

    current = load_hints(force=True)
    propose_fn = propose_fn or _make_propose_fn()
    select_fn = select_fn or _make_select_fn()
    failures = evalset.mine_failures(evalset.read_raw_turns(days=7))

    candidate = (propose_fn(current, cases, failures) or "").strip()
    if not candidate or candidate == (current or "").strip():
        _bump(state, accepted=False)
        print("[selftune] no candidate change proposed.", flush=True)
        return {"result": "no_change", "n": len(cases)}

    cur = score.score_prompt(cases, select_fn, current)
    cand = score.score_prompt(cases, select_fn, candidate)
    promote, reason = score.gate(cur, cand)
    if promote:
        save_hints(candidate)
    _bump(state, accepted=promote)

    stats = {
        "result": "promoted" if promote else "rejected",
        "reason": reason,
        "current": {k: cur[k] for k in ("pass_rate", "must_pass_ok", "token_estimate")},
        "candidate": {k: cand[k] for k in ("pass_rate", "must_pass_ok", "token_estimate")},
        "n": len(cases),
        "streak": state.get("streak"),
    }
    print(f"[selftune] cycle complete: {json.dumps(stats)}", flush=True)
    return stats
