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
EVAL_MODEL = os.environ.get("KB_TUNE_EVAL_MODEL", "gemini-3.6-flash")  # choice A: production
K_CANDIDATES = int(os.environ.get("KB_TUNE_CANDIDATES", "3"))  # propose N, gate the best

# Diversity steers so K candidates explore different edits (EvoPrompt-style, no
# extra search machinery). Indexed by candidate number.
_VARIANTS = (
    "Make the single highest-impact fix.",
    "Take a different angle — address a root cause, not the obvious surface fix.",
    "Be maximally concise: the smallest edit that still helps.",
)

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

    # temperature=0 → reproducible scoring, so a pass-rate delta reflects the hint
    # edit, not sampling noise (the old unpinned default made promotions coin-flips).
    model = ChatGoogleGenerativeAI(model=EVAL_MODEL, temperature=0, **agent._GCP_KWARGS)

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

    def propose(current_hints, good_cases, failures, score_fails=None, variant=0):
        eval_fails = score_fails or []
        ctx = (
            "CURRENT GUIDANCE:\n" + (current_hints or "(empty)") + "\n\n"
            "RECENT FAILURES (fix these):\n"
            + "\n".join(
                f"- {f['message']!r} errored={f['errored_tools']} gave_up={f['gave_up']}"
                + (f" user_said={f['category']}" if f.get("category") else "")
                + (f" note={f['note']!r}" if f.get("note") else "")
                for f in failures[:20]
            )
            + "\n\nWHERE THE CURRENT GUIDANCE FAILS THE EVAL (these are what to move):\n"
            + "\n".join(
                f"- [{f['bucket']}] {f['message']!r}"
                + (" — should commit to a tool but did not"
                   if f.get("require_any_tool")
                   else f" expected={f['expected']} selected={f['selected']}")
                for f in eval_fails[:20]
            )
            + "\n\nGOOD EXAMPLES (message → tools that worked):\n"
            + "\n".join(f"- {c['message']!r} → {c['expected_tools']}"
                        for c in good_cases[:20]
                        if c.get("source") in ("mined", "human-verified"))
            + "\n\nAPPROACH FOR THIS REVISION: " + _VARIANTS[variant % len(_VARIANTS)]
        )
        resp = llm.invoke([SystemMessage(content=_PROPOSE_SYS), HumanMessage(content=ctx)])
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    return propose


def _safe_propose(propose_fn, current, good_cases, failures, score_fails, variant):
    """Call propose_fn with the rich signature; tolerate older 3-arg fakes (tests)."""
    try:
        return propose_fn(current, good_cases, failures, score_fails, variant)
    except TypeError:
        return propose_fn(current, good_cases, failures)


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

    # Score current FIRST so the proposer can reflect on what the eval fails (GEPA).
    cur = score.score_prompt(cases, select_fn, current)
    if not score.has_driver_coverage(cur):
        # Fail-safe, but the loop is inert until 👍 feedback or logged stalls exist.
        print("[selftune] no driver coverage (no human-verified/recovery cases) — "
              "nothing can promote this cycle.", flush=True)

    # Propose up to K diverse candidates; dedup identical / no-op proposals.
    seen = {(current or "").strip()}
    candidates: list[str] = []
    for i in range(K_CANDIDATES):
        cand_text = (_safe_propose(propose_fn, current, cases, failures, cur.get("fails"), i) or "").strip()
        if cand_text and cand_text not in seen:
            seen.add(cand_text)
            candidates.append(cand_text)
    if not candidates:
        _bump(state, accepted=False)
        print("[selftune] no candidate change proposed.", flush=True)
        return {"result": "no_change", "n": len(cases)}

    # Score each candidate; keep the best PROMOTABLE one (the weighted gate decides).
    scored = [(c, score.score_prompt(cases, select_fn, c)) for c in candidates]
    best = None
    for c, cstats in scored:
        ok, reason = score.gate(cur, cstats)
        if ok:
            # Rank by quality-adjusted gain so a precise candidate beats an
            # over-calling one that merely saturates recovery.
            gain = score.quality_gain(cur, cstats)
            if best is None or gain > best[3]:
                best = (c, cstats, reason, gain)

    promote = best is not None
    if promote:
        cand_text, cand, reason, _gain = best
        save_hints(cand_text)
    else:
        cand = scored[0][1]
        _ok, reason = score.gate(cur, cand)  # report a representative verdict
    _bump(state, accepted=promote)

    def _rates(s):
        out = {k: s[k] for k in ("pass_rate", "must_pass_ok", "token_estimate")}
        out["buckets"] = {b: round(v["pass_rate"], 3) for b, v in s.get("buckets", {}).items()}
        return out

    stats = {
        "result": "promoted" if promote else "rejected",
        "reason": reason,
        "candidates": len(candidates),
        "current": _rates(cur),
        "candidate": _rates(cand),
        "n": len(cases),
        "streak": state.get("streak"),
    }
    print(f"[selftune] cycle complete: {json.dumps(stats)}", flush=True)
    return stats
