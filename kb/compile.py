"""Compile resolved support tickets → troubleshooting PLAYBOOKS (in-boundary Gemini).

Option C semantics: a resolved GitHub issue / Smartsheet ticket is a *historical
work-log record*, NOT a fact about the product. So we do NOT transcribe one
article per issue. Instead:

  (A) DISTILL: per-doc we extract a troubleshooting SIGNAL (area, symptom,
      root_cause, resolution, status); per *area* we SYNTHESIZE a durable
      troubleshooting playbook (common symptoms → likely causes → resolution
      steps) plus a dated, source-cited "past resolved issues" list. Resolutions
      are framed as HISTORICAL, never as current product facts.
  (B) The raw tickets stay in ``support/raw/`` as the searchable log; agents drill
      into specifics via the playbook's issue refs + the existing GitHub tools.

Bounded + resumable + single-flight (same machinery as before): per-doc manifest
+ signals are checkpointed after every doc; playbooks regenerate per "dirty" area
and clear as written, so interruptions resume with no re-extraction.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date

from kb import storage
from kb.gemini import get_kb_llm, kb_engine_info

RAW_PREFIX = "support/raw/"
PLAYBOOK_PREFIX = "support/playbooks/"
INDEX_PATH = "support/playbooks/index.md"
MANIFEST_PATH = "support/playbooks/.manifest.json"
SIGNALS_PATH = "support/playbooks/.signals.json"
_INTERNAL = (MANIFEST_PATH, SIGNALS_PATH)

_EXTRACT_SYS = (
    "You read ONE resolved VirtualDojo support ticket / GitHub issue — a HISTORICAL "
    "record of a past problem and its fix, NOT a statement of current product "
    "behavior. Return ONLY a JSON object: area (lowercase-hyphen product-area slug "
    "<=48 chars, e.g. 'quote-importer', 'ui-readability', 'rfx-filtering'), symptom "
    "(what was reported/observed), root_cause (why, if stated, else ''), resolution "
    "(what fixed it, if stated, else ''), status (resolved|in-progress|wont-fix|"
    "unknown), sensitive (true if it contains financials/legal/PII/secrets). OMIT "
    "any sensitive values. Facts only; empty string for anything not in the ticket."
)

_PLAYBOOK_SYS = (
    "You write ONE troubleshooting PLAYBOOK for a VirtualDojo support area, in "
    "markdown, SYNTHESIZED across MULTIPLE resolved tickets — durable guidance for a "
    "support agent, NOT a restatement of individual tickets. Structure: a short H1; "
    "'## Common symptoms'; '## Likely causes'; '## Resolution steps' (generalized "
    "from how these were resolved). Then '## Past resolved issues (historical)' — a "
    "brief, dated, source-cited bullet list. CRITICAL: frame everything as "
    "troubleshooting patterns and HISTORICAL resolutions — NEVER assert a past fix "
    "as current product behavior (it may have changed since). Ground ONLY in the "
    "provided signals; no speculation; no financials/PII/secrets. Output ONLY the "
    "markdown body (frontmatter is added programmatically)."
)


def _json_from(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _llm_text(llm, system: str, user: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def _frontmatter(title: str, sources: list[str], confidence: str) -> str:
    src = "[" + ", ".join(sorted(set(sources))[:40]) + "]"
    return (
        "---\n"
        "kind: troubleshooting-playbook\n"
        f"title: {title}\n"
        f"sources: {src}\n"
        f"last_verified: {date.today().isoformat()}\n"
        f"confidence: {confidence}\n"
        "---\n\n"
    )


def _load_json(path: str, default):
    raw = storage.read_text(path)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9-]", "", str(s or "").lower())[:48]


def _write_playbook(llm, area: str, rec: dict) -> None:
    items = rec.get("items", [])
    lines = []
    for it in items[:60]:
        lines.append(
            f"- source={it.get('source','?')} status={it.get('status','?')} "
            f"symptom={it.get('symptom','')!r} cause={it.get('root_cause','')!r} "
            f"resolution={it.get('resolution','')!r}"
        )
    prompt = f"Area: {area}\nTitle: {rec.get('title') or area}\n\nSignals:\n" + "\n".join(lines)
    body = _llm_text(llm, _PLAYBOOK_SYS, prompt).strip()
    confidence = "high" if len(items) >= 3 else "needs-review"
    sources = [it.get("source", "") for it in items if it.get("source")]
    storage.write_text(
        f"{PLAYBOOK_PREFIX}{area}.md",
        _frontmatter(rec.get("title") or area, sources, confidence) + body + "\n",
    )


def _refresh_index() -> int:
    playbooks = sorted(
        p for p in storage.list_paths(PLAYBOOK_PREFIX)
        if p.endswith(".md") and not p.endswith("index.md")
    )
    lines = [
        "# Support Troubleshooting Playbooks",
        "",
        "Auto-synthesized from resolved tickets by in-boundary Gemini. These are "
        "troubleshooting PATTERNS and HISTORICAL resolutions, not current product "
        "facts. Do not edit by hand.",
        "",
    ]
    for p in playbooks:
        lines.append(f"- [[{p[len(PLAYBOOK_PREFIX):-3]}]]")
    storage.write_text(INDEX_PATH, "\n".join(lines) + "\n")
    return len(playbooks)


def compile_support(llm=None, max_docs: int | None = None) -> dict:
    """Distill up to ``max_docs`` new/changed tickets into per-area playbooks.

    Resumable + bounded. Returns content-free stats.
    """
    llm = llm or get_kb_llm()
    manifest: dict = _load_json(MANIFEST_PATH, {})
    state: dict = _load_json(SIGNALS_PATH, {})
    areas: dict = state.get("areas", {})
    dirty: set = set(state.get("dirty", []))

    def _save():
        storage.write_text(MANIFEST_PATH, json.dumps(manifest), content_type="application/json")
        storage.write_text(
            SIGNALS_PATH,
            json.dumps({"areas": areas, "dirty": sorted(dirty)}),
            content_type="application/json",
        )

    docs = [
        (p, t) for p, t in storage.list_text(RAW_PREFIX)
        if "/.state/" not in p and p not in _INTERNAL
    ]
    pending = [(p, t, hashlib.sha256(t.encode("utf-8")).hexdigest()) for p, t in docs]
    pending = [(p, t, h) for p, t, h in pending if manifest.get(p) != h]
    batch = pending[:max_docs] if max_docs else pending

    # --- Extract phase: one troubleshooting signal per ticket (checkpointed) ---
    processed = flagged = 0
    for path, text, h in batch:
        data = _json_from(_llm_text(llm, _EXTRACT_SYS, text)) or {}
        if data.get("sensitive"):
            flagged += 1
        area = _slugify(data.get("area"))
        if area:
            rec = areas.setdefault(area, {"title": area.replace("-", " ").title(), "items": []})
            rec["items"].append({
                "source": path.rsplit("/", 1)[-1],
                "symptom": data.get("symptom", ""),
                "root_cause": data.get("root_cause", ""),
                "resolution": data.get("resolution", ""),
                "status": data.get("status", "unknown"),
            })
            dirty.add(area)
        manifest[path] = h
        processed += 1
        _save()  # checkpoint every doc → interruption-safe

    # --- Synthesis phase: regenerate each dirty area's playbook ---
    playbooks_written = 0
    for area in sorted(dirty):
        if area not in areas:
            dirty.discard(area)
            continue
        _write_playbook(llm, area, areas[area])
        dirty.discard(area)
        playbooks_written += 1
        _save()

    total = _refresh_index()
    stats = {
        "engine": kb_engine_info(),
        "raw_docs_seen": len(docs),
        "raw_docs_processed": processed,
        "sensitive_flagged_omitted": flagged,
        "playbooks_written": playbooks_written,
        "playbooks_total": total,
        "docs_remaining": max(0, len(pending) - processed),
    }
    print(
        f"[kb.compile] batch: processed={processed} playbooks={playbooks_written} "
        f"remaining={stats['docs_remaining']} total_playbooks={total}",
        flush=True,
    )
    return stats
