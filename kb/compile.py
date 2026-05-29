"""Compile ``support/raw/`` → ``support/wiki/`` with in-boundary Gemini.

Designed to run as a small, resumable, bounded in-process job on the serving
instance, tolerant of Cloud Run interruptions (deploys/drains/SIGKILL):

- **Checkpoint as you go (#1):** the manifest (processed doc → hash) and the
  accumulated knowledge units are persisted after EVERY doc, and articles are
  regenerated per "dirty" topic and cleared as they're written. So an interrupted
  run resumes exactly where it stopped — no re-extraction, no doom loop.
- **Bounded batches (#2):** each call processes at most ``max_docs`` unprocessed
  docs, so an interruption costs ≤ one small batch and the compile never hogs the
  instance. Repeated ticks converge to a fully compiled wiki.
- Single-flight locking (#3) is handled by the caller (``kb.run``).

Compliance: the only LLM is ``kb.gemini.get_kb_llm`` (regional Vertex, in-boundary).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date

from kb import storage
from kb.gemini import get_kb_llm, kb_engine_info

RAW_PREFIX = "support/raw/"
WIKI_PREFIX = "support/wiki/"
INDEX_PATH = "support/index.md"
MANIFEST_PATH = "support/wiki/.manifest.json"
UNITS_PATH = "support/wiki/.units.json"
_INTERNAL = (MANIFEST_PATH, UNITS_PATH)

_EXTRACT_SYS = (
    "You extract knowledge from a single VirtualDojo support source document. "
    "Return ONLY a JSON object with keys: topic_slug (lowercase-hyphen, <=48 chars), "
    "title (short), one_line (third person), key_facts (array of short factual "
    "strings grounded ONLY in the document), sensitive (true if the doc contains "
    "financials, legal, PII, or secrets), sensitive_kinds (array). OMIT any "
    "sensitive values from key_facts — never echo them. Facts only; no speculation."
)

_ARTICLE_SYS = (
    "You write ONE concise, third-person VirtualDojo support knowledge article in "
    "markdown, grounded ONLY in the provided facts (facts-only rule — no "
    "speculation, no invented specifics). Do NOT include any financials, legal, "
    "PII, or secrets. Start with a short H1, then tight prose/bullets, and end with "
    "a '## Related' section linking sibling topics as [[slug]]. Output ONLY the "
    "article body (no frontmatter — it is added programmatically)."
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
    src = "[" + ", ".join(sorted(set(sources))) + "]"
    return (
        "---\n"
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


def _write_article(llm, slug: str, unit: dict, siblings: list[str]) -> None:
    related = [s for s in siblings if s != slug][:8]
    prompt = (
        f"Topic slug: {slug}\nTitle: {unit.get('title') or slug}\n"
        f"Sibling topics (for [[backlinks]]): {related}\n\n"
        "Facts (grounded sources):\n- " + "\n- ".join(unit.get("facts", [])[:60])
    )
    body = _llm_text(llm, _ARTICLE_SYS, prompt).strip()
    confidence = "high" if len(unit.get("facts", [])) >= 3 else "needs-review"
    article = _frontmatter(unit.get("title") or slug, unit.get("sources", []), confidence) + body + "\n"
    storage.write_text(f"{WIKI_PREFIX}{slug}.md", article)


def _refresh_index() -> int:
    articles = sorted(
        p for p in storage.list_paths(WIKI_PREFIX)
        if p.endswith(".md") and not p.endswith("index.md")
    )
    lines = [
        "# Support Knowledge Index",
        "",
        "Auto-maintained by the in-boundary Gemini compile. Do not edit by hand.",
        "",
    ]
    for p in articles:
        lines.append(f"- [[{p[len(WIKI_PREFIX):-3]}]]")
    storage.write_text(INDEX_PATH, "\n".join(lines) + "\n")
    return len(articles)


def compile_support(llm=None, max_docs: int | None = None) -> dict:
    """Process up to ``max_docs`` new/changed raw docs into the wiki, resumably.

    Returns content-free stats. Safe to call repeatedly: each call advances the
    manifest and converges the wiki; interruptions lose at most the in-flight doc.
    """
    llm = llm or get_kb_llm()
    manifest: dict = _load_json(MANIFEST_PATH, {})
    state: dict = _load_json(UNITS_PATH, {})
    units: dict = state.get("units", {})
    dirty: set = set(state.get("dirty", []))

    def _save():
        storage.write_text(MANIFEST_PATH, json.dumps(manifest), content_type="application/json")
        storage.write_text(
            UNITS_PATH,
            json.dumps({"units": units, "dirty": sorted(dirty)}),
            content_type="application/json",
        )

    docs = [
        (p, t) for p, t in storage.list_text(RAW_PREFIX)
        if "/.state/" not in p and p not in _INTERNAL
    ]
    pending = [(p, t, hashlib.sha256(t.encode("utf-8")).hexdigest()) for p, t in docs]
    pending = [(p, t, h) for p, t, h in pending if manifest.get(p) != h]
    batch = pending[: max_docs] if max_docs else pending

    # --- Extract phase: per-doc, checkpoint after each (resumable) ---
    processed = flagged = 0
    for path, text, h in batch:
        data = _json_from(_llm_text(llm, _EXTRACT_SYS, text)) or {}
        if data.get("sensitive"):
            flagged += 1
        slug = _slugify(data.get("topic_slug"))
        if slug:
            u = units.setdefault(slug, {"title": data.get("title") or slug, "facts": [], "sources": []})
            for f in (data.get("key_facts") or []):
                if f and f not in u["facts"]:
                    u["facts"].append(f)
            if path not in u["sources"]:
                u["sources"].append(path)
            dirty.add(slug)
        manifest[path] = h
        processed += 1
        _save()  # checkpoint every doc → interruption-safe

    # --- Article phase: regenerate each dirty topic, clear as written ---
    articles_written = 0
    siblings = sorted(units.keys())
    for slug in sorted(dirty):
        if slug not in units:
            dirty.discard(slug)
            continue
        _write_article(llm, slug, units[slug], siblings)
        dirty.discard(slug)
        articles_written += 1
        _save()  # checkpoint after each article → resume skips done ones

    total_articles = _refresh_index()

    stats = {
        "engine": kb_engine_info(),
        "raw_docs_seen": len(docs),
        "raw_docs_processed": processed,
        "sensitive_flagged_omitted": flagged,
        "articles_written": articles_written,
        "articles_total": total_articles,
        "docs_remaining": max(0, len(pending) - processed),
    }
    print(
        f"[kb.compile] batch: processed={processed} wrote={articles_written} "
        f"remaining={stats['docs_remaining']} total_articles={total_articles}",
        flush=True,
    )
    return stats
