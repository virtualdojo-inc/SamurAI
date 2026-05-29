"""Compile ``support/raw/`` → ``support/wiki/`` with in-boundary Gemini.

Map-reduce over the raw sources:
  1. EXTRACT (map): per raw doc, Gemini pulls a topic slug + one-liner + key facts,
     and flags any financials/PII/secrets (which are omitted, per the README).
  2. CLUSTER: group knowledge units by topic slug.
  3. WRITE (reduce): per topic, Gemini writes a concise, third-person article
     grounded ONLY in the clustered facts, with provenance frontmatter
     (title, sources, last_verified, confidence) and [[backlinks]].
  4. INDEX: refresh ``support/index.md``.

Compliance: the only LLM is ``kb.gemini.get_kb_llm`` (regional Vertex, in-boundary).
``conversation-history/`` may be read for continuity but is NEVER cited as a
source (echo-chamber guard). All stats returned are content-free (counts/paths).
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
    "PII, or secrets; if a fact looks sensitive, omit it. Start with a short H1, "
    "then tight prose/bullets. End with a '## Related' section linking sibling "
    "topics as [[slug]]. Output ONLY the article body (no frontmatter — it is "
    "added programmatically)."
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


def compile_support(llm=None, max_docs: int | None = None) -> dict:
    """Compile new/changed raw support docs into the wiki. Content-free stats."""
    llm = llm or get_kb_llm()
    manifest_raw = storage.read_text(MANIFEST_PATH)
    manifest = json.loads(manifest_raw) if manifest_raw else {}

    # 1. EXTRACT — only new/changed raw docs (skip state/manifest files).
    docs = [
        (p, t)
        for p, t in storage.list_text(RAW_PREFIX)
        if "/.state/" not in p and not p.endswith(".manifest.json")
    ]
    units: list[dict] = []
    processed = flagged = 0
    for path, text in docs:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if manifest.get(path) == h:
            continue  # unchanged since last compile (idempotent)
        if max_docs and processed >= max_docs:
            break
        data = _json_from(_llm_text(llm, _EXTRACT_SYS, text)) or {}
        manifest[path] = h
        processed += 1
        if data.get("sensitive"):
            flagged += 1
        slug = data.get("topic_slug")
        if not slug:
            continue
        units.append(
            {
                "slug": re.sub(r"[^a-z0-9-]", "", str(slug).lower())[:48],
                "title": data.get("title") or slug,
                "facts": [f for f in (data.get("key_facts") or []) if f],
                "source": path,
            }
        )

    # 2. CLUSTER by slug.
    clusters: dict[str, dict] = {}
    for u in units:
        c = clusters.setdefault(u["slug"], {"title": u["title"], "facts": [], "sources": []})
        c["facts"].extend(u["facts"])
        c["sources"].append(u["source"])

    # 3. WRITE one article per cluster.
    siblings = sorted(clusters.keys())
    written: list[str] = []
    for slug, c in clusters.items():
        related = [s for s in siblings if s != slug][:8]
        prompt = (
            f"Topic slug: {slug}\nTitle: {c['title']}\n"
            f"Sibling topics (for [[backlinks]]): {related}\n\n"
            "Facts (grounded sources):\n- " + "\n- ".join(c["facts"][:60])
        )
        body = _llm_text(llm, _ARTICLE_SYS, prompt).strip()
        confidence = "high" if len(c["facts"]) >= 3 else "needs-review"
        article = _frontmatter(c["title"], c["sources"], confidence) + body + "\n"
        storage.write_text(f"{WIKI_PREFIX}{slug}.md", article)
        written.append(slug)

    # 4. INDEX (regenerate from current wiki articles).
    all_articles = sorted(
        p for p in storage.list_paths(WIKI_PREFIX)
        if p.endswith(".md") and not p.endswith("index.md")
    )
    index_lines = [
        "# Support Knowledge Index",
        "",
        "Auto-maintained by the in-boundary Gemini compile. Do not edit by hand.",
        "",
    ]
    for p in all_articles:
        slug = p[len(WIKI_PREFIX):-3]
        index_lines.append(f"- [[{slug}]]")
    storage.write_text(INDEX_PATH, "\n".join(index_lines) + "\n")

    storage.write_text(MANIFEST_PATH, json.dumps(manifest), content_type="application/json")

    return {
        "engine": kb_engine_info(),
        "raw_docs_seen": len(docs),
        "raw_docs_processed": processed,
        "sensitive_flagged_omitted": flagged,
        "articles_written": len(written),
        "articles_total": len(all_articles),
    }
