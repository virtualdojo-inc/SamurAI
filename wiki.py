"""Knowledge wiki for SamurAI (the committed, LLM-maintained knowledge base).

Companion to ``skills.py``. Where a *skill* is procedural know-how, a *knowledge
article* is conceptual/reference knowledge — facts about VirtualDojo infra, the
deploy pipeline, recurring decisions, etc. Articles are markdown under
``knowledge/`` with YAML frontmatter (``title`` + ``summary``) and Obsidian-style
``[[wikilinks]]`` backlinks. ``knowledge/INDEX.md`` is an auto-maintained index.

Progressive disclosure (same as skills):
  Level 1 — every article's title + summary, injected into the system prompt via
            :func:`knowledge_index_text`. Cheap; enough to know what exists.
  Level 2 — full article body via the ``read_knowledge`` tool; ``search_wiki``
            does a naive keyword search across skills/ + knowledge/.

The nightly ``wiki-compile`` job (and the weekly health-check) curate this wiki
from raw conversations. It is committed to git, so it is versioned and reviewable
— unlike the runtime LangMem vector store, which it complements.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
SKILLS_DIR = Path(__file__).parent / "skills"
_INDEX_NAME = "INDEX.md"
_MAX_SUMMARY = 1024

_catalog_cache: list[dict] | None = None


def _parse_article(path: Path) -> dict | None:
    """Parse a knowledge article into ``{name, title, summary, body}``.

    ``name`` is the filename stem (used by ``read_knowledge``). Returns ``None``
    (logged) for malformed articles so one bad file can't crash startup.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("[wiki] could not read %s (%s); skipping", path, e)
        return None
    if not text.lstrip().startswith("---"):
        logger.warning("[wiki] %s missing frontmatter; skipping", path)
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        logger.warning("[wiki] %s malformed frontmatter; skipping", path)
        return None
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        logger.warning("[wiki] %s bad YAML frontmatter (%s); skipping", path, e)
        return None
    if not isinstance(meta, dict):
        logger.warning("[wiki] %s frontmatter is not a mapping; skipping", path)
        return None

    title = str(meta.get("title") or "").strip()
    summary = str(meta.get("summary") or "").strip()
    body = parts[2].strip()
    if not title or not summary or len(summary) > _MAX_SUMMARY:
        logger.warning("[wiki] %s missing/invalid title or summary; skipping", path)
        return None
    return {"name": path.stem, "title": title, "summary": summary, "body": body}


def load_knowledge_catalog(force: bool = False) -> list[dict]:
    """Load and cache all valid knowledge articles (excluding INDEX.md)."""
    global _catalog_cache
    if _catalog_cache is not None and not force:
        return _catalog_cache
    articles: list[dict] = []
    if KNOWLEDGE_DIR.is_dir():
        for md in sorted(KNOWLEDGE_DIR.glob("*.md")):
            if md.name == _INDEX_NAME:
                continue
            parsed = _parse_article(md)
            if parsed is not None:
                articles.append(parsed)
    _catalog_cache = articles
    logger.info(
        "[wiki] loaded %d knowledge articles: %s",
        len(articles),
        [a["name"] for a in articles],
    )
    return articles


def knowledge_index_text() -> str:
    """Compact title+summary index for the system prompt (level-1 disclosure)."""
    articles = load_knowledge_catalog()
    if not articles:
        return ""
    lines = [
        "## Knowledge base",
        (
            "You maintain a knowledge wiki. When a question relates to an article "
            "below, call `read_knowledge(name)` to load it; use `search_wiki(query)` "
            "to search across skills and knowledge. Articles:"
        ),
    ]
    for a in articles:
        lines.append(f"- **{a['name']}** ({a['title']}) — {a['summary']}")
    return "\n".join(lines)


@tool
def read_knowledge(name: str) -> str:
    """Read a knowledge-base article in full.

    Call this when a question relates to one of the articles listed under
    'Knowledge base' in your system prompt.

    Args:
        name: The article name (filename stem), e.g. 'virtualdojo-infra'.
    """
    for a in load_knowledge_catalog():
        if a["name"] == name:
            # Article bodies carry their own H1 heading by convention.
            return a["body"]
    available = ", ".join(a["name"] for a in load_knowledge_catalog()) or "(none)"
    return f"No knowledge article named '{name}'. Available: {available}"


@tool
def search_wiki(query: str, limit: int = 8) -> str:
    """Search the wiki (skills/ + knowledge/) for a keyword or phrase.

    A naive case-insensitive substring search over all markdown files; returns
    matching files with a short context snippet. Use it to find relevant
    procedural skills or knowledge articles before answering a broad question.

    Args:
        query: Keyword or phrase to search for.
        limit: Max number of matching files to return (default 8).
    """
    q = (query or "").strip().lower()
    if not q:
        return "Provide a non-empty query."
    hits: list[str] = []
    for base in (SKILLS_DIR, KNOWLEDGE_DIR):
        if not base.is_dir():
            continue
        for md in sorted(base.rglob("*.md")):
            try:
                text = md.read_text(encoding="utf-8")
            except OSError:
                continue
            lower = text.lower()
            if q in lower:
                idx = lower.index(q)
                start = max(0, idx - 120)
                end = min(len(text), idx + len(q) + 120)
                snippet = text[start:end].replace("\n", " ").strip()
                rel = md.relative_to(Path(__file__).parent)
                hits.append(f"- {rel}: …{snippet}…")
                if len(hits) >= limit:
                    break
        if len(hits) >= limit:
            break
    if not hits:
        return f"No wiki matches for '{query}'."
    return "Wiki matches:\n" + "\n".join(hits)


WIKI_TOOLS = [read_knowledge, search_wiki]
