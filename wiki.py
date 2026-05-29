"""Knowledge wiki loader for SamurAI.

Knowledge now lives in the in-boundary bucket ``gs://virtualdojo-knowledge``,
mounted read-only on the bot via GCS FUSE at ``KB_KNOWLEDGE_ROOT`` (default
``/knowledge``). The bot serves the scopes it has read access to:
``engineering/``, ``support/``, ``customers/onboarding/`` — reading each scope's
``wiki/*.md``. Because it reads the live mount, nightly compile updates are
picked up WITHOUT a redeploy.

A transition fallback to the repo's ``knowledge/`` directory remains until that
directory is retired (so local/dev and the pre-mount window keep working).

Progressive disclosure (unchanged): every article's title + summary is injected
into the system prompt via :func:`knowledge_index_text`; full bodies load on
demand via ``read_knowledge``; ``search_wiki`` does a naive keyword search.

Frontmatter is schema-tolerant: ``title`` is required (or derived from the first
H1); ``summary`` is used if present, else derived from the body. This accepts
both migrated articles (title+summary) and compile output (title + sources +
last_verified + confidence).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import yaml
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# GCS FUSE mount of gs://virtualdojo-knowledge (read what we serve).
KB_KNOWLEDGE_ROOT = os.environ.get("KB_KNOWLEDGE_ROOT", "/knowledge")
SERVE_SCOPES = ["engineering", "support", "customers/onboarding"]

# Repo skills are still served from the repo; only knowledge moved to the bucket.
SKILLS_DIR = Path(__file__).parent / "skills"
# Transition fallback only — removed when repo knowledge/ is retired.
_REPO_KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
_MAX_SUMMARY = 1024

_catalog_cache: list[dict] | None = None


def _scope_wiki_dirs() -> list[Path]:
    root = Path(KB_KNOWLEDGE_ROOT)
    dirs = [root / scope / "wiki" for scope in SERVE_SCOPES]
    return [d for d in dirs if d.is_dir()]


def _derive_summary(body: str, title: str) -> str:
    for line in body.splitlines():
        s = line.strip()
        if s and not s.startswith("#") and not s.startswith("---"):
            return s[:200]
    return title


def _parse_article(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("[wiki] could not read %s (%s); skipping", path, e)
        return None

    meta: dict = {}
    body = text
    if text.lstrip().startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                loaded = yaml.safe_load(parts[1])
                meta = loaded if isinstance(loaded, dict) else {}
            except yaml.YAMLError as e:
                logger.warning("[wiki] %s bad frontmatter (%s); using body only", path, e)
            body = parts[2].strip()

    title = str(meta.get("title") or "").strip()
    if not title:
        m = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        title = m.group(1).strip() if m else path.stem
    summary = str(meta.get("summary") or "").strip() or _derive_summary(body, title)
    if len(summary) > _MAX_SUMMARY:
        summary = summary[:_MAX_SUMMARY]
    return {"name": path.stem, "title": title, "summary": summary, "body": body}


def _article_paths() -> list[Path]:
    """Markdown article paths from the mounted scopes, or the repo fallback."""
    paths: list[Path] = []
    for d in _scope_wiki_dirs():
        for md in sorted(d.glob("*.md")):
            if md.name.lower() in ("index.md", ".keep"):
                continue
            paths.append(md)
    if paths:
        return paths
    # Transition fallback: repo knowledge/ (removed at retirement).
    if _REPO_KNOWLEDGE_DIR.is_dir():
        return [
            md for md in sorted(_REPO_KNOWLEDGE_DIR.glob("*.md"))
            if md.name != "INDEX.md"
        ]
    return []


def load_knowledge_catalog(force: bool = False) -> list[dict]:
    global _catalog_cache
    if _catalog_cache is not None and not force:
        return _catalog_cache
    seen: set[str] = set()
    articles: list[dict] = []
    for md in _article_paths():
        parsed = _parse_article(md)
        if parsed and parsed["name"] not in seen:
            seen.add(parsed["name"])
            articles.append(parsed)
    _catalog_cache = articles
    logger.info(
        "[wiki] loaded %d knowledge articles from %s: %s",
        len(articles), KB_KNOWLEDGE_ROOT, [a["name"] for a in articles],
    )
    return articles


def knowledge_index_text() -> str:
    articles = load_knowledge_catalog()
    if not articles:
        return ""
    lines = [
        "## Knowledge base",
        (
            "You maintain a knowledge wiki (in-boundary bucket). When a question "
            "relates to an article below, call `read_knowledge(name)`; use "
            "`search_wiki(query)` to search. Articles:"
        ),
    ]
    for a in articles:
        lines.append(f"- **{a['name']}** ({a['title']}) — {a['summary']}")
    return "\n".join(lines)


@tool
def read_knowledge(name: str) -> str:
    """Read a knowledge-base article in full.

    Call this when a question relates to an article listed under 'Knowledge base'.

    Args:
        name: The article name (filename stem), e.g. 'virtualdojo-infra'.
    """
    for a in load_knowledge_catalog():
        if a["name"] == name:
            return a["body"]
    available = ", ".join(a["name"] for a in load_knowledge_catalog()) or "(none)"
    return f"No knowledge article named '{name}'. Available: {available}"


@tool
def search_wiki(query: str, limit: int = 8) -> str:
    """Search the wiki (repo skills + bucket knowledge) for a keyword or phrase.

    Naive case-insensitive substring search; returns matching files with a short
    snippet. Use it to find relevant skills or knowledge before answering.

    Args:
        query: Keyword or phrase to search for.
        limit: Max number of matching files to return (default 8).
    """
    q = (query or "").strip().lower()
    if not q:
        return "Provide a non-empty query."
    hits: list[str] = []
    search_dirs = [SKILLS_DIR, *_scope_wiki_dirs()]
    if len(search_dirs) == 1 and _REPO_KNOWLEDGE_DIR.is_dir():
        search_dirs.append(_REPO_KNOWLEDGE_DIR)  # transition fallback
    for base in search_dirs:
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
                snippet = text[max(0, idx - 120): idx + len(q) + 120].replace("\n", " ").strip()
                hits.append(f"- {md.name}: …{snippet}…")
                if len(hits) >= limit:
                    break
        if len(hits) >= limit:
            break
    return "Wiki matches:\n" + "\n".join(hits) if hits else f"No wiki matches for '{query}'."


WIKI_TOOLS = [read_knowledge, search_wiki]
