"""Knowledge wiki loader for SamurAI — reads from the in-boundary bucket.

Knowledge lives in ``gs://virtualdojo-knowledge``. The bot reads it at runtime via
the google-cloud-storage client (``kb.storage``) using **prefix lists** over the
scopes it is granted (``engineering/``, ``support/``, ``customers/onboarding/``).

Why the client and not a GCS FUSE mount: gcsfuse requires blanket bucket-level
``storage.objects.list`` (a ``storageLayout`` probe), which conflicts with the
"no blanket read of other scopes" rule. A prefix-scoped client list is gated
correctly by the conditioned IAM, so the bot only ever reads the three granted
scopes. Reads are live (TTL-cached), so nightly compile updates are picked up
without a redeploy.

Progressive disclosure unchanged: title+summary injected into the prompt via
:func:`knowledge_index_text`; full bodies via ``read_knowledge``; ``search_wiki``
does a naive keyword search over repo skills + the cached knowledge bodies.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path

import yaml
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Scopes the bot is granted (conditioned objectViewer). For each scope we serve
# curated knowledge from these subdirs: wiki/ (reference articles), playbooks/
# (synthesized troubleshooting playbooks), troubleshooting/. Raw tickets are NOT
# served — they're a searchable log, not knowledge.
SERVE_SCOPES = ["engineering", "support", "customers/onboarding"]
SERVE_SUBDIRS = ["wiki", "playbooks", "troubleshooting"]
# Prefixes with unbounded article counts (kb/compile.py emits one playbook per
# area — hundreds of files). These are listed by NAME only: never bulk-downloaded,
# never given per-article index lines in the prompt. Bodies are fetched one at a
# time on demand by read_knowledge / search_wiki.
LAZY_PREFIXES = ["support/playbooks/"]
# Repo skills stay repo-local; only knowledge moved to the bucket.
SKILLS_DIR = Path(__file__).parent / "skills"
# Transition fallback to repo knowledge/ until it is retired.
_REPO_KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
_MAX_SUMMARY = 1024
_CACHE_TTL = 300  # seconds — balance freshness (nightly updates) vs per-turn latency

_catalog_cache: list[dict] | None = None
_cache_ts: float = 0.0
# stem -> full object path for lazily-served articles (refreshed with the catalog)
_lazy_names_cache: dict[str, str] = {}
# stem -> parsed article, populated one blob at a time as tools request them
_lazy_body_cache: dict[str, dict] = {}
_refresh_lock = threading.Lock()
_refreshing = False


def _derive_summary(body: str, title: str) -> str:
    for line in body.splitlines():
        s = line.strip()
        if s and not s.startswith("#") and not s.startswith("---"):
            return s[:200]
    return title


def _parse(path_name: str, text: str) -> dict | None:
    meta: dict = {}
    body = text
    if text.lstrip().startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                loaded = yaml.safe_load(parts[1])
                meta = loaded if isinstance(loaded, dict) else {}
            except yaml.YAMLError:
                meta = {}
            body = parts[2].strip()
    title = str(meta.get("title") or "").strip()
    if not title:
        m = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        title = m.group(1).strip() if m else Path(path_name).stem
    summary = str(meta.get("summary") or "").strip() or _derive_summary(body, title)
    return {"name": Path(path_name).stem, "title": title, "summary": summary[:_MAX_SUMMARY], "body": body}


def _load_from_bucket() -> tuple[list[dict], dict[str, str]]:
    """Read curated knowledge (<scope>/<subdir>/*.md) via the in-boundary client.

    Returns ``(articles, lazy_names)``: small curated prefixes are downloaded in
    full; :data:`LAZY_PREFIXES` are listed by name only (one paged list call, no
    body downloads) and returned as a ``stem -> object path`` map.
    """
    from kb import storage  # lazy: avoids hard dep at import / in tests

    articles: list[dict] = []
    lazy_names: dict[str, str] = {}
    seen: set[str] = set()
    for scope in SERVE_SCOPES:
        for sub in SERVE_SUBDIRS:
            prefix = f"{scope}/{sub}/"
            if prefix in LAZY_PREFIXES:
                try:
                    for path_name in storage.list_paths(prefix):
                        base = path_name.rsplit("/", 1)[-1].lower()
                        if base in ("index.md", ".keep") or not base.endswith(".md"):
                            continue
                        stem = Path(path_name).stem
                        if stem not in seen:
                            lazy_names[stem] = path_name
                except Exception as e:
                    logger.warning("[wiki] %s list failed: %s", prefix, e)
                continue
            try:
                items = storage.list_text(prefix)
            except Exception as e:
                logger.warning("[wiki] %s/%s read failed: %s", scope, sub, e)
                continue
            for path_name, text in items:
                base = path_name.rsplit("/", 1)[-1].lower()
                if base in ("index.md", ".keep") or not base.endswith(".md"):
                    continue
                parsed = _parse(path_name, text)
                if parsed and parsed["name"] not in seen:
                    seen.add(parsed["name"])
                    articles.append(parsed)
    return articles, lazy_names


def _load_from_repo_fallback() -> list[dict]:
    out: list[dict] = []
    if _REPO_KNOWLEDGE_DIR.is_dir():
        for md in sorted(_REPO_KNOWLEDGE_DIR.glob("*.md")):
            if md.name == "INDEX.md":
                continue
            try:
                out.append(_parse(md.name, md.read_text(encoding="utf-8")))
            except OSError:
                continue
    return out


def _refresh_catalog() -> list[dict]:
    """Synchronous full refresh — network I/O; call off the event loop."""
    global _catalog_cache, _cache_ts, _lazy_names_cache
    articles, lazy_names = _load_from_bucket()
    if not articles and not lazy_names:
        articles = _load_from_repo_fallback()  # transition safety
    _catalog_cache = articles
    _lazy_names_cache = lazy_names
    _cache_ts = time.time()
    logger.info(
        "[wiki] loaded %d knowledge articles (+%d lazy) from bucket: %s",
        len(articles), len(lazy_names), [a["name"] for a in articles],
    )
    return articles


def _refresh_in_background() -> None:
    """Kick a single-flight daemon-thread refresh; callers keep the stale cache.

    The GCS client is synchronous — a refresh on the request path would freeze
    the event loop (and the typing indicator) for the duration.
    """
    global _refreshing

    def _run():
        global _refreshing
        try:
            _refresh_catalog()
        except Exception as e:
            logger.warning("[wiki] background refresh failed: %s", e)
        finally:
            _refreshing = False

    with _refresh_lock:
        if _refreshing:
            return
        _refreshing = True
    threading.Thread(target=_run, name="wiki-refresh", daemon=True).start()


def load_knowledge_catalog(force: bool = False) -> list[dict]:
    """Return the curated-article catalog, serving stale while revalidating.

    A cold cache (first call on a new instance) loads synchronously — cheap now
    that lazy prefixes are name-listed, not downloaded. A stale cache is served
    as-is and refreshed in a background thread.
    """
    if force:
        return _refresh_catalog()
    if _catalog_cache is None:
        return _refresh_catalog()
    if (time.time() - _cache_ts) >= _CACHE_TTL:
        _refresh_in_background()
    return _catalog_cache


def knowledge_index_text() -> str:
    articles = load_knowledge_catalog()
    if not articles and not _lazy_names_cache:
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
    if _lazy_names_cache:
        lines.append(
            f"- Plus {len(_lazy_names_cache)} support troubleshooting playbooks "
            "(not listed individually). Find one with `search_wiki(query)`, then "
            "read it with `read_knowledge(name)`."
        )
    return "\n".join(lines)


def _fetch_lazy_article(name: str) -> dict | None:
    """Fetch a single lazily-served article body on demand (one blob GET)."""
    cached = _lazy_body_cache.get(name)
    if cached is not None:
        return cached
    path = _lazy_names_cache.get(name)
    if not path:
        return None
    try:
        from kb import storage

        text = storage.read_text(path)
    except Exception as e:
        logger.warning("[wiki] lazy fetch of %s failed: %s", path, e)
        return None
    if text is None:
        return None
    parsed = _parse(path, text)
    if parsed:
        _lazy_body_cache[name] = parsed
    return parsed


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
    lazy = _fetch_lazy_article(name)
    if lazy:
        return lazy["body"]
    available = ", ".join(a["name"] for a in load_knowledge_catalog()) or "(none)"
    return f"No knowledge article named '{name}'. Available: {available}"


@tool
def search_wiki(query: str, limit: int = 8) -> str:
    """Search the wiki (repo skills + bucket knowledge) for a keyword or phrase.

    Naive case-insensitive substring search; returns matching items with a short
    snippet. Use it to find relevant skills or knowledge before answering.

    Args:
        query: Keyword or phrase to search for.
        limit: Max number of matches to return (default 8).
    """
    q = (query or "").strip().lower()
    if not q:
        return "Provide a non-empty query."
    hits: list[str] = []
    # Repo skills (local files).
    if SKILLS_DIR.is_dir():
        for md in sorted(SKILLS_DIR.rglob("*.md")):
            try:
                text = md.read_text(encoding="utf-8")
            except OSError:
                continue
            if q in text.lower():
                i = text.lower().index(q)
                hits.append(f"- {md.name}: …{text[max(0,i-120):i+len(q)+120].replace(chr(10),' ').strip()}…")
                if len(hits) >= limit:
                    return "Wiki matches:\n" + "\n".join(hits)
    # Bucket knowledge (already-downloaded bodies — no extra GCS calls).
    for a in load_knowledge_catalog():
        blob = f"{a['title']}\n{a['body']}"
        if q in blob.lower():
            i = blob.lower().index(q)
            hits.append(f"- {a['name']}: …{blob[max(0,i-120):i+len(q)+120].replace(chr(10),' ').strip()}…")
            if len(hits) >= limit:
                break
    # Lazily-served playbooks: match by name (playbooks are named per area/topic),
    # fetching at most a handful of bodies for snippets. Bodies are never bulk-
    # downloaded — that's the whole point of the lazy tier.
    if len(hits) < limit and _lazy_names_cache:
        tokens = [t for t in re.split(r"[^a-z0-9]+", q) if len(t) >= 3]
        matched = [
            n for n in sorted(_lazy_names_cache)
            if q in n.lower() or any(t in n.lower() for t in tokens)
        ]
        for n in matched[: max(0, limit - len(hits))]:
            a = _fetch_lazy_article(n)
            if not a:
                continue
            blob = f"{a['title']}\n{a['body']}"
            i = blob.lower().find(q)
            snippet = (
                blob[max(0, i - 120):i + len(q) + 120] if i >= 0 else a["summary"]
            )
            hits.append(f"- {n}: …{snippet.replace(chr(10), ' ').strip()}…")
    return "Wiki matches:\n" + "\n".join(hits) if hits else f"No wiki matches for '{query}'."


WIKI_TOOLS = [read_knowledge, search_wiki]
