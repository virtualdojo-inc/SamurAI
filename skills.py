"""Agent Skills for SamurAI.

A *skill* is a directory under ``skills/`` containing a ``SKILL.md`` file with
YAML frontmatter (``name`` + ``description``) and a markdown body of procedural
knowledge. This follows Anthropic's Agent Skills model with progressive
disclosure:

  Level 1 (always in context): each skill's ``name`` + ``description``, injected
           into the system prompt via :func:`skills_catalog_text`. Cheap — just
           enough for the agent to know a skill exists and when it is relevant.
  Level 2 (on demand): the full ``SKILL.md`` body, fetched by the ``get_skill``
           tool only when the agent decides the skill is relevant.

This mirrors the keyword-gated ``PROMPT_SECTIONS`` / ``TOOL_GROUPS`` pattern in
``agent.py``, but skills are self-service: the agent pulls a skill's body when it
judges the task relevant, rather than relying on keyword matching.

Skills are plain files, so the nightly self-improvement pipeline can tune them
(edit ``SKILL.md``), add new ones, or refine descriptions via ordinary PRs.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path

import yaml
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).parent / "skills"

# Anthropic skill-name rules: <=64 chars, lowercase letters/numbers/hyphens,
# and must not contain the reserved words below.
_NAME_RE = re.compile(r"^[a-z0-9-]{1,64}$")
_RESERVED_WORDS = ("anthropic", "claude")
_MAX_DESCRIPTION = 1024

# Editable skills live in the in-boundary bucket at this prefix. Chosen
# deliberately: it sits UNDER the `support/` scope (so the runtime SA's existing
# conditioned read+write grant covers it — verified against live IAM) but is
# OUTSIDE the compile's source prefix (`support/raw/`) and the wiki's served
# subdirs (`support/{wiki,playbooks,troubleshooting}`), so a bucket skill is
# never harvested into a playbook nor served as a knowledge article. The guard
# test in tests/test_skills.py enforces that separation.
SKILLS_BUCKET_PREFIX = "support/skills/"
# Off by default (kill switch, mirrors the KB_*_ENABLED pattern): keeps the bucket
# read dormant until explicitly enabled, and keeps tests/local hermetic.
_BUCKET_ENV = "SKILLS_BUCKET_ENABLED"
_CACHE_TTL = 300  # seconds — pick up bucket edits without a redeploy (mirrors wiki.py)

_catalog_cache: list[dict] | None = None
_cache_ts: float = 0.0


def _parse_skill_text(text: str, source: str) -> dict | None:
    """Parse SKILL.md *text* into ``{name, description, body, dir}`` or None.

    ``source`` is a label for logging (a file path or a bucket object name).
    Returns ``None`` (and logs a warning) for any malformed skill so one bad
    file can never crash agent startup.
    """
    # Frontmatter is a leading '---' delimited block.
    if not text.lstrip().startswith("---"):
        logger.warning("[skills] %s missing frontmatter; skipping", source)
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        logger.warning("[skills] %s malformed frontmatter; skipping", source)
        return None

    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        logger.warning("[skills] %s bad YAML frontmatter (%s); skipping", source, e)
        return None
    if not isinstance(meta, dict):
        logger.warning("[skills] %s frontmatter is not a mapping; skipping", source)
        return None

    # `key:` with no value parses to None in YAML — coerce to "" so the
    # validation below rejects it instead of seeing the string "None".
    name = str(meta.get("name") or "").strip()
    description = str(meta.get("description") or "").strip()
    body = parts[2].strip()

    if not _NAME_RE.match(name) or any(w in name for w in _RESERVED_WORDS):
        logger.warning("[skills] %s has invalid name %r; skipping", source, name)
        return None
    if not description or len(description) > _MAX_DESCRIPTION:
        logger.warning("[skills] %s has invalid/empty description; skipping", source)
        return None

    return {"name": name, "description": description, "body": body, "dir": source}


def _parse_skill_md(path: Path) -> dict | None:
    """Parse a repo SKILL.md file into ``{name, description, body, dir}``."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("[skills] could not read %s (%s); skipping", path, e)
        return None
    parsed = _parse_skill_text(text, str(path))
    if parsed is not None:
        parsed["dir"] = path.parent.name
    return parsed


def _load_repo_skills() -> list[dict]:
    """Skills committed to the repo under ``skills/<name>/SKILL.md``."""
    skills: list[dict] = []
    if SKILLS_DIR.is_dir():
        for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
            parsed = _parse_skill_md(skill_md)
            if parsed is not None:
                skills.append(parsed)
    return skills


def _bucket_skills_enabled() -> bool:
    return os.environ.get(_BUCKET_ENV, "").strip().lower() in ("1", "true", "on", "yes")


def _load_bucket_skills() -> list[dict]:
    """Editable skills authored to the in-boundary bucket (``support/skills/``).

    Read via the google-cloud-storage client (same in-boundary path as wiki.py).
    Dormant unless ``SKILLS_BUCKET_ENABLED`` is set. Guarded: any failure (no
    bucket, no creds, in tests) yields an empty list so the repo skills still
    load — the bucket is additive, never required.
    """
    if not _bucket_skills_enabled():
        return []
    try:
        from kb import storage  # lazy: avoids hard dep at import / in tests

        items = storage.list_text(SKILLS_BUCKET_PREFIX)
    except Exception as e:
        logger.warning("[skills] bucket read failed (%s); using repo skills only", e)
        return []

    out: list[dict] = []
    for path_name, text in items:
        base = path_name.rsplit("/", 1)[-1].lower()
        if base in ("index.md", ".keep") or not base.endswith(".md"):
            continue
        parsed = _parse_skill_text(text, path_name)
        if parsed is not None:
            out.append(parsed)
    # Precedence on a name clash: a hand-authored top-level skill
    # (``support/skills/<name>.md``, written by save_skill) beats a synced
    # catalog skill (``support/skills/synced/<name>.md``, written by
    # kb/sync_skills). _load_catalog applies bucket skills last-wins, so we emit
    # synced entries FIRST and top-level LAST — the local human override wins.
    out.sort(key=lambda s: 0 if "/synced/" in str(s.get("dir", "")) else 1)
    return out


_refresh_lock = threading.Lock()
_refreshing = False


def _refresh_in_background() -> None:
    """Single-flight daemon-thread refresh; callers keep serving the stale cache.

    The bucket list is a synchronous GCS call — refreshing on the request path
    (load_skill_catalog is called during prompt assembly, on the event loop)
    would block the loop for the duration.
    """
    global _refreshing

    def _run():
        global _refreshing
        try:
            _load_catalog()
        except Exception as e:
            logger.warning("[skills] background refresh failed: %s", e)
        finally:
            _refreshing = False

    with _refresh_lock:
        if _refreshing:
            return
        _refreshing = True
    threading.Thread(target=_run, name="skills-refresh", daemon=True).start()


def load_skill_catalog(force: bool = False) -> list[dict]:
    """Load + cache skills: repo skills, with bucket skills overriding by name.

    A bucket skill wins on a name clash so a chat-time edit (authored to
    ``support/skills/``) takes effect without a redeploy. TTL-cached; a stale
    cache is served as-is and refreshed in a background thread.
    """
    if not force and _catalog_cache is not None:
        if (time.time() - _cache_ts) >= _CACHE_TTL:
            _refresh_in_background()
        return _catalog_cache
    return _load_catalog()


def _load_catalog() -> list[dict]:
    """Synchronous full load — bucket I/O when enabled; call off the event loop."""
    global _catalog_cache, _cache_ts

    # Repo first (dedup keep-first among repo files), then bucket overrides by name.
    by_name: dict[str, dict] = {}
    order: list[str] = []
    for s in _load_repo_skills():
        if s["name"] in by_name:
            logger.warning("[skills] duplicate skill name %r; keeping first", s["name"])
            continue
        order.append(s["name"])
        by_name[s["name"]] = s
    for s in _load_bucket_skills():
        if s["name"] not in by_name:
            order.append(s["name"])
        else:
            logger.info("[skills] bucket overrides repo skill %r", s["name"])
        by_name[s["name"]] = s

    deduped = [by_name[n] for n in order]
    _catalog_cache = deduped
    _cache_ts = time.time()
    logger.info(
        "[skills] loaded %d skills: %s", len(deduped), [s["name"] for s in deduped]
    )
    return deduped


def skills_catalog_text() -> str:
    """Compact name+description list for the system prompt (level-1 disclosure)."""
    skills = load_skill_catalog()
    if not skills:
        return ""
    lines = [
        "## Available skills",
        (
            "You have procedural skills available. When a task matches a skill's "
            "description, call `get_skill(name)` to load its full instructions "
            "BEFORE acting. Do not guess a skill's contents from its name."
        ),
    ]
    for s in skills:
        lines.append(f"- **{s['name']}** — {s['description']}")
    return "\n".join(lines)


@tool
def get_skill(name: str) -> str:
    """Load the full instructions for a named skill.

    Call this when the current task matches one of the skills listed under
    'Available skills' in your system prompt. Returns the skill's complete
    procedural guidance.

    Args:
        name: The skill name, e.g. 'troubleshooting-cloud-run'.
    """
    catalog = load_skill_catalog()
    for s in catalog:
        if s["name"] == name:
            try:  # names+counts-only usage telemetry; never break the tool
                import skill_usage
                skill_usage.record(name)
            except Exception:
                pass
            return s["body"]
    available = ", ".join(s["name"] for s in catalog) or "(none)"
    return f"No skill named '{name}'. Available skills: {available}"


SKILL_TOOLS = [get_skill]
