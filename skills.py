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
import re
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

_catalog_cache: list[dict] | None = None


def _parse_skill_md(path: Path) -> dict | None:
    """Parse a SKILL.md into ``{name, description, body, dir}``.

    Returns ``None`` (and logs a warning) for any malformed skill so one bad
    file can never crash agent startup.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("[skills] could not read %s (%s); skipping", path, e)
        return None

    # Frontmatter is a leading '---' delimited block.
    if not text.lstrip().startswith("---"):
        logger.warning("[skills] %s missing frontmatter; skipping", path)
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        logger.warning("[skills] %s malformed frontmatter; skipping", path)
        return None

    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        logger.warning("[skills] %s bad YAML frontmatter (%s); skipping", path, e)
        return None
    if not isinstance(meta, dict):
        logger.warning("[skills] %s frontmatter is not a mapping; skipping", path)
        return None

    # `key:` with no value parses to None in YAML — coerce to "" so the
    # validation below rejects it instead of seeing the string "None".
    name = str(meta.get("name") or "").strip()
    description = str(meta.get("description") or "").strip()
    body = parts[2].strip()

    if not _NAME_RE.match(name) or any(w in name for w in _RESERVED_WORDS):
        logger.warning("[skills] %s has invalid name %r; skipping", path, name)
        return None
    if not description or len(description) > _MAX_DESCRIPTION:
        logger.warning("[skills] %s has invalid/empty description; skipping", path)
        return None

    return {"name": name, "description": description, "body": body, "dir": path.parent.name}


def load_skill_catalog(force: bool = False) -> list[dict]:
    """Load and cache all valid skills from :data:`SKILLS_DIR`."""
    global _catalog_cache
    if _catalog_cache is not None and not force:
        return _catalog_cache

    skills: list[dict] = []
    if SKILLS_DIR.is_dir():
        for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
            parsed = _parse_skill_md(skill_md)
            if parsed is not None:
                skills.append(parsed)

    # Guard against two skills declaring the same name.
    seen: set[str] = set()
    deduped: list[dict] = []
    for s in skills:
        if s["name"] in seen:
            logger.warning("[skills] duplicate skill name %r; keeping first", s["name"])
            continue
        seen.add(s["name"])
        deduped.append(s)

    _catalog_cache = deduped
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
            return s["body"]
    available = ", ".join(s["name"] for s in catalog) or "(none)"
    return f"No skill named '{name}'. Available skills: {available}"


SKILL_TOOLS = [get_skill]
