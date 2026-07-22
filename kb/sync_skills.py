"""In-boundary sync of approved skills from GitHub into the SamurAI skills bucket.

Consume side of the skills feedback loop (plan Part F). Approved skills live in the
private ``virtualdojo-inc/virtualdojo-skills`` repo (source of truth) at
``plugins/virtualdojo-skills/skills/<name>/SKILL.md``. This module pulls them, in
boundary, into ``support/skills/synced/`` — the bucket prefix ``skills.py`` already
serves — so a merge to the catalog appears in SamurAI within the skills-cache TTL,
no redeploy.

Boundary posture:
- **Reads only, inward.** It fetches *already-sanitized* skill files via the GitHub
  Contents API (the bot already egresses to api.github.com; ``samurai-dojo`` has
  ``contents:read``). Nothing from the bucket is sent out here.
- Runs **in-process on the serving instance** (never a GitHub runner), under its own
  single-flight lease lock, gated by ``SKILLS_SYNC_ENABLED``.

Prefix discipline (mirrors the reasoning in skills.py:SKILLS_BUCKET_PREFIX):
- Writes only under ``support/skills/synced/`` — a distinct sub-prefix from the
  hand-authored ``support/skills/<name>.md`` (written by tools/skill_authoring).
- **Prunes only synced-owned files** — a skill removed from the catalog is deleted
  from ``synced/`` but a hand-authored bucket skill is never touched.
- On a name clash, the hand-authored skill wins (see skills._load_bucket_skills).
"""

from __future__ import annotations

import logging
import os

from kb import storage
from skills import SKILLS_BUCKET_PREFIX, _parse_skill_text

logger = logging.getLogger(__name__)

SKILLS_REPO = os.environ.get("SKILLS_REPO", "virtualdojo-inc/virtualdojo-skills")
SKILLS_REPO_REF = os.environ.get("SKILLS_REPO_REF", "main")
# Where approved SKILL.md files live inside the source repo (the plugin layout).
SKILLS_REPO_PATH = os.environ.get(
    "SKILLS_REPO_PATH", "plugins/virtualdojo-skills/skills"
)

# Distinct sub-prefix so sync never collides with hand-authored bucket skills.
SYNCED_PREFIX = SKILLS_BUCKET_PREFIX + "synced/"
SYNC_LOCK_PATH = SKILLS_BUCKET_PREFIX + ".sync.lock"
SYNC_LOCK_TTL = int(os.environ.get("SKILLS_SYNC_LOCK_TTL", "600"))


def sync_enabled() -> bool:
    return os.environ.get("SKILLS_SYNC_ENABLED", "off").lower() not in ("off", "", "0", "false", "no")


def _fetch_catalog_skills() -> list[tuple[str, str, str]]:
    """Read approved ``SKILL.md`` files from the source repo.

    Returns ``(skill_name, raw_text, source_sha)`` tuples. Read-only, inward.
    """
    from tools.github import _github  # lazy: avoids GitHub dep at import/in tests

    repo = _github().get_repo(SKILLS_REPO)
    out: list[tuple[str, str, str]] = []
    try:
        entries = repo.get_contents(SKILLS_REPO_PATH, ref=SKILLS_REPO_REF)
    except Exception as e:
        logger.warning("[skills.sync] cannot list %s/%s@%s: %s",
                       SKILLS_REPO, SKILLS_REPO_PATH, SKILLS_REPO_REF, e)
        return out

    # skills/<name>/SKILL.md — each entry in the skills dir is a skill directory.
    for entry in entries:
        if entry.type != "dir":
            continue
        skill_md = f"{entry.path}/SKILL.md"
        try:
            cf = repo.get_contents(skill_md, ref=SKILLS_REPO_REF)
        except Exception:
            logger.warning("[skills.sync] %s has no SKILL.md; skipping", entry.path)
            continue
        text = cf.decoded_content.decode("utf-8")
        parsed = _parse_skill_text(text, skill_md)
        if parsed is None:
            # malformed frontmatter — never surface it; the loader would skip it anyway
            continue
        out.append((parsed["name"], text, cf.sha))
    return out


def _stamp_provenance(text: str, name: str, source_sha: str) -> str:
    """Insert provenance markers into the skill's frontmatter.

    Additive keys only (``synced``, ``source_repo``, ``source_sha``); skills.py
    ignores unknown frontmatter keys, so the skill still parses. Identifies the file
    as synced-owned (for pruning) without altering the skill's name/description/body.
    """
    parts = text.split("---", 2)
    if len(parts) < 3:
        return text  # shouldn't happen (already parsed), leave as-is
    fm = parts[1].rstrip("\n")
    fm += (
        "\nsynced: true"
        f"\nsource_repo: {SKILLS_REPO}"
        f"\nsource_sha: {source_sha}"
    )
    return f"---{fm}\n---{parts[2]}"


def _synced_object_name(name: str) -> str:
    return f"{SYNCED_PREFIX}{name}.md"


def run_skill_sync(force: bool = False) -> dict:
    """Pull approved skills into ``support/skills/synced/``. Returns content-free stats.

    ``force=True`` bypasses the SKILLS_SYNC_ENABLED kill switch (deliberate human
    trigger) but still respects the single-flight lock.
    """
    if not force and not sync_enabled():
        logger.info("[skills.sync] SKILLS_SYNC_ENABLED is off — skipping.")
        return {"skipped": True}

    if not storage.acquire_lock(SYNC_LOCK_PATH, ttl_seconds=SYNC_LOCK_TTL):
        logger.info("[skills.sync] another sync holds the lock — skipping.")
        return {"skipped": True, "reason": "locked"}

    written = 0
    pruned = 0
    try:
        catalog = _fetch_catalog_skills()
        wanted: dict[str, str] = {}  # synced object name -> provenance-stamped text
        for name, text, sha in catalog:
            wanted[_synced_object_name(name)] = _stamp_provenance(text, name, sha)

        # Write/update. Skip unchanged content to avoid needless bucket churn.
        for obj_name, content in wanted.items():
            existing = storage.read_text(obj_name)
            if existing == content:
                continue
            storage.write_text(obj_name, content)
            written += 1

        # Prune only synced-owned files no longer in the catalog. Never touch the
        # hand-authored top-level ``support/skills/*.md``.
        for obj_name in storage.list_paths(SYNCED_PREFIX):
            if not obj_name.endswith(".md"):
                continue
            if obj_name not in wanted:
                storage.delete(obj_name)
                pruned += 1

        logger.info(
            "[skills.sync] synced=%d written=%d pruned=%d from %s@%s",
            len(catalog), written, pruned, SKILLS_REPO, SKILLS_REPO_REF,
        )
        return {"synced": len(catalog), "written": written, "pruned": pruned}
    finally:
        storage.release_lock(SYNC_LOCK_PATH)
