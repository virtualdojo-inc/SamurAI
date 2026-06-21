"""Agent tools for authoring editable skills to the in-boundary bucket.

A skill authored here lands at ``support/skills/<name>.md`` and is picked up by
the read-only skills loader (``skills.py``, TTL-cached, gated by
``SKILLS_BUCKET_ENABLED``). Writing is a real mutation, so these tools are
defended three ways:

  1. **Approver-gated** — only Devin/Cyrus, like the social tools.
  2. **Judge-gated** — registered in ``judge.WRITE_TOOL_NAMES`` so the
     fail-closed write judge inspects every call.
  3. **Validated + read-back** — content must parse as a valid SKILL.md before
     write, and the write is confirmed by reading it back.

The loader itself stays read-only; this is the only write path into
``support/skills/``. The chosen prefix is outside the compile's sources and the
wiki's served subdirs, so an authored skill is never harvested or served as
knowledge (enforced by the guard test in tests/test_skills.py).
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from skills import (
    _NAME_RE,
    _RESERVED_WORDS,
    SKILLS_BUCKET_PREFIX,
    _bucket_skills_enabled,
    _parse_skill_text,
    load_skill_catalog,
)

logger = logging.getLogger(__name__)

AUTHORIZED_SKILL_AUTHORS = {
    "cyrus@virtualdojo.com",
    "devin@virtualdojo.com",
}


def _check_author(user_email: str) -> str | None:
    if (user_email or "").lower() not in AUTHORIZED_SKILL_AUTHORS:
        return (
            "You are not authorized to edit skills. Only Devin or Cyrus can "
            "create, edit, or delete skills."
        )
    return None


def _valid_name(name: str) -> bool:
    """Name must match the loader's rule (also keeps the object path safe — no
    slashes or dots, so it can't escape the SKILLS_BUCKET_PREFIX)."""
    return bool(_NAME_RE.match(name)) and not any(w in name for w in _RESERVED_WORDS)


def _skill_path(name: str) -> str:
    return f"{SKILLS_BUCKET_PREFIX}{name}.md"


def _compose_skill_md(name: str, description: str, body: str) -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{body.strip()}\n"


def _enabled_note() -> str:
    if _bucket_skills_enabled():
        return ""
    return (
        " NOTE: bucket skills are currently disabled (SKILLS_BUCKET_ENABLED is "
        "off), so this skill is stored but will not be active until that flag is "
        "set on the service."
    )


@tool
def save_skill(name: str, description: str, body: str, user_email: str) -> str:
    """Create or edit a skill, stored in the in-boundary bucket (Devin/Cyrus only).

    Writes ``support/skills/<name>.md``. If a skill with this name already exists
    (in the bucket or the repo) this becomes the live version — a bucket skill
    overrides a repo one of the same name. The change takes effect within a few
    minutes (no redeploy).

    Args:
        name: Skill name — lowercase letters, numbers, and hyphens only
            (e.g. 'tech-issue-triage'). This is also the filename.
        description: One-line description shown in the skill catalog (<=1024 chars).
            Make it specific about WHEN to use the skill.
        body: The full markdown procedural guidance (the skill's instructions).
        user_email: The requesting user's email (from the context brackets).
            Must be an authorized skill author.
    """
    auth_err = _check_author(user_email)
    if auth_err:
        return auth_err

    name = (name or "").strip()
    if not _valid_name(name):
        return (
            f"Invalid skill name {name!r}. Use only lowercase letters, numbers, "
            "and hyphens (max 64 chars), and avoid reserved words."
        )

    md = _compose_skill_md(name, description.strip(), body)
    # Validate it round-trips through the loader's parser BEFORE writing.
    parsed = _parse_skill_text(md, _skill_path(name))
    if parsed is None or parsed["name"] != name:
        return (
            "The skill content is invalid (bad frontmatter, empty/oversized "
            "description, or name mismatch). Nothing was written."
        )

    path = _skill_path(name)
    try:
        from kb import storage

        storage.write_text(path, md)
        # Read-back verify (the project's write-then-confirm rule).
        back = storage.read_text(path)
    except Exception as e:
        logger.error("[skill_authoring] save failed for %s: %s", path, e)
        return f"Failed to write skill '{name}' to the bucket: {e}"

    if back is None or _parse_skill_text(back, path) is None:
        return f"Wrote '{name}' but read-back verification failed. Please retry."

    # Refresh the live catalog so the new/edited skill is immediately available.
    try:
        load_skill_catalog(force=True)
    except Exception as e:  # non-fatal — the next TTL refresh will pick it up
        logger.warning("[skill_authoring] catalog reload after save failed: %s", e)

    return (
        f"Saved skill '{name}' to {path} and verified the write. "
        f"It is now the live version (bucket overrides repo on name clash)." + _enabled_note()
    )


@tool
def delete_skill(name: str, user_email: str) -> str:
    """Delete a bucket-authored skill (Devin/Cyrus only).

    Removes ``support/skills/<name>.md``. This can only delete skills authored to
    the bucket — repo skills (committed to git) are unaffected and would need a
    code change to remove.

    Args:
        name: The skill name to delete.
        user_email: The requesting user's email (from the context brackets).
            Must be an authorized skill author.
    """
    auth_err = _check_author(user_email)
    if auth_err:
        return auth_err

    name = (name or "").strip()
    if not _valid_name(name):
        return f"Invalid skill name {name!r}."

    path = _skill_path(name)
    try:
        from kb import storage

        if not storage.exists(path):
            return (
                f"No bucket skill named '{name}' at {path}. (If it's a repo skill, "
                "it can't be deleted here — that needs a code change.)"
            )
        storage.delete(path)
    except Exception as e:
        logger.error("[skill_authoring] delete failed for %s: %s", path, e)
        return f"Failed to delete skill '{name}': {e}"

    try:
        load_skill_catalog(force=True)
    except Exception as e:
        logger.warning("[skill_authoring] catalog reload after delete failed: %s", e)

    return f"Deleted bucket skill '{name}' ({path})."


SKILL_AUTHORING_TOOLS = [save_skill, delete_skill]
