"""Periodic skill-catalog evaluation (plan Part G).

On a cadence, joins the approved catalog with usage counts and flags:
  * dead      — in the catalog but never triggered → propose retire
  * valuable  — high usage → protect/refine
  * duplicate — near-identical descriptions → propose merge

It **proposes, never disposes**: it files ONE ``skill-eval`` report issue in
virtualdojo-skills (the bot has ``issues:write``) for a human to act on. Retiring or
merging a skill is a human-reviewed PR, not something this does automatically.

In-boundary + read-only against GitHub (``contents:read``): it reads the catalog
(``plugins/virtualdojo-skills/skills/*/SKILL.md``) and the rollup leaderboard
(``telemetry/leaderboard.md``) via the GitHub API. Gated by ``SKILLS_EVAL_ENABLED``;
own single-flight lease lock.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

from kb import storage

logger = logging.getLogger(__name__)

SKILLS_REPO = os.environ.get("SKILLS_REPO", "virtualdojo-inc/virtualdojo-skills")
SKILLS_REPO_REF = os.environ.get("SKILLS_REPO_REF", "main")
SKILLS_REPO_PATH = os.environ.get("SKILLS_REPO_PATH", "plugins/virtualdojo-skills/skills")
LEADERBOARD_PATH = "telemetry/leaderboard.md"

EVAL_LOCK_PATH = "support/skills/.eval.lock"
EVAL_LOCK_TTL = int(os.environ.get("SKILLS_EVAL_LOCK_TTL", "600"))
VALUABLE_THRESHOLD = int(os.environ.get("SKILLS_EVAL_VALUABLE_MIN", "10"))
DUP_JACCARD = float(os.environ.get("SKILLS_EVAL_DUP_JACCARD", "0.6"))


def eval_enabled() -> bool:
    return os.environ.get("SKILLS_EVAL_ENABLED", "off").lower() not in ("off", "", "0", "false", "no")


def _fetch_catalog() -> list[tuple[str, str]]:
    """(name, description) for every approved skill. Read-only, inward."""
    from tools.github import _github
    from skills import _parse_skill_text

    repo = _github().get_repo(SKILLS_REPO)
    out: list[tuple[str, str]] = []
    try:
        entries = repo.get_contents(SKILLS_REPO_PATH, ref=SKILLS_REPO_REF)
    except Exception as e:
        logger.warning("[skills.eval] cannot list catalog: %s", e)
        return out
    for entry in entries:
        if entry.type != "dir":
            continue
        try:
            cf = repo.get_contents(f"{entry.path}/SKILL.md", ref=SKILLS_REPO_REF)
            parsed = _parse_skill_text(cf.decoded_content.decode("utf-8"), entry.path)
        except Exception:
            continue
        if parsed:
            out.append((parsed["name"], parsed["description"]))
    return out


def _fetch_usage() -> dict[str, int]:
    """Skill -> total invocations, parsed from the rollup leaderboard. {} if absent."""
    from tools.github import _github

    try:
        repo = _github().get_repo(SKILLS_REPO)
        cf = repo.get_contents(LEADERBOARD_PATH, ref=SKILLS_REPO_REF)
        text = cf.decoded_content.decode("utf-8")
    except Exception:
        return {}
    counts: dict[str, int] = {}
    # rows look like: | 1 | `skill-name` | 5 |
    for m in re.finditer(r"\|\s*\d+\s*\|\s*`([a-z0-9-]+)`\s*\|\s*(\d+)\s*\|", text):
        counts[m.group(1)] = int(m.group(2))
    return counts


def _tokens(s: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def _find_duplicates(catalog: list[tuple[str, str]]) -> list[tuple[str, str, float]]:
    dups = []
    for i in range(len(catalog)):
        for j in range(i + 1, len(catalog)):
            a, b = _tokens(catalog[i][1]), _tokens(catalog[j][1])
            if a and b:
                jac = len(a & b) / len(a | b)
                if jac >= DUP_JACCARD:
                    dups.append((catalog[i][0], catalog[j][0], round(jac, 2)))
    return dups


def evaluate(catalog: list[tuple[str, str]], usage: dict[str, int]) -> dict:
    names = [n for n, _ in catalog]
    dead = sorted(n for n in names if usage.get(n, 0) == 0)
    valuable = sorted((n for n in names if usage.get(n, 0) >= VALUABLE_THRESHOLD),
                      key=lambda n: -usage.get(n, 0))
    duplicates = _find_duplicates(catalog)
    return {
        "total": len(names),
        "dead": dead,
        "valuable": [(n, usage[n]) for n in valuable],
        "duplicates": duplicates,
        "usage": usage,
    }


def _report_md(result: dict, now: datetime) -> str:
    lines = [
        f"# Skill catalog evaluation — {now.strftime('%Y-%m-%d')}",
        "",
        f"{result['total']} skills in the catalog. Recommendations below are "
        "**proposals** — a maintainer decides. Retiring/merging a skill is a PR.",
        "",
        "## 🪦 Dead (never triggered) — consider retiring",
    ]
    lines += [f"- `{n}`" for n in result["dead"]] or ["- (none)"]
    lines += ["", f"## ⭐ Valuable (≥ {VALUABLE_THRESHOLD} invocations) — protect/refine"]
    lines += [f"- `{n}` — {c} invocations" for n, c in result["valuable"]] or ["- (none)"]
    lines += ["", "## 👯 Likely duplicates (description overlap) — consider merging"]
    lines += [f"- `{a}` ↔ `{b}` (similarity {j})" for a, b, j in result["duplicates"]] or ["- (none)"]
    lines += ["", "_Generated in-boundary from the catalog + telemetry/leaderboard.md._"]
    return "\n".join(lines) + "\n"


def run_skill_evaluation(force: bool = False) -> dict:
    """One evaluation pass → a ``skill-eval`` report issue. Returns content-free stats."""
    if not force and not eval_enabled():
        logger.info("[skills.eval] SKILLS_EVAL_ENABLED is off — skipping.")
        return {"skipped": True}
    if not storage.acquire_lock(EVAL_LOCK_PATH, ttl_seconds=EVAL_LOCK_TTL):
        logger.info("[skills.eval] another eval holds the lock — skipping.")
        return {"skipped": True, "reason": "locked"}
    try:
        catalog = _fetch_catalog()
        if not catalog:
            logger.info("[skills.eval] empty catalog — nothing to evaluate.")
            return {"total": 0}
        usage = _fetch_usage()
        result = evaluate(catalog, usage)
        now = datetime.now(timezone.utc)
        body = _report_md(result, now)

        stats = {"total": result["total"], "dead": len(result["dead"]),
                 "valuable": len(result["valuable"]), "duplicates": len(result["duplicates"])}
        try:
            from tools.github import _github
            _github().get_repo(SKILLS_REPO).create_issue(
                title=f"skill-eval: catalog review {now.strftime('%Y-%m-%d')}",
                body=body, labels=["skill-eval"])
            stats["reported"] = True
        except Exception as e:
            logger.warning("[skills.eval] could not file report issue: %s", e)
            stats["reported"] = False
        logger.info("[skills.eval] %s", stats)
        return stats
    finally:
        storage.release_lock(EVAL_LOCK_PATH)
