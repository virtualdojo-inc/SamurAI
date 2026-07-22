"""In-boundary skill capture for SamurAI (plan Part C).

Distills reusable *skills* from SamurAI's own work, entirely inside the FedRAMP
boundary, and delivers them as sanitized drafts for human review.

Corpus: the per-turn conversation log at ``/data/raw/<date>/*.json`` (the private
``samurai-bot-data`` mount) — full transcript + tool trace + human 👍/👎 feedback. This
is the rich record of SamurAI's DevOps/GitHub/CRM work (NOT the narrow, off-by-default
``support/conversation-history/``). Unit of capture = a per-``conversation_id`` rollup
over a recent window (SamurAI has no "session").

Boundary posture (this engine CAN see customer org data via support grants, so it is the
strict one):
  * Distillation + the LLM sanitization self-check run on **regional Vertex Gemini**
    (``kb.gemini.get_kb_llm`` — refuses ``global``). Never an external LLM.
  * Every candidate passes the **deterministic gate first** (``sanitizer`` FULL mode:
    PII + secrets + tenant-name denylist), which hard-blocks regardless of the LLM.
  * Only the finished, sanitized skill crosses out — delivered by filing a labeled
    ``skill-draft`` issue (the bot has ``issues:write``; a workflow in virtualdojo-skills
    turns it into a draft PR). No repo write access needed.

Resumable + bounded like ``kb/compile.py``: a manifest (conversation_id → rollup hash)
checkpoints after each conversation; a single-flight lease lock guards it; gated by
``SKILLS_DISTILL_ENABLED``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sanitizer
from kb import storage
from kb.compile import _json_from, _llm_text
from kb.gemini import get_kb_llm, kb_engine_info

logger = logging.getLogger(__name__)

DRAFTS_PREFIX = "support/skills-drafts/"
MANIFEST_PATH = DRAFTS_PREFIX + ".state/manifest.json"
AUDIT_PREFIX = DRAFTS_PREFIX + ".audit/"
LOCK_PATH = DRAFTS_PREFIX + ".distill.lock"
LOCK_TTL = int(os.environ.get("SKILLS_DISTILL_LOCK_TTL", "1800"))
TENANT_DENYLIST_PATH = DRAFTS_PREFIX + ".tenant-denylist.txt"

MAX_CONVERSATIONS = int(os.environ.get("SKILLS_DISTILL_MAX", "20"))
WINDOW_DAYS = int(os.environ.get("SKILLS_DISTILL_WINDOW_DAYS", "2"))

SKILLS_REPO = os.environ.get("SKILLS_REPO", "virtualdojo-inc/virtualdojo-skills")
JUDGE_PROMPT_VERSION = "distill-v1"

_NAME_RE = re.compile(r"^[a-z0-9-]{1,64}$")
_RESERVED = ("anthropic", "claude")


def distill_enabled() -> bool:
    return os.environ.get("SKILLS_DISTILL_ENABLED", "off").lower() not in ("off", "", "0", "false", "no")


# --- LLM prompts (in-boundary Gemini) ---------------------------------------------

_DISTILL_SYS = (
    "You extract a single REUSABLE skill from one agent conversation, or decide there "
    "is nothing worth capturing. A skill is a general, reusable PROCEDURE (a technique, "
    "recipe, or gotcha) — never a one-off answer, never a record of this specific "
    "incident. Return STRICT JSON only:\n"
    '{"worth_capturing": bool, "name": "kebab-case-technique-name", '
    '"description": "when to use it (<=1024 chars)", "body": "markdown steps", '
    '"reason": "why/why not"}.\n'
    "Rules: name after the PATTERN/technique (e.g. n-plus-one-batch-flush-fix), NEVER "
    "after a customer/tenant/contract/person. name matches ^[a-z0-9-]{1,64}$, no words "
    "'anthropic'/'claude'. The body is pattern-only: NO customer names, NO PII, NO "
    "secrets, NO verbatim log lines / paths / hostnames / IDs specific to a customer. "
    "If the conversation is trivial (greeting, no tools, no lesson) or too specific to "
    "generalize, set worth_capturing=false."
)

_SANITIZE_SYS = (
    "You are a strict sanitization judge for a FedRAMP boundary. Given a proposed skill "
    "(name, description, body) that will be published to a shared GitHub repo, decide if "
    "it contains ANY private content: customer/tenant names, PII, secrets/credentials, or "
    "verbatim data (log lines, paths, hostnames, IDs) specific to a customer. Return STRICT "
    'JSON only: {"clean": bool, "reason": "short"}. When in doubt, clean=false.'
)


def _load_tenant_names() -> list[str]:
    """Tenant/customer-name denylist for the deterministic gate (primary control).

    Sourced in-boundary from a bucket file (one name per line, #-comments ok) and/or the
    ``SKILLS_TENANT_DENYLIST`` env (comma-separated). Best-effort — an empty list still
    leaves PII+secret detection active. (Wiring to the live CRM tenant list is a follow-up.)
    """
    names: list[str] = []
    env = os.environ.get("SKILLS_TENANT_DENYLIST", "")
    names += [n.strip() for n in env.split(",") if n.strip()]
    try:
        raw = storage.read_text(TENANT_DENYLIST_PATH) or ""
        for line in raw.splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                names.append(line)
    except Exception:  # pragma: no cover - best effort
        pass
    return names


# --- Corpus: read + group the per-turn conversation log ---------------------------

def _raw_dir() -> Path:
    from conversation_log import RAW_DIR
    return RAW_DIR


def _load_recent_conversations() -> dict[str, list[dict]]:
    """Group recent turn records by conversation_id (within the window). In-boundary."""
    raw_dir = _raw_dir()
    convos: dict[str, list[dict]] = {}
    if not raw_dir.is_dir():
        return convos
    cutoff = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    for day_dir in sorted(raw_dir.iterdir()):
        if not day_dir.is_dir() or day_dir.name < cutoff:
            continue
        for f in sorted(day_dir.glob("*.json")):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            cid = rec.get("conversation_id") or f.stem
            convos.setdefault(cid, []).append(rec)
    return convos


def _rollup_text(turns: list[dict]) -> tuple[str, bool, bool]:
    """Build the distill input for one conversation. Returns (text, used_tools, positive_fb)."""
    turns = sorted(turns, key=lambda r: r.get("ts", ""))
    used_tools = False
    positive_fb = False
    parts: list[str] = []
    for t in turns:
        tools = t.get("tools") or []
        if tools:
            used_tools = True
        fb = (t.get("feedback") or {}).get("reaction", "")
        if fb in ("👍", "thumbs_up", "up", "like"):
            positive_fb = True
        parts.append(
            f"USER: {t.get('user_message','')}\n"
            f"TOOLS: {tools}\n"
            f"ASSISTANT: {t.get('assistant_response','')}"
        )
    return "\n\n---\n\n".join(parts), used_tools, positive_fb


# --- Dedup ------------------------------------------------------------------------

def _norm_tokens(s: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def _is_duplicate(name: str, description: str, existing: list[tuple[str, str]]) -> bool:
    """Lightweight dedup: exact name match, or high description-token Jaccard (>=0.6)."""
    desc_tokens = _norm_tokens(description)
    for ex_name, ex_desc in existing:
        if ex_name == name:
            return True
        ex_tokens = _norm_tokens(ex_desc)
        if desc_tokens and ex_tokens:
            j = len(desc_tokens & ex_tokens) / len(desc_tokens | ex_tokens)
            if j >= 0.6:
                return True
    return False


def _existing_skill_names_descriptions() -> list[tuple[str, str]]:
    """Catalog (bucket) + open skill-draft PRs/issues, for cross-engine dedup."""
    out: list[tuple[str, str]] = []
    # bucket catalog (served skills, incl. synced/)
    try:
        for path, text in storage.list_text("support/skills/"):
            from skills import _parse_skill_text
            parsed = _parse_skill_text(text, path)
            if parsed:
                out.append((parsed["name"], parsed["description"]))
    except Exception:  # pragma: no cover
        pass
    # already-staged drafts
    try:
        for path, text in storage.list_text(DRAFTS_PREFIX):
            if "/.state/" in path or "/.audit/" in path:
                continue
            from skills import _parse_skill_text
            parsed = _parse_skill_text(text, path)
            if parsed:
                out.append((parsed["name"], parsed["description"]))
    except Exception:  # pragma: no cover
        pass
    # open skill-draft issues/PRs in the catalog repo (cross-engine)
    try:
        from tools.github import _github
        for issue in _github().get_repo(SKILLS_REPO).get_issues(state="open", labels=["skill-draft"]):
            title = issue.title or ""
            name = title.split("skill-draft:", 1)[-1].strip() if "skill-draft:" in title else ""
            out.append((name, title))
    except Exception:  # pragma: no cover
        pass
    return out


# --- Delivery: file a labeled skill-draft issue (bridge turns it into a PR) --------

def _compose_skill_md(name: str, description: str, body: str) -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{body.strip()}\n"


def _file_draft_issue(name: str, skill_md: str, reason: str) -> str | None:
    """File a labeled skill-draft issue carrying the sanitized skill (the crossing).

    Uses the bridge contract in virtualdojo-skills (skill-name marker + begin/end).
    Only sanitized content reaches here.
    """
    body = (
        f"<!-- skill-name: {name} -->\n"
        "Auto-captured by SamurAI (in-boundary distill on Vertex Gemini). "
        "Sanitization gate (deterministic + LLM) passed before filing.\n\n"
        f"_Why:_ {reason}\n\n"
        "<!-- skill-begin -->\n"
        f"{skill_md}"
        "<!-- skill-end -->\n"
    )
    try:
        from tools.github import _github
        repo = _github().get_repo(SKILLS_REPO)
        issue = repo.create_issue(
            title=f"skill-draft: {name}", body=body, labels=["skill-draft"]
        )
        return issue.html_url
    except Exception as e:
        logger.warning("[skills.distill] could not file draft issue for %s: %s", name, e)
        return None


def _append_audit(record: dict) -> None:
    day = record.get("ts", "")[:10] or "unknown"
    path = f"{AUDIT_PREFIX}{day}.jsonl"
    prev = storage.read_text(path) or ""
    storage.write_text(path, prev + json.dumps(record) + "\n", content_type="application/json")


# --- Main -------------------------------------------------------------------------

def run_skill_distill(force: bool = False) -> dict:
    """One bounded, resumable distill tick. Returns content-free stats."""
    if not force and not distill_enabled():
        logger.info("[skills.distill] SKILLS_DISTILL_ENABLED is off — skipping.")
        return {"skipped": True}
    if not storage.acquire_lock(LOCK_PATH, ttl_seconds=LOCK_TTL):
        logger.info("[skills.distill] another distill holds the lock — skipping.")
        return {"skipped": True, "reason": "locked"}

    llm = get_kb_llm()
    tenant_names = _load_tenant_names()
    manifest: dict = {}
    try:
        raw = storage.read_text(MANIFEST_PATH)
        manifest = json.loads(raw) if raw else {}
    except Exception:  # pragma: no cover
        manifest = {}

    stats = {"engine": kb_engine_info(), "conversations": 0, "distilled": 0,
             "det_blocked": 0, "llm_blocked": 0, "duplicate": 0, "not_worth": 0, "filed": 0}
    filed_this_run: list[tuple[str, str]] = []  # (name, description) — dedup within a run

    try:
        convos = _load_recent_conversations()
        pending = []
        for cid, turns in convos.items():
            text, used_tools, positive_fb = _rollup_text(turns)
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if manifest.get(cid) == h:
                continue  # unchanged since last processed
            # Triviality filter: skip conversations that used no tools and are short,
            # unless the human thumbed them up.
            if not used_tools and len(text) < 400 and not positive_fb:
                manifest[cid] = h
                continue
            pending.append((cid, text, h, positive_fb))

        # Positive-feedback conversations first (strongest lesson signal).
        pending.sort(key=lambda x: (not x[3],))
        batch = pending[:MAX_CONVERSATIONS]
        existing = _existing_skill_names_descriptions()

        for cid, text, h, _pos in batch:
            stats["conversations"] += 1
            data = _json_from(_llm_text(llm, _DISTILL_SYS, text)) or {}
            manifest[cid] = h  # mark processed regardless of outcome (idempotent)

            if not data.get("worth_capturing"):
                stats["not_worth"] += 1
                _save_manifest(manifest)
                continue
            name = str(data.get("name") or "").strip()
            description = str(data.get("description") or "").strip()
            body = str(data.get("body") or "").strip()
            reason = str(data.get("reason") or "").strip()
            if not _NAME_RE.match(name) or any(w in name.lower() for w in _RESERVED) \
                    or not description or len(description) > 1024 or not body:
                stats["not_worth"] += 1
                _save_manifest(manifest)
                continue

            artifacts = {"name": name, "description": description, "body": body}

            # 1) Deterministic gate (hard, first).
            gate = sanitizer.scan_artifacts(artifacts, mode=sanitizer.Mode.FULL,
                                            tenant_names=tenant_names)
            # 2) LLM sanitization self-check (in-boundary), only if deterministic passed.
            llm_clean = None
            llm_reason = None
            if not gate.blocked:
                verdict = _json_from(_llm_text(
                    llm, _SANITIZE_SYS,
                    json.dumps(artifacts))) or {}
                llm_clean = bool(verdict.get("clean"))
                llm_reason = str(verdict.get("reason") or "")

            _append_audit(sanitizer.build_audit_record(
                artifacts, gate, llm_verdict=("pass" if llm_clean else "fail") if llm_clean is not None else None,
                llm_reason=llm_reason, judge_prompt_version=JUDGE_PROMPT_VERSION))

            if gate.blocked:
                stats["det_blocked"] += 1
                _save_manifest(manifest)
                continue
            if not llm_clean:
                stats["llm_blocked"] += 1
                _save_manifest(manifest)
                continue

            # 3) Dedup (catalog + open drafts + within this run).
            if _is_duplicate(name, description, existing + filed_this_run):
                stats["duplicate"] += 1
                _save_manifest(manifest)
                continue

            # 4) Stage the sanitized draft in-boundary + deliver via labeled issue.
            skill_md = _compose_skill_md(name, description, body)
            storage.write_text(f"{DRAFTS_PREFIX}{name}.md", skill_md)
            stats["distilled"] += 1
            url = _file_draft_issue(name, skill_md, reason)
            if url:
                stats["filed"] += 1
            filed_this_run.append((name, description))
            _save_manifest(manifest)

        logger.info("[skills.distill] %s", {k: v for k, v in stats.items() if k != "engine"})
        return stats
    finally:
        _save_manifest(manifest)
        storage.release_lock(LOCK_PATH)


def _save_manifest(manifest: dict) -> None:
    storage.write_text(MANIFEST_PATH, json.dumps(manifest), content_type="application/json")
