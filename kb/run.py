"""In-boundary knowledge-base pipeline entrypoint (support pilot).

ingest (GitHub issues + Smartsheet) → compile (Gemini, bounded+resumable) →
index. Runs IN-BOUNDARY on samurai-bot. Gated by ``KB_PIPELINE_ENABLED`` (kill
switch). Single-flight: a cross-instance lease lock prevents overlapping runs
during revision churn. Each tick compiles at most ``KB_COMPILE_MAX_DOCS`` docs
and converges over repeated ticks. Emits ONLY content-free summary stats.
"""

from __future__ import annotations

import json
import logging
import os

from kb import compile as kb_compile
from kb import compile_engineering as kb_compile_eng
from kb import ingest_github, ingest_repo, ingest_smartsheet, storage
from kb.gemini import kb_engine_info

logger = logging.getLogger(__name__)

LOCK_PATH = "support/playbooks/.compile.lock"
LOCK_TTL = int(os.environ.get("KB_LOCK_TTL", "1800"))  # stale-takeover after 30m
DEFAULT_MAX_DOCS = int(os.environ.get("KB_COMPILE_MAX_DOCS", "50"))

ENG_LOCK_PATH = "engineering/.pipeline.lock"
ENG_MAX_DOCS = int(os.environ.get("KB_ENG_COMPILE_MAX_DOCS", "4"))


def pipeline_enabled() -> bool:
    return os.environ.get("KB_PIPELINE_ENABLED", "off").lower() != "off"


def engineering_pipeline_enabled() -> bool:
    return os.environ.get("KB_ENG_PIPELINE_ENABLED", "off").lower() != "off"


def run_support_pipeline(force: bool = False) -> dict:
    """Run one bounded, single-flight KB tick. Returns content-free stats.

    ``force=True`` (a deliberate human trigger) bypasses the KB_PIPELINE_ENABLED
    kill switch but still respects the single-flight lock.
    """
    if not force and not pipeline_enabled():
        print("[kb.run] KB_PIPELINE_ENABLED is off — skipping.", flush=True)
        return {"skipped": True}

    # #3 single-flight: skip if another instance/tick holds the lock.
    if not storage.acquire_lock(LOCK_PATH, ttl_seconds=LOCK_TTL):
        print("[kb.run] another compile is in progress (lock held) — skipping.", flush=True)
        return {"skipped": "locked"}

    summary: dict = {"engine": kb_engine_info()}
    try:
        try:
            summary["github"] = ingest_github.refresh_github_issues()
        except Exception as e:  # don't let one source abort the rest
            summary["github"] = {"error": f"{type(e).__name__}: {e}"}
        try:
            summary["smartsheet"] = ingest_smartsheet.ingest_smartsheet()
        except Exception as e:
            summary["smartsheet"] = {"error": f"{type(e).__name__}: {e}"}
        try:
            summary["compile"] = kb_compile.compile_support(max_docs=DEFAULT_MAX_DOCS)
        except Exception as e:
            summary["compile"] = {"error": f"{type(e).__name__}: {e}"}
    finally:
        storage.release_lock(LOCK_PATH)

    blob = json.dumps(summary, default=str)
    logger.info("[kb.run] support pipeline complete: %s", blob)
    print(f"[kb.run] support pipeline complete: {blob}", flush=True)
    return summary


def run_engineering_pipeline(force: bool = False) -> dict:
    """Run one bounded, single-flight ENGINEERING KB tick. Returns content-free stats.

    ingest the virtualdojo repo (allowlisted docs + structure → article stubs) →
    compile stubs into engineering/wiki + engineering/troubleshooting articles.
    Gated on ``main`` HEAD sha: if nothing merged since the last sync (and not
    ``force``), the ingest reports ``no-merges`` and we skip the Gemini compile —
    a cheap nightly no-op. ``force=True`` (a deliberate human trigger) bypasses
    both the ``KB_ENG_PIPELINE_ENABLED`` kill switch and the no-merges gate, but
    still respects the single-flight lock.
    """
    if not force and not engineering_pipeline_enabled():
        print("[kb.run] KB_ENG_PIPELINE_ENABLED is off — skipping.", flush=True)
        return {"skipped": True}

    if not storage.acquire_lock(ENG_LOCK_PATH, ttl_seconds=LOCK_TTL):
        print("[kb.run] another engineering compile is in progress — skipping.", flush=True)
        return {"skipped": "locked"}

    summary: dict = {"engine": kb_engine_info()}
    try:
        try:
            ingest = ingest_repo.refresh_repo_knowledge(force=force)
            summary["ingest"] = ingest
        except Exception as e:  # don't let ingest failure mask the lock release
            summary["ingest"] = {"error": f"{type(e).__name__}: {e}"}
            ingest = {"error": True}

        # Cheap nightly no-op: nothing merged to main → no stub changes → skip compile.
        if not force and ingest.get("skipped") == "no-merges":
            summary["compile"] = {"skipped": "no-merges"}
        else:
            try:
                summary["compile"] = kb_compile_eng.compile_engineering(max_docs=ENG_MAX_DOCS)
            except Exception as e:
                summary["compile"] = {"error": f"{type(e).__name__}: {e}"}
    finally:
        storage.release_lock(ENG_LOCK_PATH)

    blob = json.dumps(summary, default=str)
    logger.info("[kb.run] engineering pipeline complete: %s", blob)
    print(f"[kb.run] engineering pipeline complete: {blob}", flush=True)
    return summary
