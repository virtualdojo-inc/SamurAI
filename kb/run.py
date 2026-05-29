"""Daily in-boundary knowledge-base pipeline entrypoint (support pilot).

ingest (GitHub issues + Smartsheet) → compile (Gemini) → index. Runs IN-BOUNDARY
on samurai-bot. Gated by ``KB_PIPELINE_ENABLED`` (kill switch). Emits ONLY
content-free summary stats (counts, paths, engine provenance) so logs stay clean
of protected content and can be used for compliance verification.
"""

from __future__ import annotations

import json
import logging
import os

from kb import compile as kb_compile
from kb import ingest_github, ingest_smartsheet
from kb.gemini import kb_engine_info

logger = logging.getLogger(__name__)


def pipeline_enabled() -> bool:
    return os.environ.get("KB_PIPELINE_ENABLED", "off").lower() != "off"


def run_support_pipeline(force: bool = False) -> dict:
    """Run the full support KB pipeline once. Returns content-free stats.

    ``force=True`` (a deliberate human-triggered run) bypasses the
    KB_PIPELINE_ENABLED daily kill switch.
    """
    if not force and not pipeline_enabled():
        print("[kb.run] KB_PIPELINE_ENABLED is off — skipping.", flush=True)
        return {"skipped": True}

    summary: dict = {"engine": kb_engine_info()}
    try:
        summary["github"] = ingest_github.refresh_github_issues()
    except Exception as e:  # don't let one source abort the rest
        summary["github"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        summary["smartsheet"] = ingest_smartsheet.ingest_smartsheet()
    except Exception as e:
        summary["smartsheet"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        summary["compile"] = kb_compile.compile_support()
    except Exception as e:
        summary["compile"] = {"error": f"{type(e).__name__}: {e}"}

    blob = json.dumps(summary, default=str)
    logger.info("[kb.run] support pipeline complete: %s", blob)
    print(f"[kb.run] support pipeline complete: {blob}", flush=True)
    return summary
