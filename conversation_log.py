"""Durable per-turn conversation capture — the ``raw/`` ingest for the wiki.

Each completed agent turn is written as one JSON file under
``DATA_DIR/raw/<YYYY-MM-DD>/``. **One file per turn** (never appended to a shared
file) so concurrent Cloud Run instances don't race — appends over the GCS FUSE
mount raise ``OutOfOrderError``.

These raw transcripts contain PII/business content, so they live ONLY on the
private ``samurai-bot-data`` bucket and are **never committed to git**. The
nightly ``wiki-compile`` job reads them and distills (redacted) knowledge into
the committed ``skills/`` + ``knowledge/`` wiki.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("SAMURAI_DATA_DIR", "/data")
RAW_DIR = Path(DATA_DIR) / "raw"


def log_turn(
    *,
    conversation_id: str,
    user_id: str,
    user_message: str,
    assistant_response: str,
    user_name: str = "",
    user_email: str = "",
    tools: list[str] | None = None,
    is_background_task: bool = False,
    ts: datetime | None = None,
) -> str | None:
    """Write one conversation turn to the raw log.

    Returns the path written, or ``None`` on any failure — capture must never
    break a turn, so all errors are swallowed and logged.
    """
    try:
        ts = ts or datetime.now(timezone.utc)
        day_dir = RAW_DIR / ts.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": ts.isoformat(),
            "conversation_id": conversation_id,
            "user_id": user_id,
            "user_name": user_name,
            "user_email": user_email,
            "is_background_task": is_background_task,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "tools": tools or [],
        }
        # One file per turn — no shared-file appends (GCS FUSE OutOfOrderError).
        path = day_dir / f"{ts.strftime('%H%M%S')}-{uuid.uuid4().hex[:8]}.json"
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[conversation_log] failed to log turn: %s", e)
        return None
