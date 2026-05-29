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


# Support-scope chat capture into the in-boundary knowledge bucket. This is a
# LOG, not a source of truth: the compile may read it for continuity but must
# never cite it as authoritative (README echo-chamber guard). Env-gated so it
# adds no latency/bucket dependency unless explicitly enabled.
SUPPORT_CHAT_CAPTURE = os.environ.get("KB_SUPPORT_CHAT_CAPTURE", "off").lower() != "off"
_SUPPORT_HISTORY_PREFIX = "support/conversation-history/"


def log_support_chat(
    *,
    conversation_id: str,
    user_id: str,
    user_message: str,
    assistant_response: str,
    user_name: str = "",
    tools: list[str] | None = None,
    ts: datetime | None = None,
) -> str | None:
    """Append a support-chat turn to gs://virtualdojo-knowledge/support/conversation-history/.

    Best-effort and env-gated (``KB_SUPPORT_CHAT_CAPTURE``). Runs in-boundary on
    samurai-bot; writes via the in-boundary GCS client. Never raises.
    """
    if not SUPPORT_CHAT_CAPTURE:
        return None
    try:
        from kb import storage  # lazy: avoids a hard dep at module import

        ts = ts or datetime.now(timezone.utc)
        body = (
            "---\n"
            "type: support-conversation-log\n"
            "authoritative: false  # LOG ONLY — never cite as a source\n"
            f"ts: {ts.isoformat()}\n"
            f"conversation_id: {conversation_id}\n"
            f"user_id: {user_id}\n"
            f"user_name: {user_name}\n"
            f"tools: {tools or []}\n"
            "---\n\n"
            f"**User:** {user_message}\n\n"
            f"**SamurAI:** {assistant_response}\n"
        )
        path = (
            f"{_SUPPORT_HISTORY_PREFIX}{ts.strftime('%Y-%m-%d')}/"
            f"{ts.strftime('%H%M%S')}-{uuid.uuid4().hex[:8]}.md"
        )
        storage.write_text(path, body)
        return path
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[conversation_log] failed to log support chat: %s", e)
        return None
