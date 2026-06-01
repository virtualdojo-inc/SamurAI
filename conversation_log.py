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
        # One file per turn — no shared-file appends (GCS FUSE OutOfOrderError).
        filename = f"{ts.strftime('%H%M%S')}-{uuid.uuid4().hex[:8]}.json"
        # Stable, self-describing handle (relative to RAW_DIR) so a later Teams
        # feedback submit can correlate back to this exact turn record.
        turn_id = f"{ts.strftime('%Y-%m-%d')}/{filename}"
        record = {
            "ts": ts.isoformat(),
            "turn_id": turn_id,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "user_name": user_name,
            "user_email": user_email,
            "is_background_task": is_background_task,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "tools": tools or [],
        }
        path = day_dir / filename
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[conversation_log] failed to log turn: %s", e)
        return None


def find_latest_turn_id(conversation_id: str, days: int = 2) -> str | None:
    """Return the ``turn_id`` of the most recent logged turn for a conversation.

    Used to correlate a Teams feedback click to the turn it's about: ``reply_to_id``
    on the inbound invoke is unreliable (msteams-docs #11870), and feedback is
    given on the latest bot reply, so we resolve by conversation + recency.
    Scans the last ``days`` date partitions. Best-effort; ``None`` if not found.
    """
    try:
        if not RAW_DIR.is_dir():
            return None
        day_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()], reverse=True)[:days]
        best: tuple[str, str] | None = None  # (ts, turn_id)
        for d in day_dirs:
            for f in d.glob("*.json"):
                try:
                    rec = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if rec.get("conversation_id") != conversation_id:
                    continue
                tid = rec.get("turn_id") or f"{d.name}/{f.name}"
                ts = rec.get("ts", "")
                if best is None or ts > best[0]:
                    best = (ts, tid)
        return best[1] if best else None
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[conversation_log] find_latest_turn_id failed: %s", e)
        return None


def record_feedback(
    *,
    conversation_id: str,
    turn_id: str = "",
    reaction: str = "",
    category: str = "",
    text: str = "",
    ts: datetime | None = None,
) -> str | None:
    """Attach a human 👍/👎 feedback record onto its turn's raw log file.

    The self-tuning eval reads this ``feedback`` field as an INDEPENDENT,
    human-verified signal (vs. grading the model against its own past choices).
    Falls back to the conversation's latest turn if ``turn_id`` is missing/stale.
    Best-effort; returns the path written or ``None``.
    """
    try:
        tid = turn_id or find_latest_turn_id(conversation_id) or ""
        if not tid:
            logger.warning("[conversation_log] no turn to attach feedback (conv=%s)", conversation_id)
            return None
        path = RAW_DIR / tid
        if not path.is_file():
            logger.warning("[conversation_log] feedback turn not found: %s", tid)
            return None
        rec = json.loads(path.read_text(encoding="utf-8"))
        rec["feedback"] = {
            "reaction": reaction,
            "category": category,
            "text": text,
            "ts": (ts or datetime.now(timezone.utc)).isoformat(),
        }
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[conversation_log] record_feedback failed: %s", e)
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
