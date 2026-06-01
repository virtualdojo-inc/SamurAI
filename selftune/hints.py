"""The single mutable prompt layer: ``learned_hints.md``.

The self-tuning split is dead simple:
  - **Immutable core** = everything in ``agent.py`` (identity, mission, AUTONOMY
    RULES, facts-only, tool *schemas*). The loop can NEVER touch it.
  - **Mutable layer** = this one doc, ``/data/selftune/learned_hints.md`` (on the
    in-boundary samurai-bot-data bucket). It holds learned operational hints +
    tool "when-to-call" guidance. The propose→evaluate→promote loop edits ONLY
    this file; the runtime injects it AFTER the core, so it refines but never
    overrides the core rules.

Loaded with a TTL cache (like the wiki) so promotions appear without a redeploy.
A hard size cap stops a runaway loop from bloating the prompt.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("SAMURAI_DATA_DIR", "/data")
HINTS_PATH = Path(DATA_DIR) / "selftune" / "learned_hints.md"
HISTORY_DIR = Path(DATA_DIR) / "selftune" / "history"
_CACHE_TTL = 300  # seconds — promotions picked up within 5 min, no redeploy
_MAX_HINTS_CHARS = 6000  # hard cap (~1.5k tokens); the loop is rewarded for staying small
_MAX_HISTORY = 20

_cache: str | None = None
_cache_ts: float = 0.0


def load_hints(force: bool = False) -> str:
    """Return the raw learned-hints doc (TTL-cached). '' if absent/unreadable."""
    global _cache, _cache_ts
    if _cache is not None and not force and (time.time() - _cache_ts) < _CACHE_TTL:
        return _cache
    text = ""
    try:
        if HINTS_PATH.is_file():
            text = HINTS_PATH.read_text(encoding="utf-8")
    except OSError as e:  # pragma: no cover - defensive
        logger.warning("[selftune.hints] read failed: %s", e)
    _cache, _cache_ts = text, time.time()
    return text


def learned_hints_text() -> str:
    """The injectable prompt block (header + capped body), or '' if no hints yet."""
    body = load_hints().strip()
    if not body:
        return ""
    if len(body) > _MAX_HINTS_CHARS:
        logger.warning("[selftune.hints] hints exceed cap (%d) — truncating", _MAX_HINTS_CHARS)
        body = body[:_MAX_HINTS_CHARS]
    return (
        "## Learned operational guidance (auto-tuned)\n"
        "Refinements learned from past interactions — tool routing and usage "
        "tips. These REFINE the rules above; if anything here conflicts with the "
        "core rules or autonomy rules, the core rules win.\n\n" + body
    )


def save_hints(text: str) -> None:
    """Write a new learned-hints doc, backing up the previous version for rollback."""
    HINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if HINTS_PATH.is_file():
        try:
            HISTORY_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            (HISTORY_DIR / f"learned_hints.{stamp}.md").write_text(
                HINTS_PATH.read_text(encoding="utf-8"), encoding="utf-8"
            )
            _prune_history()
        except OSError as e:  # pragma: no cover - best effort
            logger.warning("[selftune.hints] history backup failed: %s", e)
    HINTS_PATH.write_text(text, encoding="utf-8")
    global _cache, _cache_ts
    _cache, _cache_ts = None, 0.0  # invalidate so the next turn reloads


def _prune_history() -> None:
    backups = sorted(HISTORY_DIR.glob("learned_hints.*.md"))
    for old in backups[:-_MAX_HISTORY]:
        try:
            old.unlink()
        except OSError:
            pass


def rollback_hints() -> bool:
    """Restore the most recent history backup (one-step rollback). True if restored."""
    if not HISTORY_DIR.is_dir():
        return False
    backups = sorted(HISTORY_DIR.glob("learned_hints.*.md"))
    if not backups:
        return False
    latest = backups[-1]
    HINTS_PATH.write_text(latest.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        latest.unlink()
    except OSError:
        pass
    global _cache, _cache_ts
    _cache, _cache_ts = None, 0.0
    return True
