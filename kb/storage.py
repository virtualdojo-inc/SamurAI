"""In-boundary GCS access for the VirtualDojo knowledge base.

All reads/writes target ``gs://virtualdojo-knowledge`` inside the SamurAI Assured
Workloads boundary (FedRAMP Moderate). Uses the google-cloud-storage client
directly (already a dependency) — no gcsfuse mount required, no external egress.

This module is data-agnostic plumbing: it moves bytes between the bot (running
in-boundary on Cloud Run) and the bucket. It never sends content anywhere else.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time

from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage

logger = logging.getLogger(__name__)

KB_BUCKET = os.environ.get("KB_BUCKET", "virtualdojo-knowledge")

_client: storage.Client | None = None


def _bucket():
    global _client
    if _client is None:
        _client = storage.Client()
    return _client.bucket(KB_BUCKET)


def read_text(path: str) -> str | None:
    """Read a text object at ``path`` (relative to the bucket root). None if absent."""
    blob = _bucket().blob(path)
    if not blob.exists():
        return None
    return blob.download_as_text()


def write_text(path: str, content: str, content_type: str = "text/markdown") -> None:
    """Write a text object at ``path``."""
    _bucket().blob(path).upload_from_string(content, content_type=content_type)


def exists(path: str) -> bool:
    return _bucket().blob(path).exists()


def delete(path: str) -> None:
    """Delete the object at ``path``. Raises if it does not exist."""
    _bucket().blob(path).delete()


def list_paths(prefix: str) -> list[str]:
    """List object paths under ``prefix`` (full object names, not content)."""
    return [b.name for b in _bucket().list_blobs(prefix=prefix)]


def list_text(prefix: str, suffix: str = ".md") -> list[tuple[str, str]]:
    """Yield ``(path, text)`` for every object under ``prefix`` ending in ``suffix``.

    Used by the in-boundary compile to read raw sources. Returns content, so this
    must only ever be called from in-boundary compute (never a laptop/runner).
    """
    out: list[tuple[str, str]] = []
    for blob in _bucket().list_blobs(prefix=prefix):
        if blob.name.endswith(suffix):
            out.append((blob.name, blob.download_as_text()))
    return out


# ── Single-flight lease lock (cross-instance) ──────────────────────────────
# A lock is a small object created atomically (if_generation_match=0). Holders
# refresh the timestamp; a stale lock (older than TTL) can be taken over. This
# stops multiple Cloud Run instances from running the compile at once during
# revision churn.

def acquire_lock(path: str, ttl_seconds: int = 1800) -> bool:
    """Try to acquire the lease lock at ``path``. Returns True if held by us."""
    blob = _bucket().blob(path)
    payload = json.dumps({"acquired_at": time.time(), "host": socket.gethostname()})
    try:
        blob.upload_from_string(payload, content_type="application/json", if_generation_match=0)
        return True
    except PreconditionFailed:
        # Lock object exists — take over only if it's stale.
        try:
            blob.reload()
            data = json.loads(blob.download_as_text() or "{}")
            if time.time() - float(data.get("acquired_at", 0)) > ttl_seconds:
                gen = blob.generation
                blob.upload_from_string(
                    payload, content_type="application/json", if_generation_match=gen
                )
                logger.warning("[kb.lock] took over stale lock %s", path)
                return True
        except (PreconditionFailed, ValueError, Exception) as e:  # pragma: no cover
            logger.warning("[kb.lock] could not evaluate/steal lock %s: %s", path, e)
        return False


def release_lock(path: str) -> None:
    """Release the lock at ``path`` (best-effort)."""
    try:
        _bucket().blob(path).delete()
    except Exception as e:  # pragma: no cover - best effort
        logger.warning("[kb.lock] release failed for %s: %s", path, e)
