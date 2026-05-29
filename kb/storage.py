"""In-boundary GCS access for the VirtualDojo knowledge base.

All reads/writes target ``gs://virtualdojo-knowledge`` inside the SamurAI Assured
Workloads boundary (FedRAMP Moderate). Uses the google-cloud-storage client
directly (already a dependency) — no gcsfuse mount required, no external egress.

This module is data-agnostic plumbing: it moves bytes between the bot (running
in-boundary on Cloud Run) and the bucket. It never sends content anywhere else.
"""

from __future__ import annotations

import os

from google.cloud import storage

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
