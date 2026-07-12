"""Tool for querying Google Cloud Logging.

Reads are *compressed* before they reach the model: newest-first ordering, a
compact one-line message per entry (not the whole payload blob), known-noise
lines stripped by default, and identical repeats collapsed. See
``skills/reading-cloud-logs/SKILL.md`` for the full read-efficiency guidance.
"""

import json
import os
import re

from langchain_core.tools import tool

# Documented operational noise (CLAUDE.md "Known operational notes" +
# skills/troubleshooting-cloud-run). Stripped by default; pass exclude_regex=""
# to disable, or a custom pattern to override.
DEFAULT_NOISE = (
    r"OutOfOrderError|tasks\.sqlite-journal|langmem_memories\.sqlite"
    r"|Shutdown|SIGTERM|draining|/healthz?\b"
)

_MAX_MSG_CHARS = 300


def _message(payload) -> str:
    """Extract the single most useful message string from an entry payload.

    Order of preference: structured ``message`` -> plain text -> audit-log
    status message -> a short JSON fallback. Never dumps the whole blob.
    """
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("message", "msg", "text"):
            val = payload.get(key)
            if isinstance(val, str) and val:
                return val
        # Audit / proto payloads: status.message + methodName is the signal.
        status = payload.get("status")
        if isinstance(status, dict) and status.get("message"):
            method = payload.get("methodName", "")
            return f"{method}: {status['message']}".strip(": ")
        # Fallback: compact JSON, not a pretty-printed multi-KB dump.
        return json.dumps(payload, default=str)[:_MAX_MSG_CHARS]
    return str(payload)


@tool
def query_cloud_logs(
    filter_query: str,
    project_id: str | None = None,
    max_results: int = 50,
    exclude_regex: str | None = None,
    include_regex: str | None = None,
    collapse_repeats: bool = True,
) -> str:
    """Query Google Cloud Logging entries (newest-first, compressed for reading).

    Returns compact `[timestamp] revision SEVERITY: message` lines rather than
    full payload blobs, to save time and tokens. Narrow server-side first with a
    tight filter (resource.type, service_name, severity, a time bound), then let
    this tool project + strip + collapse the rest.

    Args:
        filter_query: A Cloud Logging filter string, e.g.
            'resource.type="cloud_run_revision"
             resource.labels.service_name="samurai-bot" severity>=ERROR'
        project_id: GCP project ID. Defaults to GCP_PROJECT_ID env var.
        max_results: Max entries to fetch (newest first). Default 50.
        exclude_regex: Drop lines matching this pattern. Defaults to the known
            operational-noise pattern; pass "" to keep everything.
        include_regex: If set, keep only lines matching this pattern.
        collapse_repeats: Fold identical messages into one `(xN)` line. Default on.
    """
    from google.cloud import logging as cloud_logging

    pid = project_id or os.environ["GCP_PROJECT_ID"]
    client = cloud_logging.Client(project=pid)
    entries = list(
        client.list_entries(
            filter_=filter_query,
            order_by=cloud_logging.DESCENDING,
            max_results=max_results,
        )
    )
    if not entries:
        return "No log entries found for that filter."

    noise = DEFAULT_NOISE if exclude_regex is None else exclude_regex
    excl = re.compile(noise) if noise else None
    incl = re.compile(include_regex) if include_regex else None

    lines = []
    dropped = 0
    for entry in entries:
        ts = entry.timestamp.isoformat() if entry.timestamp else "?"
        rev = ""
        labels = getattr(getattr(entry, "resource", None), "labels", None)
        if isinstance(labels, dict):
            rev = labels.get("revision_name", "")
        msg = _message(entry.payload).replace("\n", " ").strip()
        if len(msg) > _MAX_MSG_CHARS:
            msg = msg[:_MAX_MSG_CHARS] + "…"
        line = f"[{ts}] {rev} {entry.severity}: {msg}".replace("  ", " ")
        if excl and excl.search(line):
            dropped += 1
            continue
        if incl and not incl.search(line):
            dropped += 1
            continue
        lines.append(line)

    # Read chronological (oldest -> newest) once filtered.
    lines.reverse()

    if collapse_repeats:
        collapsed = []
        for line in lines:
            if collapsed and collapsed[-1][0] == line:
                collapsed[-1][1] += 1
            else:
                collapsed.append([line, 1])
        lines = [f"{l} (x{n})" if n > 1 else l for l, n in collapsed]

    if not lines:
        return f"All {dropped} entries were filtered out (noise/include-regex)."

    out = "\n".join(lines)
    if dropped:
        out += f"\n… ({dropped} noise/filtered lines omitted)"
    return out
