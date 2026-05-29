"""Ingest Smartsheet rows into the knowledge base (in-boundary, self-discovering).

Uses SamurAI's existing Smartsheet tooling (`tools/smartsheet.py`, token from the
`smartsheet-api-token` secret) to DISCOVER sheets in-boundary and route them:

- Support-ticket sheets  → ``support/raw/smartsheet/``
- Onboarding sheets      → ``customers/onboarding/raw/smartsheet/``

Discovery + read run on samurai-bot (inside the boundary) — never from a laptop —
so ticket content never leaves the boundary. One markdown file per row with
provenance frontmatter; secrets scrubbed on the way in. No LLM involved.

Routing:
- A known support sheet id (DH Tech Issue Tracker `1146352141553540`, plus any in
  KB_SMARTSHEET_SUPPORT_SHEET_IDS) → support.
- A name containing an onboarding keyword (or any id in
  KB_SMARTSHEET_ONBOARDING_SHEET_IDS) → onboarding.
- A name containing a support keyword (issue tracker / ticket / support / bug) →
  support.
- Anything else is skipped and logged (never misrouted).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from kb import storage
from kb.ingest_github import _scrub  # reuse the secret scrubber
from tools.smartsheet import _get

logger = logging.getLogger(__name__)

SUPPORT_PREFIX = "support/raw/smartsheet/"
ONBOARDING_PREFIX = "customers/onboarding/raw/smartsheet/"

# Known support sheet: DH Tech Issue Tracker (provided by the team).
_DEFAULT_SUPPORT_IDS = {"1146352141553540"}
_ONBOARDING_KEYWORDS = ("onboard",)
_SUPPORT_KEYWORDS = ("issue tracker", "ticket", "support", "bug")


def _ids_from_env(var: str) -> set[str]:
    return {s.strip() for s in os.environ.get(var, "").split(",") if s.strip()}


def _classify(sheet_id: str, name: str) -> str | None:
    name_l = (name or "").lower()
    support_ids = _DEFAULT_SUPPORT_IDS | _ids_from_env("KB_SMARTSHEET_SUPPORT_SHEET_IDS")
    onboarding_ids = _ids_from_env("KB_SMARTSHEET_ONBOARDING_SHEET_IDS")
    if sheet_id in onboarding_ids or any(k in name_l for k in _ONBOARDING_KEYWORDS):
        return "onboarding"
    if sheet_id in support_ids or any(k in name_l for k in _SUPPORT_KEYWORDS):
        return "support"
    return None


def _row_md(sheet: dict, row: dict, cols_by_id: dict) -> tuple[str, int]:
    lines = [
        "---",
        "source: smartsheet",
        f"sheet_id: {sheet.get('id')}",
        f"sheet_name: {sheet.get('name')!r}",
        f"row_id: {row.get('id')}",
        f"row_number: {row.get('rowNumber')}",
        f"ingested_at: {datetime.now(timezone.utc).isoformat()}",
        "---",
        "",
        f"# {sheet.get('name')} — row {row.get('rowNumber')}",
        "",
    ]
    redacted = 0
    for cell in row.get("cells", []):
        col = cols_by_id.get(cell.get("columnId"))
        val = cell.get("displayValue") or cell.get("value")
        if col is None or val in (None, ""):
            continue
        clean, n = _scrub(str(val))
        redacted += n
        lines.append(f"- **{col}**: {clean}")
    return "\n".join(lines), redacted


def _ingest_sheet(sheet_id: str, prefix: str) -> dict:
    sheet = _get(f"/sheets/{sheet_id}")
    cols_by_id = {c["id"]: c["title"] for c in sheet.get("columns", [])}
    rows = sheet.get("rows", [])
    redacted_total = 0
    for row in rows:
        md, redacted = _row_md(sheet, row, cols_by_id)
        redacted_total += redacted
        storage.write_text(f"{prefix}sheet-{sheet_id}-row-{row['id']}.md", md)
    return {"sheet_id": sheet_id, "rows_written": len(rows), "secrets_redacted": redacted_total}


def ingest_smartsheet() -> dict:
    """Discover + ingest support/onboarding sheets in-boundary. Content-free stats."""
    listing = _get("/sheets") or {}
    sheets = listing.get("data", [])
    support, onboarding, skipped = [], [], []
    for s in sheets:
        sid = str(s.get("id"))
        scope = _classify(sid, s.get("name", ""))
        if scope == "support":
            support.append(_ingest_sheet(sid, SUPPORT_PREFIX))
        elif scope == "onboarding":
            onboarding.append(_ingest_sheet(sid, ONBOARDING_PREFIX))
        else:
            skipped.append(sid)
    logger.info(
        "[kb.smartsheet] support=%d onboarding=%d skipped=%d",
        len(support), len(onboarding), len(skipped),
    )
    return {
        "source": "smartsheet",
        "sheets_seen": len(sheets),
        "support_sheets": support,
        "onboarding_sheets": onboarding,
        "skipped_sheet_ids": skipped,
        "support_rows": sum(s["rows_written"] for s in support),
        "onboarding_rows": sum(s["rows_written"] for s in onboarding),
    }
