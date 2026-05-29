"""Ingest Smartsheet rows into the knowledge base.

- Support-ticket sheets  → ``support/raw/smartsheet/`` (support scope)
- Onboarding sheets      → ``customers/onboarding/raw/smartsheet/`` (different scope)

Sheet IDs are operator-configured (env), so nothing is discovered/egressed from a
laptop — this runs IN-BOUNDARY on samurai-bot and reuses ``tools/smartsheet.py``
(token from the ``smartsheet-api-token`` secret). One markdown file per row with
provenance frontmatter; secrets scrubbed on the way in. No LLM involved.

Env config (comma-separated, quote IDs to avoid precision loss):
  KB_SMARTSHEET_SUPPORT_SHEET_IDS="1146352141553540,..."
  KB_SMARTSHEET_ONBOARDING_SHEET_IDS="...."
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from kb import storage
from kb.ingest_github import _scrub  # reuse the secret scrubber
from tools.smartsheet import _get

SUPPORT_PREFIX = "support/raw/smartsheet/"
ONBOARDING_PREFIX = "customers/onboarding/raw/smartsheet/"


def _sheet_ids(env_var: str) -> list[str]:
    raw = os.environ.get(env_var, "")
    return [s.strip() for s in raw.split(",") if s.strip()]


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
    """Ingest configured support + onboarding sheets. Returns content-free stats."""
    support = [_ingest_sheet(sid, SUPPORT_PREFIX) for sid in _sheet_ids("KB_SMARTSHEET_SUPPORT_SHEET_IDS")]
    onboarding = [_ingest_sheet(sid, ONBOARDING_PREFIX) for sid in _sheet_ids("KB_SMARTSHEET_ONBOARDING_SHEET_IDS")]
    return {
        "source": "smartsheet",
        "support_sheets": support,
        "onboarding_sheets": onboarding,
        "support_rows": sum(s["rows_written"] for s in support),
        "onboarding_rows": sum(s["rows_written"] for s in onboarding),
    }
