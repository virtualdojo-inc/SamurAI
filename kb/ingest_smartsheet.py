"""Ingest Smartsheet rows into the knowledge base (in-boundary, self-discovering).

Uses SamurAI's existing Smartsheet tooling (`tools/smartsheet.py`, token from the
`smartsheet-api-token` secret) to DISCOVER sheets in-boundary and route them:

- Support-ticket sheets  → ``support/raw/smartsheet/``
- Onboarding sheets      → ``customers/onboarding/raw/smartsheet/``

Discovery + read run on samurai-bot (inside the boundary) — never from a laptop —
so ticket content never leaves the boundary. One markdown file per row with
provenance frontmatter; secrets scrubbed on the way in. Row ingest uses no LLM.

Optionally (gated by KB_LOOM_INGEST_ENABLED), rows linking a Loom video get an
in-boundary Vertex analysis (kb.ingest_loom) written as an `authoritative: false`
companion note — a derived LOG, deduped so a given Loom is analyzed once.

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
import re
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

# Loom analysis on ticket rows (gated). Row ingest itself uses NO LLM; when
# KB_LOOM_INGEST_ENABLED is on, rows linking a Loom get an in-boundary Vertex
# analysis (kb.ingest_loom) written as an `authoritative: false` companion note —
# a derived LOG, never asserted as product fact. Deduped: skip if already written.
_LOOM_URL_RE = re.compile(r"https?://(?:www\.)?loom\.com/share/[0-9a-fA-F]{16,}")


def _loom_ingest_enabled() -> bool:
    return os.environ.get("KB_LOOM_INGEST_ENABLED", "").lower() in {"on", "1", "true", "yes"}


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


def _extract_loom_urls(row: dict) -> list[str]:
    """Loom share URLs found in a row's cell text or hyperlink targets."""
    urls: set[str] = set()
    for cell in row.get("cells", []):
        for v in (cell.get("displayValue"), cell.get("value")):
            if isinstance(v, str):
                urls.update(_LOOM_URL_RE.findall(v))
        link = (cell.get("hyperlink") or {}).get("url")
        if isinstance(link, str):
            m = _LOOM_URL_RE.search(link)
            if m:
                urls.add(m.group(0))
    return sorted(urls)


def _loom_note_md(sheet_id: str, row: dict, url: str, loom_id: str, analysis) -> str:
    """Companion note for a row's Loom — marked authoritative:false (a derived log)."""
    eng = analysis.engine if isinstance(analysis.engine, dict) else {}
    understanding, _ = _scrub(analysis.understanding or analysis.visual_summary or "(no analysis)")
    lines = [
        "---",
        "source: loom",
        "authoritative: false",
        f"sheet_id: {sheet_id}",
        f"row_id: {row.get('id')}",
        f"row_number: {row.get('rowNumber')}",
        f"loom_id: {loom_id}",
        f"loom_url: {url}",
        f"narration_source: {analysis.narration_source}",
        f"engine: {eng.get('engine')}/{eng.get('model')}@{eng.get('location')}",
        f"ingested_at: {datetime.now(timezone.utc).isoformat()}",
        "---",
        "",
        f"# Loom analysis — {analysis.title or loom_id} (sheet {sheet_id}, row {row.get('rowNumber')})",
        "",
        f"Source video (a recorded log, not product documentation): {url}",
        f"Duration: {analysis.duration:.0f}s | Narration: {analysis.narration_source}",
        "",
        "## What the video shows",
        understanding,
    ]
    if analysis.narration:
        narration, _ = _scrub(analysis.narration)
        lines += ["", "## Narration", narration]
    return "\n".join(lines)


def _ingest_row_looms(sheet_id: str, row: dict, prefix: str) -> int:
    """Analyze + persist any new Looms linked on a row. Returns count analyzed.
    Best-effort: a failed analysis never breaks row ingest. Deduped via storage.exists."""
    from kb.ingest_loom import ingest_loom, loom_id_from_url

    count = 0
    for url in _extract_loom_urls(row):
        loom_id = loom_id_from_url(url)
        if not loom_id:
            continue
        path = f"{prefix}sheet-{sheet_id}-row-{row['id']}-loom-{loom_id}.md"
        if storage.exists(path):  # already analyzed — skip the expensive re-run
            continue
        try:
            analysis = ingest_loom(url)
        except Exception as e:  # noqa: BLE001 - never let one Loom break ingest
            logger.warning("[kb.smartsheet] loom analysis failed for %s: %s", url, e)
            continue
        storage.write_text(path, _loom_note_md(sheet_id, row, url, loom_id, analysis))
        count += 1
    return count


def _ingest_sheet(sheet_id: str, prefix: str) -> dict:
    sheet = _get(f"/sheets/{sheet_id}")
    cols_by_id = {c["id"]: c["title"] for c in sheet.get("columns", [])}
    rows = sheet.get("rows", [])
    redacted_total = 0
    looms_total = 0
    loom_on = _loom_ingest_enabled()
    for row in rows:
        md, redacted = _row_md(sheet, row, cols_by_id)
        redacted_total += redacted
        storage.write_text(f"{prefix}sheet-{sheet_id}-row-{row['id']}.md", md)
        if loom_on:
            looms_total += _ingest_row_looms(sheet_id, row, prefix)
    return {
        "sheet_id": sheet_id,
        "rows_written": len(rows),
        "secrets_redacted": redacted_total,
        "looms_analyzed": looms_total,
    }


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
    looms = sum(s.get("looms_analyzed", 0) for s in support + onboarding)
    logger.info(
        "[kb.smartsheet] support=%d onboarding=%d skipped=%d looms_analyzed=%d",
        len(support), len(onboarding), len(skipped), looms,
    )
    return {
        "source": "smartsheet",
        "sheets_seen": len(sheets),
        "support_sheets": support,
        "onboarding_sheets": onboarding,
        "skipped_sheet_ids": skipped,
        "support_rows": sum(s["rows_written"] for s in support),
        "onboarding_rows": sum(s["rows_written"] for s in onboarding),
        "looms_analyzed": looms,
    }
