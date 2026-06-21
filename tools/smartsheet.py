"""Tools for reading and updating Smartsheet sheets via the REST API v2."""

import os
from typing import Any, Optional

import httpx
from langchain_core.tools import tool

_API_BASE = "https://api.smartsheet.com/2.0"


def _normalize_col(title: str) -> str:
    """Collapse a column title to alphanumerics + lowercase.

    Smartsheet column titles often contain spaces ("Github Issue No"), but
    LLMs reliably try to call them with Python-identifier formatting
    ("Github_Issue_No") or quoted-string formatting (`'"Github Issue No"'`).
    Normalizing both sides to alphanumerics lets the model use whichever
    form comes naturally.
    """
    return "".join(c.lower() for c in title if c.isalnum())


def _token() -> str:
    tok = os.environ.get("SMARTSHEET_API_TOKEN")
    if not tok:
        raise RuntimeError(
            "SMARTSHEET_API_TOKEN is not set. Wire the GCP secret "
            "'smartsheet-api-token' into the Cloud Run service."
        )
    return tok


def _get(path: str, params: Optional[dict] = None) -> dict:
    with httpx.Client(timeout=30.0) as client:
        r = client.get(
            f"{_API_BASE}{path}",
            headers={"Authorization": f"Bearer {_token()}"},
            params=params or {},
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Smartsheet API {r.status_code}: {r.text[:500]}")
        return r.json()


def _put(path: str, json_body: Any) -> dict:
    with httpx.Client(timeout=60.0) as client:
        r = client.put(
            f"{_API_BASE}{path}",
            headers={
                "Authorization": f"Bearer {_token()}",
                "Content-Type": "application/json",
            },
            json=json_body,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Smartsheet API {r.status_code}: {r.text[:500]}")
        return r.json()


@tool
def smartsheet_list_sheets(modified_since: Optional[str] = None) -> dict:
    """List Smartsheet sheets the API token has access to.

    Args:
        modified_since: Optional ISO-8601 timestamp (e.g. "2026-05-01T00:00:00Z").
            When provided, only sheets modified at or after this time are returned.

    Returns:
        A dict with `total` (int) and `sheets` (list of {id, name, modified_at,
        access_level, permalink}). Returns at most 100 sheets.

        IDs are returned as STRINGS, not numbers, because Smartsheet sheet
        IDs are 16-digit 64-bit integers and LLMs lose precision on the last
        few digits when they appear as JSON numbers. Pass the id back to
        smartsheet_get_sheet / smartsheet_update_row exactly as it was
        returned here — do not "clean it up" by converting to an int.
    """
    params: dict = {"pageSize": 100}
    if modified_since:
        params["modifiedSince"] = modified_since
    data = _get("/sheets", params=params)
    sheets = [
        {
            # Stringify the ID so the LLM can't accidentally truncate the
            # 16-digit number. The Smartsheet REST API accepts string IDs
            # in URL paths, so round-tripping str→URL is safe.
            "id": str(s["id"]),
            "name": s["name"],
            "modified_at": s.get("modifiedAt"),
            "access_level": s.get("accessLevel"),
            "permalink": s.get("permalink"),
        }
        for s in data.get("data", [])
    ]
    return {"total": data.get("totalCount", len(sheets)), "sheets": sheets}


def get_sheet(
    sheet_id: str,
    max_rows: int = 100,
    column_names: Optional[list[str]] = None,
) -> dict:
    """Read rows from a Smartsheet sheet as compact {column: value} dicts.

    Plain (non-tool) entry point so in-process workers (e.g. the tracker-triage
    pipeline) can read a sheet without routing through the LLM tool wrapper.
    The ``smartsheet_get_sheet`` tool is a thin wrapper over this.

    Returns a dict with ``name``, ``total_rows``, ``columns`` (titles in order),
    and ``rows`` (list of {column_title: display_value, ..., _row_id, _row_number}).
    Empty cells are omitted from each row dict.
    """
    sheet_id = str(sheet_id)
    params = {"exclude": "filteredOutRows,nonexistentCells"}
    sheet = _get(f"/sheets/{sheet_id}", params=params)

    col_by_id = {c["id"]: c["title"] for c in sheet.get("columns", [])}
    all_titles = [c["title"] for c in sheet.get("columns", [])]
    if column_names:
        wanted = {n for n in column_names}
        titles_out = [t for t in all_titles if t in wanted]
    else:
        titles_out = all_titles
    title_set = set(titles_out)

    rows_out = []
    for row in sheet.get("rows", [])[:max_rows]:
        compact = {}
        for cell in row.get("cells", []):
            title = col_by_id.get(cell.get("columnId"))
            if title not in title_set:
                continue
            value = cell.get("displayValue")
            if value is None:
                value = cell.get("value")
            if value is None or value == "":
                continue
            compact[title] = value
        if compact:
            # _row_id is stringified for the same reason as sheet_id: LLMs
            # truncate the last digits of 16-digit numeric IDs in tool calls.
            compact["_row_id"] = str(row["id"]) if row.get("id") is not None else None
            compact["_row_number"] = row.get("rowNumber")
            rows_out.append(compact)

    return {
        "name": sheet.get("name"),
        "total_rows": sheet.get("totalRowCount", len(sheet.get("rows", []))),
        "columns": titles_out,
        "rows": rows_out,
    }


@tool
def smartsheet_get_sheet(
    sheet_id: str,
    max_rows: int = 100,
    column_names: Optional[list[str]] = None,
) -> dict:
    """Read rows from a Smartsheet sheet as compact {column: value} dicts.

    Args:
        sheet_id: The Smartsheet sheet ID as a STRING (from
            smartsheet_list_sheets — pass it through exactly as returned).
            Smartsheet sheet IDs are 16-digit 64-bit integers; passing them
            as JSON numbers risks the LLM truncating the last few digits
            into a 404. Always quote it.
        max_rows: Maximum rows to return (default 100). The sheet may have more;
            check `total_rows` in the response.
        column_names: Optional list of column titles to include. If omitted, all
            columns are returned. Use this to reduce token usage on wide sheets.

    Returns:
        A dict with `name`, `total_rows`, `columns` (list of titles in order),
        and `rows` (list of {column_title: display_value, ...}). Each row
        also includes `_row_id` (string — pass it through verbatim to
        smartsheet_update_row) and `_row_number` (display position).
        Empty cells are omitted from each row dict.
    """
    return get_sheet(sheet_id, max_rows=max_rows, column_names=column_names)


@tool
def smartsheet_update_row(
    sheet_id: str,
    row_id: str,
    cell_values: dict,
) -> dict:
    """Update one or more cells on a single existing Smartsheet row.

    Use this when the user asks to change anything on a specific row — set
    a priority, mark something done, fix a typo in a description, etc.
    Natural-language requests like "update issue 692's priority to High in
    the DH Tech tracker" route here.

    Workflow:
    1. Call smartsheet_list_sheets if you don't know the sheet_id.
    2. Call smartsheet_get_sheet (optionally with column_names to keep it
       cheap) to locate the target row. Find the row by whatever identifier
       the user mentioned — a "Github Issue No" cell, a name, a status —
       then read the `_row_id` field on that row.
    3. Call this tool with sheet_id, that _row_id, and cell_values.
    4. Read back with smartsheet_get_sheet to verify the change landed.

    Args:
        sheet_id: The Smartsheet sheet ID as a STRING (from
            smartsheet_list_sheets). Always quote it — LLMs lose precision
            on 16-digit numeric IDs and produce 404s.
        row_id: The API row ID as a STRING — the `_row_id` field on the
            row returned by smartsheet_get_sheet. Pass it through verbatim.
            Do NOT pass the user-facing "Row ID" column or `_row_number` —
            those are display values, not the API ID.
        cell_values: Mapping of column title -> new cell value. Column
            titles are matched against the live sheet; unknown titles
            raise (with the available titles listed) rather than silently
            no-op. Pass only the cells you want to change. Titles are
            matched flexibly — "Github Issue No", "Github_Issue_No", and
            "github issue no" all resolve to the same column — but the
            EXACT title from smartsheet_get_sheet is always safest. Do
            not wrap the title in extra quotes.

    Returns:
        A dict with `message` ("SUCCESS" on success), `resultCode`, and
        the updated row payload from Smartsheet.
    """
    # Normalize the IDs at the boundary. The LLM-facing signature is str
    # (precision protection) but Smartsheet's JSON body expects an integer
    # `id` field. Python ints have unlimited precision, so str -> int is
    # lossless inside the process.
    sheet_id_str = str(sheet_id)
    try:
        row_id_int = int(str(row_id))
    except ValueError:
        raise RuntimeError(
            f"row_id must be a numeric ID (got {row_id!r}). Pass the _row_id "
            f"field from smartsheet_get_sheet verbatim — not the user-facing "
            f"'Row ID' column."
        )

    cols = _get(f"/sheets/{sheet_id_str}/columns", params={"includeAll": "true"})
    col_by_title = {c["title"]: c["id"] for c in cols.get("data", [])}

    # Build a normalized index for fuzzy matching. Ambiguous normalized keys
    # (two columns that collapse to the same form) are recorded but not
    # used — those must be passed by exact title.
    norm_index: dict[str, list[int]] = {}
    for title, cid in col_by_title.items():
        norm_index.setdefault(_normalize_col(title), []).append(cid)

    cells = []
    unknown = []
    ambiguous = []
    for title, value in cell_values.items():
        cid = col_by_title.get(title)
        if cid is None:
            candidates = norm_index.get(_normalize_col(title), [])
            if len(candidates) == 1:
                cid = candidates[0]
            elif len(candidates) > 1:
                ambiguous.append(title)
                continue
        if cid is None:
            unknown.append(title)
            continue
        cells.append({"columnId": cid, "value": value})

    if ambiguous:
        raise RuntimeError(
            f"Column titles {ambiguous} matched multiple columns on sheet "
            f"{sheet_id} after normalization. Pass the exact title from "
            f"smartsheet_get_sheet. Available columns: {sorted(col_by_title)}"
        )
    if unknown:
        raise RuntimeError(
            f"Unknown column titles for sheet {sheet_id}: {unknown}. "
            f"Available columns: {sorted(col_by_title)}"
        )
    if not cells:
        raise RuntimeError("cell_values is empty; nothing to update.")

    try:
        return _put(
            f"/sheets/{sheet_id_str}/rows",
            json_body=[{"id": row_id_int, "cells": cells}],
        )
    except RuntimeError as e:
        # Smartsheet returns 404 with errorCode 1006 ("Not Found") when
        # the rowId doesn't exist on the sheet. Two common causes:
        # (a) the LLM passed the sheet_id as row_id by accident, or
        # (b) the LLM passed the user-facing "Row ID" column value (a
        # short int displayed in the sheet) instead of `_row_id` (the
        # 16-digit API ID). Re-raise with an actionable hint.
        msg = str(e)
        if "404" in msg and ("1006" in msg or "Not Found" in msg):
            hint = (
                f"Smartsheet returned 404 Not Found for row_id={row_id_int} "
                f"on sheet {sheet_id_str}. The row ID you passed does not "
                f"exist on this sheet. Common mistakes:\n"
                f"  - Did you pass the sheet_id ({sheet_id_str}) as the "
                f"row_id by accident? sheet_id and row_id are different.\n"
                f"  - Did you pass the user-facing 'Row ID' column value "
                f"(a small int like 56 or 101)? That's a display value, not "
                f"the API row ID. Use the `_row_id` field on each row "
                f"returned by smartsheet_get_sheet — it's a ~16-digit number.\n"
                f"Original error: {msg}"
            )
            raise RuntimeError(hint) from e
        raise


SMARTSHEET_TOOLS = [
    smartsheet_list_sheets,
    smartsheet_get_sheet,
    smartsheet_update_row,
]
