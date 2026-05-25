"""Tools for reading and updating Smartsheet sheets via the REST API v2."""

import os
from typing import Any, Optional

import httpx
from langchain_core.tools import tool

_API_BASE = "https://api.smartsheet.com/2.0"


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
    """
    params: dict = {"pageSize": 100}
    if modified_since:
        params["modifiedSince"] = modified_since
    data = _get("/sheets", params=params)
    sheets = [
        {
            "id": s["id"],
            "name": s["name"],
            "modified_at": s.get("modifiedAt"),
            "access_level": s.get("accessLevel"),
            "permalink": s.get("permalink"),
        }
        for s in data.get("data", [])
    ]
    return {"total": data.get("totalCount", len(sheets)), "sheets": sheets}


@tool
def smartsheet_get_sheet(
    sheet_id: int,
    max_rows: int = 100,
    column_names: Optional[list[str]] = None,
) -> dict:
    """Read rows from a Smartsheet sheet as compact {column: value} dicts.

    Args:
        sheet_id: The numeric Smartsheet sheet ID (from smartsheet_list_sheets).
        max_rows: Maximum rows to return (default 100). The sheet may have more;
            check `total_rows` in the response.
        column_names: Optional list of column titles to include. If omitted, all
            columns are returned. Use this to reduce token usage on wide sheets.

    Returns:
        A dict with `name`, `total_rows`, `columns` (list of titles in order),
        and `rows` (list of {column_title: display_value, ...}). Each row
        also includes `_row_id` (the API row ID, required for
        smartsheet_update_row) and `_row_number` (display position).
        Empty cells are omitted from each row dict.
    """
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
            compact["_row_id"] = row.get("id")
            compact["_row_number"] = row.get("rowNumber")
            rows_out.append(compact)

    return {
        "name": sheet.get("name"),
        "total_rows": sheet.get("totalRowCount", len(sheet.get("rows", []))),
        "columns": titles_out,
        "rows": rows_out,
    }


@tool
def smartsheet_update_row(
    sheet_id: int,
    row_id: int,
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
        sheet_id: The numeric Smartsheet sheet ID.
        row_id: The numeric API row ID to update — the `_row_id` field on
            the row returned by smartsheet_get_sheet. Do NOT pass the
            user-facing "Row ID" column or `_row_number` — those are
            display values, not the API ID.
        cell_values: Mapping of column title -> new cell value. Column
            titles are matched against the live sheet; unknown titles
            raise (with the available titles listed) rather than silently
            no-op. Pass only the cells you want to change.

    Returns:
        A dict with `message` ("SUCCESS" on success), `resultCode`, and
        the updated row payload from Smartsheet.
    """
    cols = _get(f"/sheets/{sheet_id}/columns", params={"includeAll": "true"})
    col_by_title = {c["title"]: c["id"] for c in cols.get("data", [])}

    cells = []
    unknown = []
    for title, value in cell_values.items():
        if title not in col_by_title:
            unknown.append(title)
            continue
        cells.append({"columnId": col_by_title[title], "value": value})

    if unknown:
        raise RuntimeError(
            f"Unknown column titles for sheet {sheet_id}: {unknown}. "
            f"Available columns: {sorted(col_by_title)}"
        )
    if not cells:
        raise RuntimeError("cell_values is empty; nothing to update.")

    return _put(f"/sheets/{sheet_id}/rows", json_body=[{"id": row_id, "cells": cells}])


SMARTSHEET_TOOLS = [
    smartsheet_list_sheets,
    smartsheet_get_sheet,
    smartsheet_update_row,
]
