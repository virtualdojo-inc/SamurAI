"""Tests for tools.smartsheet — list sheets and read sheet rows."""

import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _token_env(monkeypatch):
    monkeypatch.setenv("SMARTSHEET_API_TOKEN", "test-token")


def _sheet_list_payload():
    return {
        "pageNumber": 1,
        "pageSize": 100,
        "totalCount": 2,
        "data": [
            {
                "id": 111,
                "name": "Issue Tracker",
                "accessLevel": "OWNER",
                "modifiedAt": "2026-05-20T12:00:00Z",
                "permalink": "https://app.smartsheet.com/sheets/aaa",
            },
            {
                "id": 222,
                "name": "Project Plan",
                "accessLevel": "EDITOR_SHARE",
                "modifiedAt": "2026-05-19T08:00:00Z",
                "permalink": "https://app.smartsheet.com/sheets/bbb",
            },
        ],
    }


def _sheet_detail_payload():
    return {
        "name": "Issue Tracker",
        "totalRowCount": 3,
        "columns": [
            {"id": 1, "title": "Priority", "type": "PICKLIST"},
            {"id": 2, "title": "Status", "type": "PICKLIST"},
            {"id": 3, "title": "Description", "type": "TEXT_NUMBER"},
        ],
        "rows": [
            {
                "rowNumber": 1,
                "cells": [
                    {"columnId": 1, "value": "High", "displayValue": "High"},
                    {"columnId": 2, "value": "Open", "displayValue": "Open"},
                    {"columnId": 3, "value": "Login broken", "displayValue": "Login broken"},
                ],
            },
            {
                "rowNumber": 2,
                "cells": [
                    {"columnId": 1, "value": "Low"},
                    {"columnId": 2, "value": "Closed"},
                    {"columnId": 3, "value": None},  # empty cell should drop
                ],
            },
            {
                "rowNumber": 3,
                "cells": [
                    {"columnId": 1, "value": "Med", "displayValue": "Med"},
                    {"columnId": 3, "displayValue": "No status set"},
                ],
            },
        ],
    }


# --- smartsheet_list_sheets ---


@patch("tools.smartsheet._get")
def test_list_sheets_returns_compact_records(mock_get):
    from tools.smartsheet import smartsheet_list_sheets

    mock_get.return_value = _sheet_list_payload()
    result = smartsheet_list_sheets.invoke({})

    assert result["total"] == 2
    assert len(result["sheets"]) == 2
    first = result["sheets"][0]
    # IDs are STRINGIFIED to prevent LLM precision loss on 16-digit IDs.
    assert first["id"] == "111"
    assert isinstance(first["id"], str)
    assert first["name"] == "Issue Tracker"
    assert first["modified_at"] == "2026-05-20T12:00:00Z"
    assert first["access_level"] == "OWNER"
    assert "permalink" in first


@patch("tools.smartsheet._get")
def test_list_sheets_passes_modified_since(mock_get):
    from tools.smartsheet import smartsheet_list_sheets

    mock_get.return_value = {"data": [], "totalCount": 0}
    smartsheet_list_sheets.invoke({"modified_since": "2026-05-01T00:00:00Z"})

    path, kwargs = mock_get.call_args.args[0], mock_get.call_args.kwargs
    assert path == "/sheets"
    assert kwargs["params"]["modifiedSince"] == "2026-05-01T00:00:00Z"
    assert kwargs["params"]["pageSize"] == 100


# --- smartsheet_get_sheet ---


@patch("tools.smartsheet._get")
def test_get_sheet_returns_compact_rows(mock_get):
    from tools.smartsheet import smartsheet_get_sheet

    mock_get.return_value = _sheet_detail_payload()
    result = smartsheet_get_sheet.invoke({"sheet_id": "111"})

    assert result["name"] == "Issue Tracker"
    assert result["total_rows"] == 3
    assert result["columns"] == ["Priority", "Status", "Description"]
    assert len(result["rows"]) == 3

    # Empty cells are stripped.
    row2 = next(r for r in result["rows"] if r["_row_number"] == 2)
    assert "Description" not in row2
    assert row2["Priority"] == "Low"
    assert row2["Status"] == "Closed"

    # displayValue is preferred when present, falls back to value.
    row1 = next(r for r in result["rows"] if r["_row_number"] == 1)
    assert row1["Description"] == "Login broken"


@patch("tools.smartsheet._get")
def test_get_sheet_column_filter(mock_get):
    from tools.smartsheet import smartsheet_get_sheet

    mock_get.return_value = _sheet_detail_payload()
    result = smartsheet_get_sheet.invoke(
        {"sheet_id": "111", "column_names": ["Priority", "Status"]}
    )

    assert result["columns"] == ["Priority", "Status"]
    for row in result["rows"]:
        assert set(row.keys()) <= {"Priority", "Status", "_row_id", "_row_number"}


@patch("tools.smartsheet._get")
def test_get_sheet_respects_max_rows(mock_get):
    from tools.smartsheet import smartsheet_get_sheet

    mock_get.return_value = _sheet_detail_payload()
    result = smartsheet_get_sheet.invoke({"sheet_id": "111", "max_rows": 1})

    assert len(result["rows"]) == 1
    assert result["rows"][0]["_row_number"] == 1


@patch("tools.smartsheet._get")
def test_get_sheet_excludes_nonexistent_cells(mock_get):
    from tools.smartsheet import smartsheet_get_sheet

    mock_get.return_value = _sheet_detail_payload()
    smartsheet_get_sheet.invoke({"sheet_id": "111"})

    path, kwargs = mock_get.call_args.args[0], mock_get.call_args.kwargs
    assert path == "/sheets/111"
    assert kwargs["params"]["exclude"] == "filteredOutRows,nonexistentCells"


# --- auth ---


def test_missing_token_raises(monkeypatch):
    from tools.smartsheet import _token

    monkeypatch.delenv("SMARTSHEET_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="SMARTSHEET_API_TOKEN"):
        _token()


@patch("tools.smartsheet.httpx.Client")
def test_get_includes_bearer_header(mock_client_cls):
    from tools.smartsheet import _get

    instance = mock_client_cls.return_value.__enter__.return_value
    response = instance.get.return_value
    response.status_code = 200
    response.json.return_value = {"ok": True}

    _get("/sheets")

    headers = instance.get.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer test-token"


@patch("tools.smartsheet.httpx.Client")
def test_get_raises_on_http_error(mock_client_cls):
    from tools.smartsheet import _get

    instance = mock_client_cls.return_value.__enter__.return_value
    response = instance.get.return_value
    response.status_code = 401
    response.text = "Unauthorized"

    with pytest.raises(RuntimeError, match="401"):
        _get("/sheets")


# --- smartsheet_update_row (renamed from smartsheet_update_row_cells) ---


def test_smartsheet_update_row_is_registered_under_natural_name():
    """The tool name must be `smartsheet_update_row` — the previous name
    `smartsheet_update_row_cells` was unintuitive enough that Gemini kept
    hallucinating the shorter form and bailing on real update requests."""
    from tools.smartsheet import SMARTSHEET_TOOLS

    names = {t.name for t in SMARTSHEET_TOOLS}
    assert "smartsheet_update_row" in names
    assert "smartsheet_update_row_cells" not in names


@patch("tools.smartsheet._put")
@patch("tools.smartsheet._get")
def test_update_row_resolves_column_titles_to_ids(mock_get, mock_put):
    from tools.smartsheet import smartsheet_update_row

    mock_get.return_value = {
        "data": [
            {"id": 9001, "title": "Priority"},
            {"id": 9002, "title": "Status"},
        ]
    }
    mock_put.return_value = {"message": "SUCCESS", "resultCode": 0}

    result = smartsheet_update_row.invoke(
        {
            "sheet_id": "111",
            "row_id": "7777",
            "cell_values": {"Priority": "High", "Status": "In Progress"},
        }
    )

    assert result["message"] == "SUCCESS"
    sent = mock_put.call_args
    body = sent.kwargs["json_body"]
    assert body[0]["id"] == 7777
    sent_cells = {c["columnId"]: c["value"] for c in body[0]["cells"]}
    assert sent_cells == {9001: "High", 9002: "In Progress"}


@patch("tools.smartsheet._get")
def test_update_row_rejects_unknown_column(mock_get):
    from tools.smartsheet import smartsheet_update_row

    mock_get.return_value = {
        "data": [{"id": 9001, "title": "Priority"}]
    }

    with pytest.raises(RuntimeError, match="Unknown column titles"):
        smartsheet_update_row.invoke(
            {
                "sheet_id": "111",
                "row_id": "7777",
                "cell_values": {"NotARealColumn": "x"},
            }
        )


@patch("tools.smartsheet._get")
def test_update_row_rejects_empty_cell_values(mock_get):
    from tools.smartsheet import smartsheet_update_row

    mock_get.return_value = {"data": [{"id": 9001, "title": "Priority"}]}

    with pytest.raises(RuntimeError, match="nothing to update"):
        smartsheet_update_row.invoke(
            {"sheet_id": "111", "row_id": "7777", "cell_values": {}}
        )


# --- Column-title fuzzy matching (2026-05 Devin incident) ---


@patch("tools.smartsheet._put")
@patch("tools.smartsheet._get")
def test_update_row_fuzzy_matches_python_identifier_style(mock_get, mock_put):
    """Gemini keeps formatting 'Github Issue No' as 'Github_Issue_No' (Python
    identifier style) and getting Unknown-column errors. Fuzzy match should
    rescue these and route to the correct column."""
    from tools.smartsheet import smartsheet_update_row

    mock_get.return_value = {
        "data": [
            {"id": 9001, "title": "Github Issue No"},
            {"id": 9002, "title": "Priority"},
        ]
    }
    mock_put.return_value = {"message": "SUCCESS", "resultCode": 0}

    result = smartsheet_update_row.invoke(
        {
            "sheet_id": "111",
            "row_id": "7777",
            "cell_values": {"Github_Issue_No": "#711"},
        }
    )

    assert result["message"] == "SUCCESS"
    body = mock_put.call_args.kwargs["json_body"]
    assert body[0]["cells"][0]["columnId"] == 9001
    assert body[0]["cells"][0]["value"] == "#711"


@patch("tools.smartsheet._put")
@patch("tools.smartsheet._get")
def test_update_row_fuzzy_matches_extra_quotes(mock_get, mock_put):
    """Model sometimes wraps the title in extra quotes: '\"Github Issue No\"'."""
    from tools.smartsheet import smartsheet_update_row

    mock_get.return_value = {"data": [{"id": 9001, "title": "Github Issue No"}]}
    mock_put.return_value = {"message": "SUCCESS"}

    smartsheet_update_row.invoke(
        {
            "sheet_id": "111",
            "row_id": "7777",
            "cell_values": {'"Github Issue No"': "#711"},
        }
    )

    body = mock_put.call_args.kwargs["json_body"]
    assert body[0]["cells"][0]["columnId"] == 9001


@patch("tools.smartsheet._put")
@patch("tools.smartsheet._get")
def test_update_row_fuzzy_matches_lowercase(mock_get, mock_put):
    from tools.smartsheet import smartsheet_update_row

    mock_get.return_value = {"data": [{"id": 9001, "title": "Github Issue No"}]}
    mock_put.return_value = {"message": "SUCCESS"}

    smartsheet_update_row.invoke(
        {"sheet_id": "111", "row_id": "7777", "cell_values": {"github issue no": "#711"}}
    )

    body = mock_put.call_args.kwargs["json_body"]
    assert body[0]["cells"][0]["columnId"] == 9001


@patch("tools.smartsheet._get")
def test_update_row_rejects_ambiguous_normalized_match(mock_get):
    """If two columns normalize to the same form, the fuzzy match must NOT
    silently pick one — that would corrupt the wrong column."""
    from tools.smartsheet import smartsheet_update_row

    mock_get.return_value = {
        "data": [
            {"id": 9001, "title": "Github Issue No"},
            {"id": 9002, "title": "GITHUBISSUENO"},  # normalizes to same form
        ]
    }

    with pytest.raises(RuntimeError, match="matched multiple columns"):
        smartsheet_update_row.invoke(
            {
                "sheet_id": "111",
                "row_id": "7777",
                "cell_values": {"Github_Issue_No": "#711"},
            }
        )


@patch("tools.smartsheet._put")
@patch("tools.smartsheet._get")
def test_update_row_exact_match_still_wins_when_present(mock_get, mock_put):
    """Exact title should always be preferred — fuzzy match is a fallback."""
    from tools.smartsheet import smartsheet_update_row

    mock_get.return_value = {
        "data": [
            {"id": 9001, "title": "Github_Issue_No"},  # exact match
            {"id": 9002, "title": "Github Issue No"},  # fuzzy match
        ]
    }
    mock_put.return_value = {"message": "SUCCESS"}

    smartsheet_update_row.invoke(
        {
            "sheet_id": "111",
            "row_id": "7777",
            "cell_values": {"Github_Issue_No": "#711"},
        }
    )

    body = mock_put.call_args.kwargs["json_body"]
    assert body[0]["cells"][0]["columnId"] == 9001  # exact match wins


# --- 16-digit-ID precision protection (Devin May 25 incident) ---


@patch("tools.smartsheet._get")
def test_list_sheets_returns_ids_as_strings(mock_get):
    """16-digit Smartsheet IDs must be returned as strings, not numbers.

    Devin's May 25 session: smartsheet_list_sheets returned ID
    1146352141553540 as a JSON number, the LLM transcribed it as
    1146352141553536 on the next tool call (last 3 digits corrupted),
    Smartsheet returned 404. Stringifying defends against this.
    """
    from tools.smartsheet import smartsheet_list_sheets

    mock_get.return_value = {
        "totalCount": 1,
        "data": [
            {
                "id": 1146352141553540,  # the actual DH Tech tracker ID
                "name": "Issue Tracker DH Tech",
                "accessLevel": "EDITOR_SHARE",
                "modifiedAt": "2026-05-25T12:00:00Z",
                "permalink": "https://app.smartsheet.com/sheets/xyz",
            }
        ],
    }
    result = smartsheet_list_sheets.invoke({})
    sheet = result["sheets"][0]

    assert isinstance(sheet["id"], str)
    # The exact 16 digits must round-trip — no precision loss
    assert sheet["id"] == "1146352141553540"


@patch("tools.smartsheet._get")
def test_get_sheet_accepts_string_sheet_id(mock_get):
    """The LLM-facing signature is str. Passing the ID as a string must
    work end-to-end — that's the whole point."""
    from tools.smartsheet import smartsheet_get_sheet

    mock_get.return_value = {
        "name": "X",
        "totalRowCount": 0,
        "columns": [],
        "rows": [],
    }
    smartsheet_get_sheet.invoke({"sheet_id": "1146352141553540"})

    # The URL must contain the exact 16-digit ID — not 1.14635214155354e+15
    # or any other scientific-notation / float-rounded variant.
    called_path = mock_get.call_args.args[0]
    assert called_path == "/sheets/1146352141553540"


@patch("tools.smartsheet._get")
def test_get_sheet_stringifies_row_ids_in_output(mock_get):
    """Row IDs are also 16-digit ints — must be stringified too so the LLM
    can pass them to smartsheet_update_row without precision loss."""
    from tools.smartsheet import smartsheet_get_sheet

    mock_get.return_value = {
        "name": "X",
        "totalRowCount": 1,
        "columns": [{"id": 1, "title": "A"}],
        "rows": [
            {
                "id": 7458800573808516,
                "rowNumber": 1,
                "cells": [{"columnId": 1, "value": "hi", "displayValue": "hi"}],
            }
        ],
    }
    result = smartsheet_get_sheet.invoke({"sheet_id": "111"})
    row = result["rows"][0]
    assert isinstance(row["_row_id"], str)
    assert row["_row_id"] == "7458800573808516"


@patch("tools.smartsheet._put")
@patch("tools.smartsheet._get")
def test_update_row_accepts_16_digit_string_ids_round_trip(mock_get, mock_put):
    """Pass the IDs through as strings end-to-end; the JSON body to
    Smartsheet must carry the exact same digits."""
    from tools.smartsheet import smartsheet_update_row

    mock_get.return_value = {"data": [{"id": 9001, "title": "Priority"}]}
    mock_put.return_value = {"message": "SUCCESS"}

    smartsheet_update_row.invoke(
        {
            "sheet_id": "1146352141553540",
            "row_id": "7458800573808516",
            "cell_values": {"Priority": "High"},
        }
    )

    # URL has the sheet_id verbatim
    assert mock_put.call_args.args[0] == "/sheets/1146352141553540/rows"
    # Body has the row_id as an int with no precision loss — Python int
    # is arbitrary-precision so str -> int -> JSON round-trips losslessly
    body = mock_put.call_args.kwargs["json_body"]
    assert body[0]["id"] == 7458800573808516


def test_update_row_rejects_non_numeric_row_id():
    """A row_id that isn't a numeric string is almost certainly the
    user-facing 'Row ID' column or something else wrong — fail loud."""
    from tools.smartsheet import smartsheet_update_row

    with pytest.raises(RuntimeError, match="numeric ID"):
        smartsheet_update_row.invoke(
            {
                "sheet_id": "111",
                "row_id": "not-a-number",
                "cell_values": {"X": "y"},
            }
        )


# --- 404 row-not-found hint (2026-05-25 incident) ---


@patch("tools.smartsheet._put")
@patch("tools.smartsheet._get")
def test_update_row_404_hints_at_sheet_id_confusion(mock_get, mock_put):
    """When Smartsheet returns 404 errorCode 1006 (row not found), the
    re-raised error must explain the two common LLM mistakes (passing
    sheet_id as row_id, or using the user-facing 'Row ID' column)."""
    from tools.smartsheet import smartsheet_update_row

    mock_get.return_value = {"data": [{"id": 9001, "title": "Priority"}]}
    mock_put.side_effect = RuntimeError(
        'Smartsheet API 404: {"refId":"abc","errorCode":1006,'
        '"message":"Not Found","detail":{"rowId":1146352141553540}}'
    )

    with pytest.raises(RuntimeError) as exc_info:
        smartsheet_update_row.invoke(
            {
                "sheet_id": "1146352141553540",
                "row_id": "1146352141553540",  # sheet_id accidentally used as row_id
                "cell_values": {"Priority": "High"},
            }
        )

    msg = str(exc_info.value)
    # Hint must call out the sheet_id-as-row_id confusion
    assert "sheet_id" in msg.lower() and "row_id" in msg.lower()
    # Hint must explain the Row ID column vs _row_id distinction
    assert "_row_id" in msg
    assert "Row ID" in msg
    # Original error is preserved so the model can still see the raw 404
    assert "404" in msg


@patch("tools.smartsheet._put")
@patch("tools.smartsheet._get")
def test_update_row_non_404_errors_pass_through_unchanged(mock_get, mock_put):
    """Only 404/1006 gets the hint treatment — other errors (auth, rate
    limit, etc.) propagate as-is so the model doesn't get misleading advice."""
    from tools.smartsheet import smartsheet_update_row

    mock_get.return_value = {"data": [{"id": 9001, "title": "Priority"}]}
    mock_put.side_effect = RuntimeError("Smartsheet API 429: rate limited")

    with pytest.raises(RuntimeError, match="429"):
        smartsheet_update_row.invoke(
            {
                "sheet_id": "111",
                "row_id": "7458800573808516",
                "cell_values": {"Priority": "High"},
            }
        )
