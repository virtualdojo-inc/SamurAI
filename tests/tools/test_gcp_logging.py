"""Tests for tools.gcp_logging.query_cloud_logs."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


def _make_entry(payload="some error", severity="ERROR", timestamp=None):
    entry = MagicMock()
    entry.payload = payload
    entry.severity = severity
    entry.timestamp = timestamp
    return entry


@patch("google.cloud.logging.Client")
def test_returns_formatted_entries(mock_client_cls):
    from tools.gcp_logging import query_cloud_logs

    ts = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    entries = [
        _make_entry("disk full", "ERROR", ts),
        _make_entry("high latency", "WARNING", ts),
    ]
    mock_client_cls.return_value.list_entries.return_value = entries

    result = query_cloud_logs.invoke({"filter_query": 'severity>=ERROR'})

    assert "disk full" in result
    assert "high latency" in result
    assert "ERROR" in result
    assert "WARNING" in result


@patch("google.cloud.logging.Client")
def test_no_entries_found(mock_client_cls):
    from tools.gcp_logging import query_cloud_logs

    mock_client_cls.return_value.list_entries.return_value = []

    result = query_cloud_logs.invoke({"filter_query": "resource.type=\"nothing\""})
    assert result == "No log entries found for that filter."


@patch("google.cloud.logging.Client")
def test_uses_env_project_id(mock_client_cls):
    from tools.gcp_logging import query_cloud_logs

    mock_client_cls.return_value.list_entries.return_value = []

    query_cloud_logs.invoke({"filter_query": "test"})
    mock_client_cls.assert_called_with(project="test-project")


@patch("google.cloud.logging.Client")
def test_explicit_project_id_overrides_env(mock_client_cls):
    from tools.gcp_logging import query_cloud_logs

    mock_client_cls.return_value.list_entries.return_value = []

    query_cloud_logs.invoke({"filter_query": "test", "project_id": "other-project"})
    mock_client_cls.assert_called_with(project="other-project")


@patch("google.cloud.logging.Client")
def test_filter_query_passed_through(mock_client_cls):
    from google.cloud import logging as gcl

    from tools.gcp_logging import query_cloud_logs

    mock_client_cls.return_value.list_entries.return_value = []
    filter_str = 'resource.type="cloud_run_revision" severity>=ERROR'

    query_cloud_logs.invoke({"filter_query": filter_str})
    # Now also requests newest-first ordering (the recent-errors fix).
    mock_client_cls.return_value.list_entries.assert_called_once_with(
        filter_=filter_str, order_by=gcl.DESCENDING, max_results=50
    )


@patch("google.cloud.logging.Client")
def test_entry_with_no_timestamp(mock_client_cls):
    from tools.gcp_logging import query_cloud_logs

    entries = [_make_entry("oops", "ERROR", None)]
    mock_client_cls.return_value.list_entries.return_value = entries

    result = query_cloud_logs.invoke({"filter_query": "test"})
    assert "[?]" in result
    assert "oops" in result
