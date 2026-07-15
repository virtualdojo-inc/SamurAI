"""Salesforce tools for Quotely org case management.

Tools:
- query_cases: Search/query cases (read-only)
- get_case_details: Get full case details by ID (read-only)
- add_case_comment: Add a comment to a case (judge-gated)
- update_case_status: Update case status (judge-gated)
"""

import os
import logging
import threading
import time

import requests
from langchain_core.tools import tool

from simple_salesforce import Salesforce, format_soql
from simple_salesforce.exceptions import SalesforceExpiredSession

logger = logging.getLogger(__name__)

SF_INSTANCE_URL = "https://quotely.my.salesforce.com"
SF_API_VERSION = "67.0"

# Cached Salesforce connection. Salesforce blocks *simultaneous* refresh-token
# exchanges of the same token ("Token request is already being processed") and
# caps concurrent access/refresh token pairs per user, so minting a fresh token
# on every tool call made bursts (e.g. closing 18 cases) fail intermittently
# with 400s. We exchange once, cache the access token, and reuse it; a lock
# single-flights the refresh so concurrent tool calls don't storm the endpoint.
# Ref: Salesforce "OAuth 2.0 Refresh Token Flow" — cache and reuse tokens.
_conn_lock = threading.Lock()
_cached_conn: "Salesforce | None" = None
_cached_conn_expiry: float = 0.0
# Conservative TTL — comfortably shorter than a typical SF session timeout, and
# the burst we care about happens within seconds either way. On a mid-TTL
# session expiry, tools invalidate the cache and the next call re-exchanges.
_TOKEN_TTL_SECONDS = 10 * 60


def _get_refresh_token() -> str:
    """Read the Salesforce OAuth refresh token from the environment.

    Injected by Cloud Run from Secret Manager via
    `--update-secrets=SF_CLI_REFRESH_TOKEN=sf-cli-refresh-token:latest`, matching
    how every other secret in this service is provided (env-var injection — see
    app.py, db/session.py). This avoids a runtime Secret Manager API round-trip,
    the extra `secretmanager.versions.access` IAM grant, and the
    google-cloud-secret-manager dependency.
    """
    token = os.environ.get("SF_CLI_REFRESH_TOKEN")
    if not token:
        raise RuntimeError(
            "SF_CLI_REFRESH_TOKEN is not set. Mount it on Cloud Run with "
            "--update-secrets=SF_CLI_REFRESH_TOKEN=sf-cli-refresh-token:latest."
        )
    return token


def _exchange_refresh_token() -> Salesforce:
    """Exchange the refresh token for a fresh access token + connection."""
    refresh_token = _get_refresh_token()
    client_id = "PlatformCLI"

    token_url = f"{SF_INSTANCE_URL}/services/oauth2/token"
    response = requests.post(token_url, data={
        'grant_type': 'refresh_token',
        'client_id': client_id,
        'refresh_token': refresh_token,
    }, timeout=30)
    # On a 4xx/5xx, Salesforce returns a JSON OAuth error body such as
    # {"error": "invalid_grant", "error_description": "..."}. raise_for_status()
    # discards it, leaving only a bare "400 Bad Request" in the traceback — which
    # is exactly what made the burst-close failures hard to diagnose. Log the
    # status, Salesforce request id, and body first (the body carries the error
    # code + description; it does NOT contain the refresh/access token).
    if not response.ok:
        logger.error(
            "[salesforce] token exchange failed: HTTP %s sfdc_request_id=%s body=%s",
            response.status_code,
            response.headers.get("Sfdc-Request-Id") or response.headers.get("x-request-id"),
            response.text[:1000],
        )
    response.raise_for_status()
    token_data = response.json()

    return Salesforce(
        instance_url=token_data['instance_url'],
        session_id=token_data['access_token'],
        version=SF_API_VERSION,
    )


def _create_sf_connection() -> Salesforce:
    """Return a cached Salesforce connection, refreshing at most once per TTL.

    Reuses one access token across tool calls (incl. a burst of case closes)
    instead of exchanging the refresh token every call. A threading lock
    single-flights the refresh so concurrent calls (tools run in a thread pool)
    don't send simultaneous same-refresh-token requests, which Salesforce
    rejects. Call _invalidate_sf_connection() when a session-expiry error is
    seen so the next call re-exchanges.
    """
    global _cached_conn, _cached_conn_expiry
    if _cached_conn is not None and time.monotonic() < _cached_conn_expiry:
        return _cached_conn
    with _conn_lock:
        # Re-check inside the lock: another thread may have refreshed while we
        # waited, so we don't pile on a second exchange.
        if _cached_conn is not None and time.monotonic() < _cached_conn_expiry:
            return _cached_conn
        conn = _exchange_refresh_token()
        _cached_conn = conn
        _cached_conn_expiry = time.monotonic() + _TOKEN_TTL_SECONDS
        return conn


def _invalidate_sf_connection() -> None:
    """Drop the cached connection so the next call re-exchanges (call on a
    session-expiry / auth error)."""
    global _cached_conn, _cached_conn_expiry
    with _conn_lock:
        _cached_conn = None
        _cached_conn_expiry = 0.0


@tool
def query_cases(
    subject_keyword: str = "",
    status: str = "",
    limit: int = 20,
    include_closed: bool = False,
) -> str:
    """List / search Salesforce support cases. THE tool for any case request.

    Use this for ANY request to list, search, or find cases — including phrasings
    like "list the cases", "show open cases", "salesforce cases", or "quotely
    cases" (the cases live in Quotely's Salesforce org). This is the ONLY tool for
    Salesforce Case records. Do NOT use the tenant/CRM support-grant tools
    (list_tenant_support_grants, read_tenant_records) for cases — those are for
    customer-tenant data access and will send the user through an SSO sign-in.

    Returns a summary of matching cases.

    Args:
        subject_keyword: Search keyword to match against case subjects (fuzzy).
        status: Filter by case status (e.g. 'New', 'Waiting for Customer', 'Closed').
        limit: Max number of results to return (default 20, max 200).
        include_closed: If True, include closed cases. Ignored when a specific
            `status` is given (that status governs). Default lists open cases only.
    """
    try:
        sf = _create_sf_connection()

        # Parameterized SOQL — values are bound via format_soql (escapes quotes /
        # LIKE wildcards) so a case subject/status with an apostrophe can't break
        # the query and untrusted input can't inject SOQL.
        where_clauses: list[str] = []
        params: list = []
        # Only force "open" when the caller neither asked for a specific status
        # nor opted into closed cases — otherwise the old hard IsClosed=false
        # silently contradicted status='Closed' and returned nothing.
        if not include_closed and not status:
            where_clauses.append("IsClosed = false")
        if subject_keyword:
            where_clauses.append("Subject LIKE '%{:like}%'")
            params.append(subject_keyword)
        if status:
            where_clauses.append("Status = {}")
            params.append(status)

        capped = min(max(int(limit), 1), 200)
        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        query = format_soql(
            "SELECT CaseNumber, Subject, Status, Priority, CreatedDate, Owner.Username "
            "FROM Case" + where_sql + " ORDER BY CreatedDate DESC LIMIT {}",
            *params,
            capped,
        )

        result = sf.query(query)
        records = result.get('records', [])

        if not records:
            return "No cases found matching the criteria."

        # Format results
        lines = [f"Found {len(records)} case(s):\n"]
        for rec in records:
            owner = rec.get('Owner', {})
            username = owner.get('Username', 'N/A') if isinstance(owner, dict) else owner
            created_date = rec.get('CreatedDate', '')
            if isinstance(created_date, str) and len(created_date) > 10:
                created_date = created_date[:10]

            lines.append(
                f"  {rec['CaseNumber']}: {rec['Subject']} "
                f"(Status: {rec['Status']}, Priority: {rec.get('Priority', 'N/A')}, "
                f"Created: {created_date}, Owner: {username})"
            )

        # We hit the row cap — there may be more. Note it accurately.
        if len(records) >= capped:
            lines.append(f"\n... showing the first {capped}. Narrow the search or raise `limit` for more.")

        return "\n".join(lines)

    except Exception as e:
        if isinstance(e, SalesforceExpiredSession):
            _invalidate_sf_connection()
        logger.exception("[salesforce] query_cases error")
        return "Error querying cases. The failure was logged for the team to review."


@tool
def get_case_details(case_id: str) -> str:
    """Get full details of one Salesforce support case (by ID or case number).

    Use for any single Salesforce case lookup. Pairs with query_cases. Not related
    to the tenant/CRM support-grant tools.

    Args:
        case_id: The Salesforce Case ID (e.g. '500XXXXXXXXXXXX' or case number like '00001009').
    """
    try:
        sf = _create_sf_connection()

        # Try to get the case by ID first
        case_fields = [
            "CaseNumber", "Subject", "Status", "Priority", "Description",
            "Reason", "Type", "Origin", "CreatedDate", "LastModifiedDate",
            "Owner.Username", "Account.Name", "Contact.Name", "Contact.Email",
            "Contact.Phone", "IsClosed"
        ]

        fields_sql = ', '.join(case_fields)  # fixed identifier list, not user input

        try:
            result = sf.query(
                format_soql(f"SELECT {fields_sql} FROM Case WHERE Id = {{}} LIMIT 1", case_id)
            )
        except Exception:
            # If direct ID lookup fails, try querying by CaseNumber
            result = sf.query(
                format_soql(f"SELECT {fields_sql} FROM Case WHERE CaseNumber = {{}} LIMIT 1", case_id)
            )

        records = result.get('records', [])

        if not records:
            return f"Case not found: {case_id}"

        rec = records[0]
        lines = [f"Case Details: {rec.get('CaseNumber', case_id)}\n"]
        lines.append(f"  Subject: {rec.get('Subject', 'N/A')}")
        lines.append(f"  Status: {rec.get('Status', 'N/A')}")
        lines.append(f"  Priority: {rec.get('Priority', 'N/A')}")
        lines.append(f"  Type: {rec.get('Type', 'N/A')}")
        lines.append(f"  Reason: {rec.get('Reason', 'N/A')}")
        lines.append(f"  Origin: {rec.get('Origin', 'N/A')}")

        owner = rec.get('Owner', {})
        owner_username = owner.get('Username', 'N/A') if isinstance(owner, dict) else owner
        lines.append(f"  Owner: {owner_username}")

        account = rec.get('Account', {})
        account_name = account.get('Name', 'N/A') if isinstance(account, dict) else account
        lines.append(f"  Account: {account_name}")

        contact = rec.get('Contact', {})
        contact_name = contact.get('Name', 'N/A') if isinstance(contact, dict) else contact
        contact_email = contact.get('Email', 'N/A') if isinstance(contact, dict) else 'N/A'
        contact_phone = contact.get('Phone', 'N/A') if isinstance(contact, dict) else 'N/A'
        lines.append(f"  Contact: {contact_name} ({contact_email}, {contact_phone})")

        created = rec.get('CreatedDate', 'N/A')
        if isinstance(created, str) and len(created) > 10:
            created = created[:10]
        lines.append(f"  Created: {created}")

        modified = rec.get('LastModifiedDate', 'N/A')
        if isinstance(modified, str) and len(modified) > 10:
            modified = modified[:10]
        lines.append(f"  Last Modified: {modified}")

        desc = rec.get('Description', '')
        if desc:
            lines.append(f"  Description: {desc[:500]}")

        closure = rec.get('ClosureNotes', '')
        if closure:
            lines.append(f"  Closure Notes: {closure[:500]}")

        return "\n".join(lines)

    except Exception as e:
        if isinstance(e, SalesforceExpiredSession):
            _invalidate_sf_connection()
        logger.exception("[salesforce] get_case_details error")
        return "Error retrieving case details. The failure was logged for the team to review."


@tool
def add_case_comment(
    case_id: str,
    comment: str,
    publish_to_customer: bool = False,
) -> str:
    """Add a comment to a Salesforce case. Internal-only by default.

    IMPORTANT: This is a judge-gated action. All write operations to Salesforce
    require approval through the safety judge before execution.

    By default the comment is INTERNAL (staff-only). Only set
    publish_to_customer=True when the user has EXPLICITLY asked to share/send the
    comment to the customer — publishing exposes the text to the customer via the
    portal and cannot be un-sent. If the user did not clearly ask to publish
    externally, leave it False.

    Args:
        case_id: The Salesforce Case ID or Case Number (e.g. '500XXXXXXXXXXXX' or '00001009').
        comment: The comment text to add. Keep it professional and specific.
        publish_to_customer: If True, the comment is published to the customer
            (customer-visible). Default False = internal/staff-only. Set True ONLY
            when the user explicitly requested sharing the comment with the customer.
    """
    try:
        sf = _create_sf_connection()

        # Resolve case number -> internal Id (parameterized to prevent SOQL injection).
        if not case_id.startswith('500'):
            result = sf.query(
                format_soql("SELECT Id FROM Case WHERE CaseNumber = {} LIMIT 1", case_id)
            )
            records = result.get('records', [])
            if not records:
                return f"Error: Case not found: {case_id}"
            case_id = records[0]['Id']

        # CaseComment is the correct object for a case comment. (The legacy Note
        # object does not attach as a case comment and has no public/internal
        # flag.) IsPublished=True makes the comment customer-visible; default is
        # internal. create() returns a dict with id/success.
        result = sf.CaseComment.create({
            'ParentId': case_id,
            'CommentBody': comment,
            'IsPublished': publish_to_customer,
        })

        if result.get('success'):
            visibility = 'customer-visible' if publish_to_customer else 'internal'
            return (
                f"Added {visibility} comment to case {case_id} "
                f"(CaseComment ID: {result.get('id')})."
            )
        errors = result.get('errors', ['Unknown error'])
        return f"Error adding comment to case {case_id}: {errors}"

    except Exception as e:
        if isinstance(e, SalesforceExpiredSession):
            _invalidate_sf_connection()
        logger.exception("[salesforce] add_case_comment error")
        return "Error adding comment. The failure was logged for the team to review."


@tool
def update_case_status(
    case_id: str,
    new_status: str,
    close_case: bool = False,
    closure_notes: str = "",
) -> str:
    """Update a Salesforce case status and optionally close it.

    IMPORTANT: This is a judge-gated action. All write operations to Salesforce
    require approval through the safety judge before execution.

    Args:
        case_id: The Salesforce Case ID or Case Number (e.g. '500XXXXXXXXXXXX' or '00001009').
        new_status: The new status to set (e.g. 'New', 'In Progress', 'Waiting for Customer', 'Closed').
        close_case: If True, sets Status to 'Closed' and records closure notes.
        closure_notes: Notes for when the case is closed. (Used only if close_case=True)
    """
    try:
        sf = _create_sf_connection()

        # Resolve case number -> internal Id (parameterized to prevent SOQL
        # injection — an unescaped case_id here could otherwise resolve and
        # then CLOSE the wrong case).
        if not case_id.startswith('500'):
            result = sf.query(
                format_soql("SELECT Id FROM Case WHERE CaseNumber = {} LIMIT 1", case_id)
            )
            records = result.get('records', [])
            if not records:
                return f"Error: Case not found: {case_id}"
            case_id = records[0]['Id']

        # Build the update payload. The record Id goes in the URL (passed as the
        # first positional arg below), NOT in the body.
        update_data = {'Status': new_status}

        if close_case:
            update_data['Status'] = 'Closed'
            if closure_notes:
                update_data['ClosureNotes'] = closure_notes

        # simple_salesforce: SFType.update(record_id, data) -> int HTTP status
        # code (204 on success), NOT a dict. Passing only the dict raises
        # "SFType.update() missing 1 required positional argument: 'data'".
        status_code = sf.Case.update(case_id, update_data)

        if 200 <= int(status_code) < 300:
            return f"Case {case_id} updated to status: {update_data['Status']}"
        return f"Error updating case {case_id}: Salesforce returned HTTP {status_code}"

    except Exception as e:
        if isinstance(e, SalesforceExpiredSession):
            _invalidate_sf_connection()
        logger.exception("[salesforce] update_case_status error")
        return "Error updating case status. The failure was logged for the team to review."


# Exported tool list (mirrors the other tools/ modules). query_cases and
# get_case_details are read-only; add_case_comment and update_case_status are
# judge-gated writes (registered in judge.py's write-tool list).
SALESFORCE_TOOLS = [
    query_cases,
    get_case_details,
    add_case_comment,
    update_case_status,
]
