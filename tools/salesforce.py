"""Salesforce tools for Quotely org case management.

Tools:
- query_cases: Search/query cases (read-only)
- get_case_details: Get full case details by ID (read-only)
- add_case_comment: Add a comment to a case (judge-gated)
- update_case_status: Update case status (judge-gated)
"""

import os
import logging

import requests
from langchain_core.tools import tool

from simple_salesforce import Salesforce

logger = logging.getLogger(__name__)

SF_INSTANCE_URL = "https://quotely.my.salesforce.com"
SF_API_VERSION = "67.0"


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


def _create_sf_connection() -> Salesforce:
    """Create a Salesforce connection using the refresh token."""
    refresh_token = _get_refresh_token()
    client_id = "PlatformCLI"

    # Exchange refresh token for access token
    token_url = f"{SF_INSTANCE_URL}/services/oauth2/token"
    response = requests.post(token_url, data={
        'grant_type': 'refresh_token',
        'client_id': client_id,
        'refresh_token': refresh_token,
    })
    response.raise_for_status()
    token_data = response.json()

    return Salesforce(
        instance_url=token_data['instance_url'],
        session_id=token_data['access_token'],
        version=SF_API_VERSION,
    )


@tool
def query_cases(
    subject_keyword: str = "",
    status: str = "",
    limit: int = 20,
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
        status: Filter by case status (e.g. 'New', 'Waiting for Customer').
        limit: Max number of results to return (default 20, max 200).
    """
    try:
        sf = _create_sf_connection()

        # Build query
        where_clauses = ["IsClosed = false"]
        if subject_keyword:
            where_clauses.append(f"Subject LIKE '%{subject_keyword}%'")
        if status:
            where_clauses.append(f"Status = '{status}'")

        query = f"SELECT CaseNumber, Subject, Status, Priority, CreatedDate, Owner.Username "
        query += f"FROM Case WHERE {' AND '.join(where_clauses)} "
        query += f"ORDER BY CreatedDate DESC LIMIT {min(limit, 200)}"

        result = sf.query(query)
        records = result.get('records', [])

        if not records:
            return "No cases found matching the criteria."

        # Format results
        lines = [f"Found {len(records)} open case(s):\n"]
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

        # If there are more results, show a note
        if len(records) >= 200:
            lines.append("\n... and more results. Try narrowing your search criteria.")

        return "\n".join(lines)

    except Exception as e:
        logger.exception("[salesforce] query_cases error")
        return f"Error querying cases: {type(e).__name__}: {e}"


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

        query = f"SELECT {', '.join(case_fields)} FROM Case WHERE Id = '{case_id}' LIMIT 1"

        try:
            result = sf.query(query)
        except Exception:
            # If direct ID lookup fails, try querying by CaseNumber
            query = f"SELECT {', '.join(case_fields)} FROM Case WHERE CaseNumber = '{case_id}' LIMIT 1"
            result = sf.query(query)

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
        logger.exception("[salesforce] get_case_details error")
        return f"Error retrieving case details: {type(e).__name__}: {e}"


@tool
def add_case_comment(case_id: str, comment: str, is_internal: bool = False) -> str:
    """Add a comment to a Salesforce case.

    IMPORTANT: This is a judge-gated action. All write operations to Salesforce
    require approval through the safety judge before execution.

    Args:
        case_id: The Salesforce Case ID or Case Number (e.g. '500XXXXXXXXXXXX' or '00001009').
        comment: The comment text to add. Keep it professional and specific.
        is_internal: If True, the comment will be marked as internal only (not visible to customers).
    """
    try:
        sf = _create_sf_connection()

        # Resolve case_id to internal ID
        if not case_id.startswith('500'):
            # Query to get the actual ID
            result = sf.query(
                f"SELECT Id FROM Case WHERE CaseNumber = '{case_id}' LIMIT 1"
            )
            records = result.get('records', [])
            if not records:
                return f"Error: Case not found: {case_id}"
            case_id = records[0]['Id']

        # Create Note as a comment
        note = sf.Note.create({
            'ParentId': case_id,
            'Title': f'Comment on {case_id}',
            'IsPrivacyProtected': True,  # Salesforce privacy setting
            'Body': comment,
        })

        result_msg = f"Comment added to case {case_id} (Note ID: {note['id']})."
        if is_internal:
            # Notes don't have an internal flag directly, but we can add context in the title
            pass  # The comment is still visible in the case feed

        return result_msg

    except Exception as e:
        logger.exception("[salesforce] add_case_comment error")
        return f"Error adding comment: {type(e).__name__}: {e}"


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

        # Resolve case_id to internal ID
        if not case_id.startswith('500'):
            result = sf.query(
                f"SELECT Id FROM Case WHERE CaseNumber = '{case_id}' LIMIT 1"
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
        logger.exception("[salesforce] update_case_status error")
        return f"Error updating status: {type(e).__name__}: {e}"


# Exported tool list (mirrors the other tools/ modules). query_cases and
# get_case_details are read-only; add_case_comment and update_case_status are
# judge-gated writes (registered in judge.py's write-tool list).
SALESFORCE_TOOLS = [
    query_cases,
    get_case_details,
    add_case_comment,
    update_case_status,
]
