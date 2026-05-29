"""FedRAMP documentation and code review tools for the virtualdojo-inc/Fedramp repo."""

import base64
import re

import httpx
from langchain_core.tools import tool

from tools.github import _github_token

FEDRAMP_REPO = "virtualdojo-inc/Fedramp"
AUTHORIZED_EDITORS = {"devin@virtualdojo.com"}

# Stores pending file uploads for the FileConsentCard flow
# Key: conversation_id, Value: {file_path, content, summary}
_pending_file_uploads: dict[str, dict] = {}

# Card render data for app.py to pick up (FileConsentCard)
# Key: conversation_id, Value: {file_path, content, summary, user_email}
_pending_fedramp_cards: dict[str, dict] = {}

# Stores OneDrive content URLs after successful upload
# Key: conversation_id, Value: {file_path, content_url}
_uploaded_files: dict[str, dict] = {}


def _auth_headers() -> dict:
    return {
        "Authorization": f"Bearer {_github_token()}",
        "Accept": "application/vnd.github+json",
    }


def _check_editor(user_email: str) -> str | None:
    """Return error message if user is not an authorized editor, else None."""
    if user_email.lower() not in AUTHORIZED_EDITORS:
        return "You are not authorized to edit FedRAMP documents."
    return None


@tool
def fedramp_read_document(file_path: str) -> str:
    """Read a file from the virtualdojo-inc/Fedramp repository.

    Args:
        file_path: Path to the file within the repo (e.g. "policies/AC-Policy.md").
    """
    resp = httpx.get(
        f"https://api.github.com/repos/{FEDRAMP_REPO}/contents/{file_path}",
        headers=_auth_headers(),
        timeout=30,
    )
    if resp.status_code == 404:
        return f"File not found: {file_path} in {FEDRAMP_REPO}. Check the path and try fedramp_list_documents to browse."
    resp.raise_for_status()
    data = resp.json()

    content = base64.b64decode(data["content"]).decode("utf-8")
    if len(content) > 10000:
        content = content[:10000] + "\n\n... [truncated — file is very large, showing first 10,000 characters]"
    return content


@tool
def fedramp_list_documents(path: str = "") -> str:
    """List files and folders in the virtualdojo-inc/Fedramp repository.

    Args:
        path: Directory path within the repo (empty string for root).
    """
    resp = httpx.get(
        f"https://api.github.com/repos/{FEDRAMP_REPO}/contents/{path}",
        headers=_auth_headers(),
        timeout=30,
    )
    if resp.status_code == 404:
        return f"Path not found: '{path}' in {FEDRAMP_REPO}."
    resp.raise_for_status()
    items = resp.json()

    if not isinstance(items, list):
        return f"'{path}' is a file, not a directory. Use fedramp_read_document to read it."

    lines = [f"Contents of /{path or '(root)'}:\n"]
    for item in sorted(items, key=lambda x: (x["type"] != "dir", x["name"])):
        kind = "dir" if item["type"] == "dir" else "file"
        size = f"  ({item['size']} bytes)" if item["type"] == "file" else ""
        lines.append(f"  [{kind}] {item['name']}{size}")

    return "\n".join(lines)


@tool
def fedramp_search_documents(query: str) -> str:
    """Search the FedRAMP repo for content matching a query.

    Args:
        query: Search term or phrase to find in the repo.
    """
    resp = httpx.get(
        "https://api.github.com/search/code",
        params={"q": f"{query} repo:{FEDRAMP_REPO}"},
        headers=_auth_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    items = data.get("items", [])
    if not items:
        return f"No results found for '{query}' in {FEDRAMP_REPO}."

    lines = [f"Found {data['total_count']} result(s) for '{query}' (showing up to 10):\n"]
    for item in items[:10]:
        path = item["path"]
        name = item["name"]
        # Text matches may be available
        snippets = []
        for match in item.get("text_matches", []):
            fragment = match.get("fragment", "")
            if fragment:
                snippets.append(fragment.strip()[:120])
        snippet_str = f"\n    > {snippets[0]}..." if snippets else ""
        lines.append(f"  {path}{snippet_str}")

    return "\n".join(lines)


@tool
def fedramp_propose_edit(
    file_path: str,
    proposed_content: str,
    summary: str,
    conversation_id: str,
    user_email: str,
) -> str:
    """Propose an edit to a FedRAMP document. Only authorized editors can propose edits.

    Stores the proposed content for the FileConsentCard flow in Teams.

    Args:
        file_path: Path to the file within the repo.
        proposed_content: The full proposed file content.
        summary: A short summary of what changed and why.
        conversation_id: The current conversation ID (provided automatically).
        user_email: The user's email address (provided automatically).
    """
    auth_err = _check_editor(user_email)
    if auth_err:
        return auth_err

    _pending_file_uploads[conversation_id] = {
        "file_path": file_path,
        "content": proposed_content,
        "summary": summary,
    }

    _pending_fedramp_cards[conversation_id] = {
        "card_type": "fedramp_file_consent",
        "file_path": file_path,
        "summary": summary,
        "user_email": user_email,
        "conversation_id": conversation_id,
    }

    return (
        f"Edit proposed for **{file_path}**.\n\n"
        f"**Summary:** {summary}\n\n"
        "A file consent card will be sent in Teams. You can review the document "
        "and approve or reject the upload. Once approved, use fedramp_commit_document "
        "to commit the changes to the repo."
    )


@tool
def fedramp_commit_document(
    file_path: str,
    commit_message: str,
    conversation_id: str,
    user_email: str,
) -> str:
    """Commit a proposed document edit to the FedRAMP repo. Only authorized editors can commit.

    Uses content from the FileConsentCard flow (OneDrive URL) or the original proposed content.

    Args:
        file_path: Path to the file within the repo.
        commit_message: Git commit message describing the change.
        conversation_id: The current conversation ID (provided automatically).
        user_email: The user's email address (provided automatically).
    """
    auth_err = _check_editor(user_email)
    if auth_err:
        return auth_err

    # Determine content to commit
    content = None

    # Priority 1: uploaded/edited file from OneDrive (file consent flow)
    uploaded = _uploaded_files.get(conversation_id)
    if uploaded and uploaded.get("content_url"):
        resp = httpx.get(uploaded["content_url"], timeout=30)
        resp.raise_for_status()
        content = resp.text

    # Priority 2: original proposed content (user approved without editing)
    if content is None:
        pending = _pending_file_uploads.get(conversation_id)
        if pending:
            content = pending["content"]

    if content is None:
        return (
            "No pending content found for this conversation. "
            "Use fedramp_propose_edit first to draft a document."
        )

    # Get current file SHA for updates (not needed for new files)
    sha = None
    existing = httpx.get(
        f"https://api.github.com/repos/{FEDRAMP_REPO}/contents/{file_path}",
        headers=_auth_headers(),
        timeout=30,
    )
    if existing.status_code == 200:
        sha = existing.json().get("sha")

    # Commit via GitHub contents API
    payload: dict = {
        "message": commit_message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
    }
    if sha:
        payload["sha"] = sha

    resp = httpx.put(
        f"https://api.github.com/repos/{FEDRAMP_REPO}/contents/{file_path}",
        headers=_auth_headers(),
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()

    # Clean up pending state
    _pending_file_uploads.pop(conversation_id, None)
    _uploaded_files.pop(conversation_id, None)
    _pending_fedramp_cards.pop(conversation_id, None)

    commit_sha = result.get("commit", {}).get("sha", "unknown")
    return (
        f"Committed successfully to {FEDRAMP_REPO}.\n"
        f"File: {file_path}\n"
        f"Commit SHA: {commit_sha}\n"
        f"Message: {commit_message}"
    )


@tool
def fedramp_review_code(repo: str, file_path: str) -> str:
    """Review a source file from a virtualdojo-inc repo for FedRAMP security issues.

    Checks against NIST 800-53 controls: SC-7, SC-12, CM-6, SC-18, AC-8.

    Args:
        repo: Repository in 'owner/repo' format (must be a virtualdojo-inc repo).
        file_path: Path to the source file to review.
    """
    resp = httpx.get(
        f"https://api.github.com/repos/{repo}/contents/{file_path}",
        headers=_auth_headers(),
        timeout=30,
    )
    if resp.status_code == 404:
        return f"File not found: {file_path} in {repo}."
    resp.raise_for_status()
    data = resp.json()

    content = base64.b64decode(data["content"]).decode("utf-8")
    lines = content.splitlines()

    findings: list[str] = []

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        # SC-7: CORS wildcard
        if "allow_origins" in line and "'*'" in line or '"*"' in line:
            if "allow_origins" in line:
                findings.append(
                    f"  [HIGH] SC-7 (Boundary Protection) line {i}: "
                    f"CORS wildcard allow_origins=['*'] — restrict to specific origins."
                )
        if "Access-Control-Allow-Origin" in line and "*" in line:
            findings.append(
                f"  [HIGH] SC-7 (Boundary Protection) line {i}: "
                f"Access-Control-Allow-Origin: * header — restrict to specific origins."
            )

        # SC-12: Hardcoded credentials
        cred_patterns = [
            r'password\s*=\s*["\']',
            r'secret\s*=\s*["\']',
            r'(?<!\w)key\s*=\s*["\'][A-Za-z0-9]',
        ]
        for pattern in cred_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                findings.append(
                    f"  [HIGH] SC-12 (Cryptographic Key Management) line {i}: "
                    f"Possible hardcoded credential — use environment variables or a secrets manager."
                )
                break

        # CM-6: Information leakage via print/bare except/str(exc)
        if re.match(r'\s*print\s*\(', line):
            findings.append(
                f"  [MEDIUM] CM-6 (Configuration Settings) line {i}: "
                f"print() statement — use structured logging; avoid leaking info in production."
            )
        if re.match(r'\s*except\s*:', stripped):
            findings.append(
                f"  [MEDIUM] CM-6 (Configuration Settings) line {i}: "
                f"Bare except: clause — catch specific exceptions to avoid masking errors."
            )
        if "str(exc)" in line or "str(e)" in line:
            # Check if it's in an except block context (heuristic)
            if any(
                "except" in lines[max(0, j)].strip()
                for j in range(max(0, i - 5), i)
            ):
                findings.append(
                    f"  [MEDIUM] CM-6 (Configuration Settings) line {i}: "
                    f"Exception details exposed via str() — sanitize error messages for end users."
                )

        # SC-18: Active content / XSS risks
        if "v-html" in line:
            findings.append(
                f"  [HIGH] SC-18 (Mobile Code) line {i}: "
                f"v-html directive without sanitization — risk of XSS. Use DOMPurify or equivalent."
            )
        if "innerHTML" in line and "dangerouslySetInnerHTML" not in line:
            findings.append(
                f"  [HIGH] SC-18 (Mobile Code) line {i}: "
                f"innerHTML assignment — risk of XSS. Sanitize content before insertion."
            )
        if "dangerouslySetInnerHTML" in line:
            findings.append(
                f"  [HIGH] SC-18 (Mobile Code) line {i}: "
                f"dangerouslySetInnerHTML — risk of XSS. Ensure content is sanitized."
            )

        # AC-8: Login without system use notification
        if re.search(r'(login|sign.?in|authenticate)', line, re.IGNORECASE):
            # Check nearby lines for banner/notification
            context = "\n".join(lines[max(0, i - 10):min(len(lines), i + 10)])
            if not re.search(r'(banner|notice|notification|system.use|warning)', context, re.IGNORECASE):
                findings.append(
                    f"  [LOW] AC-8 (System Use Notification) line {i}: "
                    f"Login-related code without apparent system use notification/banner."
                )

    if not findings:
        return (
            f"FedRAMP Code Review: {file_path}\n"
            f"Repository: {repo}\n\n"
            "No FedRAMP security issues found. Clean report."
        )

    # Deduplicate findings (AC-8 can fire multiple times for same context)
    seen = set()
    unique_findings = []
    for f in findings:
        if f not in seen:
            seen.add(f)
            unique_findings.append(f)

    return (
        f"FedRAMP Code Review: {file_path}\n"
        f"Repository: {repo}\n"
        f"Issues found: {len(unique_findings)}\n\n"
        + "\n".join(unique_findings)
    )


@tool
def fedramp_discard_draft(
    file_path: str, conversation_id: str, user_email: str
) -> str:
    """Discard a pending FedRAMP document draft and clean up state.

    Args:
        file_path: Path to the file that was being edited.
        conversation_id: The current conversation ID (provided automatically).
        user_email: The user's email address (provided automatically).
    """
    auth_err = _check_editor(user_email)
    if auth_err:
        return auth_err

    removed = []
    if _pending_file_uploads.pop(conversation_id, None):
        removed.append("pending upload")
    if _uploaded_files.pop(conversation_id, None):
        removed.append("uploaded file reference")
    if _pending_fedramp_cards.pop(conversation_id, None):
        removed.append("pending card")

    if not removed:
        return f"No pending draft found for {file_path} in this conversation."

    return f"Discarded draft for {file_path}. Cleaned up: {', '.join(removed)}."


# All FedRAMP doc tools for easy import
FEDRAMP_DOC_TOOLS = [
    fedramp_read_document,
    fedramp_list_documents,
    fedramp_search_documents,
    fedramp_propose_edit,
    fedramp_commit_document,
    fedramp_review_code,
    fedramp_discard_draft,
]
