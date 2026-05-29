"""OSCAL lifecycle tools for FedRAMP compliance automation.

Provides 9 tools for generating, validating, migrating, and rendering
OSCAL 1.0.4 documents (SSP, POA&M, Assessment Results) aligned to the
FedRAMP Moderate baseline.
"""

import base64
import json
import os
import re
import uuid
from datetime import datetime, timezone

import httpx
from langchain_core.tools import tool

from tools.github import _github_token

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEDRAMP_REPO = "virtualdojo-inc/Fedramp"
OSCAL_VERSION = "1.0.4"
FEDRAMP_SYSTEM_ID = "FR2615441197"
FEDRAMP_SYSTEM_NAME = "VirtualDojo AI CRM"
NIST_CATALOG_URL = (
    "https://raw.githubusercontent.com/usnistgov/oscal-content/main/"
    "nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json"
)
FEDRAMP_MODERATE_PROFILE_URL = (
    "https://raw.githubusercontent.com/GSA/fedramp-automation/master/"
    "dist/content/rev5/baselines/json/FedRAMP_rev5_MODERATE-baseline_profile.json"
)
DATA_DIR = os.environ.get("SAMURAI_DATA_DIR", "/data")
AUTHORIZED_EDITORS = {"devin@virtualdojo.com"}

# ---------------------------------------------------------------------------
# Shared pending-upload state (mirrors fedramp_docs pattern)
# ---------------------------------------------------------------------------

_pending_file_uploads: dict[str, dict] = {}
_pending_fedramp_cards: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth_headers() -> dict:
    return {
        "Authorization": f"Bearer {_github_token()}",
        "Accept": "application/vnd.github+json",
    }


def _check_editor(user_email: str) -> str | None:
    """Return error message if user is not an authorized editor, else None."""
    if user_email.lower() not in AUTHORIZED_EDITORS:
        return "You are not authorized to perform this action. Only authorized FedRAMP editors may proceed."
    return None


def _get_nist_catalog() -> dict:
    """Fetch the NIST 800-53 Rev 5 catalog JSON, caching on disk."""
    cache_path = os.path.join(DATA_DIR, "oscal_nist_catalog.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    resp = httpx.get(NIST_CATALOG_URL, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    data = resp.json()
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def _get_fedramp_profile() -> dict:
    """Fetch the FedRAMP Moderate baseline profile JSON, caching on disk."""
    cache_path = os.path.join(DATA_DIR, "oscal_fedramp_moderate_profile.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    resp = httpx.get(FEDRAMP_MODERATE_PROFILE_URL, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    data = resp.json()
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def _read_github_file(file_path: str) -> str | None:
    """Read a file from the FedRAMP repo. Returns content string or None."""
    resp = httpx.get(
        f"https://api.github.com/repos/{FEDRAMP_REPO}/contents/{file_path}",
        headers=_auth_headers(),
        timeout=30,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()
    return base64.b64decode(data["content"]).decode("utf-8")


def _read_github_json(file_path: str) -> dict | None:
    """Read a JSON file from the FedRAMP repo. Returns parsed dict or None."""
    raw = _read_github_file(file_path)
    if raw is None:
        return None
    return json.loads(raw)


def _commit_file(file_path: str, content: str, commit_message: str) -> str:
    """Commit (create or update) a file in the FedRAMP repo via the GitHub Contents API."""
    # Check if the file already exists to get its SHA
    sha = None
    existing = httpx.get(
        f"https://api.github.com/repos/{FEDRAMP_REPO}/contents/{file_path}",
        headers=_auth_headers(),
        timeout=30,
    )
    if existing.status_code == 200:
        sha = existing.json().get("sha")

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
    return result.get("commit", {}).get("sha", "unknown")


def _extract_control_ids_from_profile(profile: dict) -> list[str]:
    """Extract the list of control IDs required by a FedRAMP baseline profile."""
    control_ids: list[str] = []
    imports = profile.get("profile", {}).get("imports", [])
    for imp in imports:
        for include in imp.get("include-controls", []):
            for ctrl in include.get("with-ids", []):
                control_ids.append(ctrl)
    # Fallback: try alternate structure
    if not control_ids:
        imports = profile.get("profile", {}).get("imports", [])
        for imp in imports:
            include_all = imp.get("include-all", {})
            if include_all:
                # All controls included; we cannot enumerate easily, return empty
                break
            for selected in imp.get("include-controls", []):
                cid = selected if isinstance(selected, str) else selected.get("with-id", "")
                if cid:
                    control_ids.append(cid)
    return sorted(set(control_ids))


def _parse_markdown_table(md: str) -> list[dict]:
    """Parse a simple markdown table into a list of dicts keyed by header."""
    lines = [l.strip() for l in md.splitlines() if l.strip()]
    table_lines = [l for l in lines if l.startswith("|")]
    if len(table_lines) < 2:
        return []
    headers = [h.strip() for h in table_lines[0].split("|") if h.strip()]
    rows: list[dict] = []
    for line in table_lines[2:]:  # skip header and separator
        if set(line.replace("|", "").strip()) <= {"-", ":", " "}:
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        row = {}
        for i, h in enumerate(headers):
            row[h] = cells[i] if i < len(cells) else ""
        rows.append(row)
    return rows


def _find_control_in_catalog(catalog: dict, control_id: str) -> dict | None:
    """Search the NIST catalog for a control or enhancement by ID."""
    cid_lower = control_id.lower()

    def _search_controls(controls: list) -> dict | None:
        for ctrl in controls:
            if ctrl.get("id", "").lower() == cid_lower:
                return ctrl
            # Check enhancements (sub-controls)
            for sub in ctrl.get("controls", []):
                if sub.get("id", "").lower() == cid_lower:
                    return sub
                # Nested enhancements
                for subsub in sub.get("controls", []):
                    if subsub.get("id", "").lower() == cid_lower:
                        return subsub
        return None

    groups = catalog.get("catalog", {}).get("groups", [])
    for group in groups:
        result = _search_controls(group.get("controls", []))
        if result:
            return result
    return None


def _extract_prose(parts: list) -> str:
    """Extract prose text from OSCAL parts list."""
    texts: list[str] = []
    for part in parts:
        if part.get("prose"):
            texts.append(part["prose"])
        if part.get("parts"):
            texts.append(_extract_prose(part["parts"]))
    return "\n".join(texts)


# ---------------------------------------------------------------------------
# Tool 1: Generate SSP
# ---------------------------------------------------------------------------


@tool
def oscal_generate_ssp(conversation_id: str, user_email: str) -> str:
    """Generate an OSCAL 1.0.4 System Security Plan (SSP) for VirtualDojo AI CRM.

    Builds the full SSP JSON with system characteristics, components, and
    control implementations from the FedRAMP Moderate baseline. Commits the
    result to oscal/system-security-plan.json in the FedRAMP repo.

    Args:
        conversation_id: The current conversation ID (provided automatically).
        user_email: The user's email address (provided automatically).
    """
    auth_err = _check_editor(user_email)
    if auth_err:
        return auth_err

    # Load FedRAMP Moderate baseline profile
    profile = _get_fedramp_profile()
    control_ids = _extract_control_ids_from_profile(profile)

    # Try to read existing SSP markdown for implementation descriptions
    ssp_md = _read_github_file("FedRAMP-Moderate-SSP.md")
    impl_descriptions: dict[str, str] = {}
    if ssp_md:
        # Parse markdown for control implementation sections
        # Look for patterns like "## AC-2" or "### AC-2 Account Management"
        current_control: str | None = None
        current_text: list[str] = []
        for line in ssp_md.splitlines():
            m = re.match(r"^#{1,4}\s+((?:[A-Z]{2}-\d+)(?:\(\d+\))?)", line)
            if m:
                if current_control and current_text:
                    impl_descriptions[current_control.lower()] = "\n".join(current_text).strip()
                current_control = m.group(1)
                current_text = []
            elif current_control:
                current_text.append(line)
        if current_control and current_text:
            impl_descriptions[current_control.lower()] = "\n".join(current_text).strip()

    # Build implemented-requirements from baseline controls
    implemented_requirements = []
    for cid in control_ids:
        desc = impl_descriptions.get(cid.lower(), "Implementation pending.")
        implemented_requirements.append({
            "uuid": str(uuid.uuid4()),
            "control-id": cid,
            "statements": [
                {
                    "statement-id": f"{cid}_smt",
                    "uuid": str(uuid.uuid4()),
                    "description": desc[:2000] if len(desc) > 2000 else desc,
                }
            ],
        })

    ssp = {
        "system-security-plan": {
            "uuid": str(uuid.uuid4()),
            "metadata": {
                "title": f"{FEDRAMP_SYSTEM_NAME} System Security Plan",
                "last-modified": datetime.now(timezone.utc).isoformat(),
                "version": "1.0",
                "oscal-version": OSCAL_VERSION,
                "roles": [
                    {"id": "isso", "title": "Information System Security Officer"},
                    {"id": "admin", "title": "System Administrator"},
                    {"id": "authorizing-official", "title": "Authorizing Official"},
                ],
                "parties": [
                    {
                        "uuid": str(uuid.uuid4()),
                        "type": "organization",
                        "name": "VirtualDojo Inc.",
                    }
                ],
            },
            "import-profile": {
                "href": "catalogs/FedRAMP_rev5_MODERATE-baseline_profile.json"
            },
            "system-characteristics": {
                "system-ids": [
                    {
                        "identifier-type": "https://fedramp.gov",
                        "id": FEDRAMP_SYSTEM_ID,
                    }
                ],
                "system-name": FEDRAMP_SYSTEM_NAME,
                "description": "VirtualDojo AI CRM - FedRAMP Moderate cloud-based CRM platform",
                "security-sensitivity-level": "moderate",
                "system-information": {
                    "information-types": [
                        {
                            "uuid": str(uuid.uuid4()),
                            "title": "CRM Business Data",
                            "description": "Customer relationship management data including contacts, opportunities, and interactions.",
                            "categorizations": [
                                {
                                    "system": "https://doi.org/10.6028/NIST.SP.800-60v2r1",
                                    "information-type-ids": ["C.3.5.1"],
                                }
                            ],
                            "confidentiality-impact": {"base": "moderate"},
                            "integrity-impact": {"base": "moderate"},
                            "availability-impact": {"base": "moderate"},
                        }
                    ]
                },
                "security-impact-level": {
                    "security-objective-confidentiality": "moderate",
                    "security-objective-integrity": "moderate",
                    "security-objective-availability": "moderate",
                },
                "status": {"state": "operational"},
                "authorization-boundary": {
                    "description": "GCP us-central1, Assured Workloads FedRAMP Moderate boundary"
                },
            },
            "system-implementation": {
                "users": [
                    {
                        "uuid": str(uuid.uuid4()),
                        "title": "ISSO",
                        "role-ids": ["isso"],
                    },
                    {
                        "uuid": str(uuid.uuid4()),
                        "title": "System Administrator",
                        "role-ids": ["admin"],
                    },
                ],
                "components": [
                    {
                        "uuid": str(uuid.uuid4()),
                        "type": "this-system",
                        "title": "VirtualDojo AI CRM",
                        "description": "Primary SaaS CRM application providing AI-powered customer relationship management.",
                        "status": {"state": "operational"},
                    },
                    {
                        "uuid": str(uuid.uuid4()),
                        "type": "software",
                        "title": "Google Cloud Run",
                        "description": "Serverless container runtime in us-central1",
                        "status": {"state": "operational"},
                    },
                    {
                        "uuid": str(uuid.uuid4()),
                        "type": "software",
                        "title": "AlloyDB for PostgreSQL",
                        "description": "Managed PostgreSQL-compatible database",
                        "status": {"state": "operational"},
                    },
                    {
                        "uuid": str(uuid.uuid4()),
                        "type": "software",
                        "title": "Cloud KMS",
                        "description": "Key management with CMEK encryption",
                        "status": {"state": "operational"},
                    },
                    {
                        "uuid": str(uuid.uuid4()),
                        "type": "software",
                        "title": "Microsoft Entra ID",
                        "description": "Identity provider with MFA, Conditional Access (M365 GCC)",
                        "status": {"state": "operational"},
                    },
                ],
            },
            "control-implementation": {
                "description": "FedRAMP Moderate control implementations for VirtualDojo AI CRM",
                "implemented-requirements": implemented_requirements,
            },
        }
    }

    ssp_json = json.dumps(ssp, indent=2)

    # Commit to repo
    commit_sha = _commit_file(
        "oscal/system-security-plan.json",
        ssp_json,
        f"Generate OSCAL SSP ({len(control_ids)} controls, {OSCAL_VERSION})",
    )

    components = ssp["system-security-plan"]["system-implementation"]["components"]
    populated = sum(
        1 for r in implemented_requirements
        if r["statements"][0]["description"] != "Implementation pending."
    )

    return (
        f"OSCAL System Security Plan generated and committed.\n\n"
        f"Commit SHA: {commit_sha}\n"
        f"File: oscal/system-security-plan.json\n"
        f"OSCAL version: {OSCAL_VERSION}\n"
        f"System: {FEDRAMP_SYSTEM_NAME} ({FEDRAMP_SYSTEM_ID})\n"
        f"Controls: {len(control_ids)} total, {populated} with implementation descriptions from SSP markdown\n"
        f"Components: {len(components)}\n"
        f"Users/roles: 2 (ISSO, System Administrator)\n"
        f"Security level: Moderate (C/I/A)"
    )


# ---------------------------------------------------------------------------
# Tool 2: Generate POA&M
# ---------------------------------------------------------------------------


@tool
def oscal_generate_poam(conversation_id: str, user_email: str) -> str:
    """Generate an OSCAL Plan of Action and Milestones (POA&M) for VirtualDojo AI CRM.

    Reads existing POA&M items from VDJ-POAM-Spreadsheet.md in the repo and
    converts them to OSCAL format. Commits to oscal/poam.json.

    Args:
        conversation_id: The current conversation ID (provided automatically).
        user_email: The user's email address (provided automatically).
    """
    auth_err = _check_editor(user_email)
    if auth_err:
        return auth_err

    poam_items: list[dict] = []

    # Try to read the POA&M spreadsheet markdown
    poam_md = _read_github_file("VDJ-POAM-Spreadsheet.md")
    if poam_md:
        rows = _parse_markdown_table(poam_md)
        for row in rows:
            # Try common column names
            control_id = (
                row.get("Control ID", "")
                or row.get("Control", "")
                or row.get("control-id", "")
                or row.get("ID", "")
            )
            description = (
                row.get("Description", "")
                or row.get("Weakness", "")
                or row.get("Finding", "")
                or row.get("Vulnerability", "")
            )
            risk_level = (
                row.get("Risk Level", "")
                or row.get("Risk", "")
                or row.get("Severity", "")
                or "moderate"
            )
            milestone = (
                row.get("Milestone", "")
                or row.get("Planned Milestone", "")
                or row.get("Remediation", "")
            )
            due_date = (
                row.get("Due Date", "")
                or row.get("Completion Date", "")
                or row.get("Scheduled Completion", "")
            )
            status = (
                row.get("Status", "")
                or row.get("State", "")
                or "open"
            )

            if not control_id and not description:
                continue

            item: dict = {
                "uuid": str(uuid.uuid4()),
                "title": f"POA&M Item: {control_id}" if control_id else "POA&M Item",
                "description": description or "Pending description.",
            }

            if control_id:
                item["related-observations"] = []
                item["associated-risks"] = [
                    {
                        "uuid": str(uuid.uuid4()),
                        "title": f"Risk for {control_id}",
                        "description": description,
                        "risk-level": risk_level.lower() if risk_level else "moderate",
                    }
                ]

            if milestone:
                item["milestones"] = [
                    {
                        "uuid": str(uuid.uuid4()),
                        "title": milestone,
                        "schedule": {
                            "task-date": {"date": due_date} if due_date else {},
                        },
                    }
                ]

            if status:
                item["status"] = status.lower()

            poam_items.append(item)

    # If no items were parsed, add a placeholder
    if not poam_items:
        poam_items.append({
            "uuid": str(uuid.uuid4()),
            "title": "Initial POA&M — no open items",
            "description": "No open plan-of-action items identified at this time.",
            "status": "closed",
        })

    poam = {
        "plan-of-action-and-milestones": {
            "uuid": str(uuid.uuid4()),
            "metadata": {
                "title": f"{FEDRAMP_SYSTEM_NAME} Plan of Action and Milestones",
                "last-modified": datetime.now(timezone.utc).isoformat(),
                "version": "1.0",
                "oscal-version": OSCAL_VERSION,
                "roles": [
                    {"id": "isso", "title": "Information System Security Officer"},
                ],
                "parties": [
                    {
                        "uuid": str(uuid.uuid4()),
                        "type": "organization",
                        "name": "VirtualDojo Inc.",
                    }
                ],
            },
            "import-ssp": {"href": "system-security-plan.json"},
            "poam-items": poam_items,
        }
    }

    poam_json = json.dumps(poam, indent=2)
    commit_sha = _commit_file(
        "oscal/poam.json",
        poam_json,
        f"Generate OSCAL POA&M ({len(poam_items)} items)",
    )

    return (
        f"OSCAL POA&M generated and committed.\n\n"
        f"Commit SHA: {commit_sha}\n"
        f"File: oscal/poam.json\n"
        f"OSCAL version: {OSCAL_VERSION}\n"
        f"POA&M items: {len(poam_items)}\n"
        f"Source: {'VDJ-POAM-Spreadsheet.md (parsed)' if poam_md else 'placeholder (no spreadsheet found)'}"
    )


# ---------------------------------------------------------------------------
# Tool 3: Generate Assessment Results
# ---------------------------------------------------------------------------


@tool
def oscal_generate_assessment_results(
    control_family: str,
    project_id: str | None = None,
    conversation_id: str = "",
    user_email: str = "",
) -> str:
    """Run evidence collection for a control family and format as OSCAL Assessment Results.

    Collects evidence by calling fedramp_collect_evidence (if available) and
    maps results to OSCAL observations and findings.

    Args:
        control_family: NIST 800-53 control family (e.g. "AC", "SI", "AU").
        project_id: Optional GCP project ID for evidence collection.
        conversation_id: The current conversation ID (provided automatically).
        user_email: The user's email address (provided automatically).
    """
    observations: list[dict] = []
    findings: list[dict] = []

    # Try to import and run evidence collection
    evidence_items: list[dict] = []
    try:
        from tools.fedramp import fedramp_collect_evidence

        result = fedramp_collect_evidence.invoke(
            {"control_family": control_family, "project_id": project_id or ""}
        )
        # Parse the text result into evidence items
        for line in result.splitlines():
            line = line.strip()
            if not line or line.startswith("Evidence collection"):
                continue
            status = "pass"
            if "[FAIL]" in line or "FAIL" in line:
                status = "fail"
            elif "[WARN]" in line or "WARNING" in line:
                status = "warning"
            evidence_items.append({"description": line, "status": status})
    except (ImportError, Exception) as exc:
        evidence_items.append({
            "description": (
                f"Evidence collection not available for {control_family}: {exc}. "
                "Manual evidence should be gathered and linked."
            ),
            "status": "warning",
        })

    # Build observations and findings from evidence
    now_iso = datetime.now(timezone.utc).isoformat()
    for item in evidence_items:
        obs_uuid = str(uuid.uuid4())
        observations.append({
            "uuid": obs_uuid,
            "title": f"Observation for {control_family}",
            "description": item["description"],
            "methods": ["EXAMINE", "TEST"],
            "collected": now_iso,
        })

        finding_status = "satisfied" if item["status"] == "pass" else "not-satisfied"
        findings.append({
            "uuid": str(uuid.uuid4()),
            "title": f"Finding for {control_family}",
            "description": item["description"],
            "target": {
                "type": "objective-id",
                "target-id": control_family,
                "status": {"state": finding_status},
            },
            "related-observations": [{"observation-uuid": obs_uuid}],
        })

    ar = {
        "assessment-results": {
            "uuid": str(uuid.uuid4()),
            "metadata": {
                "title": f"{FEDRAMP_SYSTEM_NAME} Assessment Results — {control_family}",
                "last-modified": now_iso,
                "version": "1.0",
                "oscal-version": OSCAL_VERSION,
            },
            "import-ap": {"href": "#"},
            "results": [
                {
                    "uuid": str(uuid.uuid4()),
                    "title": f"Assessment of {control_family} controls",
                    "start": now_iso,
                    "observations": observations,
                    "findings": findings,
                }
            ],
        }
    }

    passed = sum(1 for f in findings if f["target"]["status"]["state"] == "satisfied")
    failed = len(findings) - passed

    # Store for optional commit via fedramp_commit_document
    if conversation_id:
        _pending_file_uploads[conversation_id] = {
            "file_path": "oscal/assessment-results.json",
            "content": json.dumps(ar, indent=2),
            "summary": f"Assessment results for {control_family}: {passed} passed, {failed} failed",
        }

    return (
        f"OSCAL Assessment Results generated for control family: {control_family}\n\n"
        f"Observations: {len(observations)}\n"
        f"Findings: {len(findings)} ({passed} satisfied, {failed} not-satisfied)\n\n"
        f"The assessment results are ready but NOT yet committed.\n"
        f"Use fedramp_commit_document with file_path='oscal/assessment-results.json' "
        f"to save them to the repository."
    )


# ---------------------------------------------------------------------------
# Tool 4: Migrate from Markdown
# ---------------------------------------------------------------------------


@tool
def oscal_migrate_from_markdown(
    file_path: str,
    document_type: str,
    conversation_id: str,
    user_email: str,
) -> str:
    """Migrate a markdown FedRAMP document to OSCAL JSON format.

    Reads the specified markdown file from the FedRAMP repo and extracts
    structured content based on the document type.

    Args:
        file_path: Path to the markdown file in the repo (e.g. "policies/AC-Policy.md").
        document_type: One of "ssp", "poam", "component", or "policy".
        conversation_id: The current conversation ID (provided automatically).
        user_email: The user's email address (provided automatically).
    """
    auth_err = _check_editor(user_email)
    if auth_err:
        return auth_err

    valid_types = {"ssp", "poam", "component", "policy"}
    if document_type.lower() not in valid_types:
        return f"Invalid document_type '{document_type}'. Must be one of: {', '.join(sorted(valid_types))}"

    md_content = _read_github_file(file_path)
    if md_content is None:
        return f"File not found: {file_path} in {FEDRAMP_REPO}."

    doc_type = document_type.lower()
    now_iso = datetime.now(timezone.utc).isoformat()
    oscal_doc: dict = {}
    summary_parts: list[str] = []

    if doc_type == "ssp":
        # Extract control implementation descriptions
        controls_found: dict[str, str] = {}
        current_ctrl: str | None = None
        current_lines: list[str] = []
        for line in md_content.splitlines():
            m = re.match(r"^#{1,4}\s+((?:[A-Z]{2}-\d+)(?:\(\d+\))?)", line)
            if m:
                if current_ctrl and current_lines:
                    controls_found[current_ctrl] = "\n".join(current_lines).strip()
                current_ctrl = m.group(1)
                current_lines = []
            elif current_ctrl:
                current_lines.append(line)
        if current_ctrl and current_lines:
            controls_found[current_ctrl] = "\n".join(current_lines).strip()

        impl_reqs = []
        for cid, desc in controls_found.items():
            impl_reqs.append({
                "uuid": str(uuid.uuid4()),
                "control-id": cid.lower(),
                "statements": [
                    {
                        "statement-id": f"{cid.lower()}_smt",
                        "uuid": str(uuid.uuid4()),
                        "description": desc[:2000],
                    }
                ],
            })

        oscal_doc = {
            "system-security-plan": {
                "uuid": str(uuid.uuid4()),
                "metadata": {
                    "title": f"{FEDRAMP_SYSTEM_NAME} SSP (migrated from {file_path})",
                    "last-modified": now_iso,
                    "version": "1.0",
                    "oscal-version": OSCAL_VERSION,
                },
                "control-implementation": {
                    "description": f"Migrated from {file_path}",
                    "implemented-requirements": impl_reqs,
                },
            }
        }
        summary_parts.append(f"Extracted {len(impl_reqs)} control implementations from SSP markdown.")

    elif doc_type == "poam":
        rows = _parse_markdown_table(md_content)
        poam_items: list[dict] = []
        for row in rows:
            control_id = row.get("Control ID", "") or row.get("Control", "") or row.get("ID", "")
            description = row.get("Description", "") or row.get("Finding", "") or row.get("Weakness", "")
            risk = row.get("Risk Level", "") or row.get("Risk", "") or "moderate"
            milestone = row.get("Milestone", "") or row.get("Remediation", "")
            due_date = row.get("Due Date", "") or row.get("Scheduled Completion", "")

            if not control_id and not description:
                continue

            item: dict = {
                "uuid": str(uuid.uuid4()),
                "title": f"POA&M: {control_id}" if control_id else "POA&M Item",
                "description": description or "Pending.",
            }
            if risk:
                item["associated-risks"] = [{
                    "uuid": str(uuid.uuid4()),
                    "title": f"Risk: {control_id}",
                    "risk-level": risk.lower(),
                }]
            if milestone:
                item["milestones"] = [{
                    "uuid": str(uuid.uuid4()),
                    "title": milestone,
                    "schedule": {"task-date": {"date": due_date}} if due_date else {},
                }]
            poam_items.append(item)

        oscal_doc = {
            "plan-of-action-and-milestones": {
                "uuid": str(uuid.uuid4()),
                "metadata": {
                    "title": f"{FEDRAMP_SYSTEM_NAME} POA&M (migrated from {file_path})",
                    "last-modified": now_iso,
                    "version": "1.0",
                    "oscal-version": OSCAL_VERSION,
                },
                "import-ssp": {"href": "system-security-plan.json"},
                "poam-items": poam_items,
            }
        }
        summary_parts.append(f"Extracted {len(poam_items)} POA&M items from markdown table.")

    elif doc_type == "component":
        # Extract component descriptions from sections
        components: list[dict] = []
        current_name: str | None = None
        current_desc_lines: list[str] = []
        for line in md_content.splitlines():
            m = re.match(r"^#{1,3}\s+(.+)", line)
            if m:
                if current_name and current_desc_lines:
                    components.append({
                        "uuid": str(uuid.uuid4()),
                        "type": "software",
                        "title": current_name,
                        "description": "\n".join(current_desc_lines).strip()[:2000],
                        "status": {"state": "operational"},
                    })
                current_name = m.group(1).strip()
                current_desc_lines = []
            elif current_name:
                current_desc_lines.append(line)
        if current_name and current_desc_lines:
            components.append({
                "uuid": str(uuid.uuid4()),
                "type": "software",
                "title": current_name,
                "description": "\n".join(current_desc_lines).strip()[:2000],
                "status": {"state": "operational"},
            })

        oscal_doc = {
            "component-definition": {
                "uuid": str(uuid.uuid4()),
                "metadata": {
                    "title": f"{FEDRAMP_SYSTEM_NAME} Components (migrated from {file_path})",
                    "last-modified": now_iso,
                    "version": "1.0",
                    "oscal-version": OSCAL_VERSION,
                },
                "components": components,
            }
        }
        summary_parts.append(f"Extracted {len(components)} component definitions.")

    elif doc_type == "policy":
        # Extract policy statements and map to control families
        policies: list[dict] = []
        current_family: str | None = None
        current_statements: list[str] = []
        for line in md_content.splitlines():
            # Try to detect control family headers like "## Access Control (AC)"
            m = re.match(r"^#{1,3}\s+.*?\(([A-Z]{2})\)", line)
            if not m:
                # Also try "## AC - Access Control"
                m = re.match(r"^#{1,3}\s+([A-Z]{2})\s*[-—:]", line)
            if m:
                if current_family and current_statements:
                    policies.append({
                        "uuid": str(uuid.uuid4()),
                        "control-family": current_family,
                        "statements": "\n".join(current_statements).strip()[:3000],
                    })
                current_family = m.group(1)
                current_statements = []
            elif current_family:
                current_statements.append(line)
        if current_family and current_statements:
            policies.append({
                "uuid": str(uuid.uuid4()),
                "control-family": current_family,
                "statements": "\n".join(current_statements).strip()[:3000],
            })

        oscal_doc = {
            "policy-document": {
                "uuid": str(uuid.uuid4()),
                "metadata": {
                    "title": f"{FEDRAMP_SYSTEM_NAME} Policy (migrated from {file_path})",
                    "last-modified": now_iso,
                    "version": "1.0",
                    "oscal-version": OSCAL_VERSION,
                },
                "policies": policies,
            }
        }
        summary_parts.append(f"Extracted {len(policies)} policy sections mapped to control families.")

    # Store the generated OSCAL JSON for the commit flow
    target_paths = {
        "ssp": "oscal/system-security-plan.json",
        "poam": "oscal/poam.json",
        "component": "oscal/component-definition.json",
        "policy": "oscal/policy-document.json",
    }
    target_path = target_paths[doc_type]

    _pending_file_uploads[conversation_id] = {
        "file_path": target_path,
        "content": json.dumps(oscal_doc, indent=2),
        "summary": "; ".join(summary_parts),
    }

    _pending_fedramp_cards[conversation_id] = {
        "card_type": "fedramp_file_consent",
        "file_path": target_path,
        "summary": "; ".join(summary_parts),
        "user_email": user_email,
        "conversation_id": conversation_id,
    }

    return (
        f"Markdown migration complete: {file_path} -> OSCAL {doc_type}\n\n"
        f"{'  '.join(summary_parts)}\n\n"
        f"Target file: {target_path}\n\n"
        f"The OSCAL content is ready. Say 'commit it' to save to the repository."
    )


# ---------------------------------------------------------------------------
# Tool 5: Update Control
# ---------------------------------------------------------------------------


@tool
def oscal_update_control(
    control_id: str,
    implementation_description: str,
    conversation_id: str,
    user_email: str,
) -> str:
    """Update the implementation description for a specific control in the SSP.

    Reads the current SSP from the repo, updates the specified control's
    statement, and stores the result as a pending edit.

    Args:
        control_id: The NIST control ID to update (e.g. "ac-2", "si-4").
        implementation_description: New implementation description text.
        conversation_id: The current conversation ID (provided automatically).
        user_email: The user's email address (provided automatically).
    """
    auth_err = _check_editor(user_email)
    if auth_err:
        return auth_err

    ssp = _read_github_json("oscal/system-security-plan.json")
    if ssp is None:
        return (
            "SSP not found at oscal/system-security-plan.json. "
            "Generate the SSP first with oscal_generate_ssp."
        )

    cid = control_id.lower()
    impl_reqs = (
        ssp.get("system-security-plan", {})
        .get("control-implementation", {})
        .get("implemented-requirements", [])
    )

    target_req = None
    for req in impl_reqs:
        if req.get("control-id", "").lower() == cid:
            target_req = req
            break

    if target_req is None:
        return (
            f"Control '{control_id}' not found in the SSP. "
            f"Available controls: {len(impl_reqs)} total. "
            f"Check the control ID format (e.g. 'ac-2', 'si-4')."
        )

    # Capture old description
    old_desc = "N/A"
    if target_req.get("statements"):
        old_desc = target_req["statements"][0].get("description", "N/A")

    # Update the description
    if target_req.get("statements"):
        target_req["statements"][0]["description"] = implementation_description
    else:
        target_req["statements"] = [{
            "statement-id": f"{cid}_smt",
            "uuid": str(uuid.uuid4()),
            "description": implementation_description,
        }]

    # Update metadata timestamp
    ssp["system-security-plan"]["metadata"]["last-modified"] = (
        datetime.now(timezone.utc).isoformat()
    )

    # Store as pending
    _pending_file_uploads[conversation_id] = {
        "file_path": "oscal/system-security-plan.json",
        "content": json.dumps(ssp, indent=2),
        "summary": f"Updated control {control_id} implementation description",
    }

    _pending_fedramp_cards[conversation_id] = {
        "card_type": "fedramp_file_consent",
        "file_path": "oscal/system-security-plan.json",
        "summary": f"Updated control {control_id}",
        "user_email": user_email,
        "conversation_id": conversation_id,
    }

    # Truncate for display
    old_display = old_desc[:200] + "..." if len(old_desc) > 200 else old_desc
    new_display = (
        implementation_description[:200] + "..."
        if len(implementation_description) > 200
        else implementation_description
    )

    return (
        f"Control {control_id.upper()} updated in SSP (pending commit).\n\n"
        f"**Before:**\n{old_display}\n\n"
        f"**After:**\n{new_display}\n\n"
        f"Use fedramp_commit_document with file_path='oscal/system-security-plan.json' to save."
    )


# ---------------------------------------------------------------------------
# Tool 6: Link Evidence
# ---------------------------------------------------------------------------


@tool
def oscal_link_evidence(
    control_id: str,
    evidence_description: str,
    evidence_url: str,
    conversation_id: str,
    user_email: str,
) -> str:
    """Link external evidence to a control in the assessment results.

    Adds a new observation entry to the OSCAL assessment-results document
    linking the evidence URL and description to the specified control.

    Args:
        control_id: The NIST control ID (e.g. "ac-2", "si-4").
        evidence_description: Description of the evidence being linked.
        evidence_url: URL to the evidence artifact (e.g. a screenshot, report, or log).
        conversation_id: The current conversation ID (provided automatically).
        user_email: The user's email address (provided automatically).
    """
    auth_err = _check_editor(user_email)
    if auth_err:
        return auth_err

    ar = _read_github_json("oscal/assessment-results.json")
    if ar is None:
        # Create a new assessment-results document
        ar = {
            "assessment-results": {
                "uuid": str(uuid.uuid4()),
                "metadata": {
                    "title": f"{FEDRAMP_SYSTEM_NAME} Assessment Results",
                    "last-modified": datetime.now(timezone.utc).isoformat(),
                    "version": "1.0",
                    "oscal-version": OSCAL_VERSION,
                },
                "import-ap": {"href": "#"},
                "results": [
                    {
                        "uuid": str(uuid.uuid4()),
                        "title": "Evidence-linked assessment results",
                        "start": datetime.now(timezone.utc).isoformat(),
                        "observations": [],
                        "findings": [],
                    }
                ],
            }
        }

    now_iso = datetime.now(timezone.utc).isoformat()
    obs_uuid = str(uuid.uuid4())

    new_observation = {
        "uuid": obs_uuid,
        "title": f"Evidence for {control_id.upper()}",
        "description": evidence_description,
        "methods": ["EXAMINE"],
        "collected": now_iso,
        "relevant-evidence": [
            {
                "href": evidence_url,
                "description": evidence_description,
            }
        ],
    }

    # Add observation to the first result set
    results = ar.get("assessment-results", {}).get("results", [])
    if results:
        results[0].setdefault("observations", []).append(new_observation)
        # Also add a finding reference
        results[0].setdefault("findings", []).append({
            "uuid": str(uuid.uuid4()),
            "title": f"Evidence linked for {control_id.upper()}",
            "description": f"Evidence artifact linked: {evidence_description}",
            "target": {
                "type": "objective-id",
                "target-id": control_id.lower(),
                "status": {"state": "satisfied"},
            },
            "related-observations": [{"observation-uuid": obs_uuid}],
        })

    # Update metadata timestamp
    ar["assessment-results"]["metadata"]["last-modified"] = now_iso

    # Store as pending
    _pending_file_uploads[conversation_id] = {
        "file_path": "oscal/assessment-results.json",
        "content": json.dumps(ar, indent=2),
        "summary": f"Linked evidence for {control_id}: {evidence_description}",
    }

    _pending_fedramp_cards[conversation_id] = {
        "card_type": "fedramp_file_consent",
        "file_path": "oscal/assessment-results.json",
        "summary": f"Linked evidence for {control_id}",
        "user_email": user_email,
        "conversation_id": conversation_id,
    }

    return (
        f"Evidence linked to control {control_id.upper()} in assessment results (pending commit).\n\n"
        f"Description: {evidence_description}\n"
        f"URL: {evidence_url}\n"
        f"Observation UUID: {obs_uuid}\n\n"
        f"Use fedramp_commit_document with file_path='oscal/assessment-results.json' to save."
    )


# ---------------------------------------------------------------------------
# Tool 7: Validate Package
# ---------------------------------------------------------------------------


@tool
def oscal_validate_package(
    file_path: str = "oscal/system-security-plan.json",
) -> str:
    """Validate an OSCAL JSON document against structural requirements.

    Checks required fields, UUID formats, OSCAL version, and document-specific
    requirements for SSP, POA&M, and assessment results.

    Args:
        file_path: Path to the OSCAL JSON file in the repo (default: oscal/system-security-plan.json).
    """
    doc = _read_github_json(file_path)
    if doc is None:
        return f"File not found: {file_path} in {FEDRAMP_REPO}. Cannot validate."

    checks: list[dict] = []
    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
    )

    def _add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    # Detect document type
    doc_type = "unknown"
    root_key = ""
    if "system-security-plan" in doc:
        doc_type = "SSP"
        root_key = "system-security-plan"
    elif "plan-of-action-and-milestones" in doc:
        doc_type = "POA&M"
        root_key = "plan-of-action-and-milestones"
    elif "assessment-results" in doc:
        doc_type = "Assessment Results"
        root_key = "assessment-results"
    elif "component-definition" in doc:
        doc_type = "Component Definition"
        root_key = "component-definition"

    _add("Document type detected", doc_type != "unknown", doc_type)

    if not root_key:
        _add("Valid root element", False, "No recognized OSCAL root element found.")
        passed_count = sum(1 for c in checks if c["passed"])
        total = len(checks)
        lines = [f"OSCAL Validation: {file_path}\nDocument type: {doc_type}\n"]
        for c in checks:
            status = "PASS" if c["passed"] else "FAIL"
            detail = f" — {c['detail']}" if c["detail"] else ""
            lines.append(f"  [{status}] {c['name']}{detail}")
        lines.append(f"\nResult: {passed_count}/{total} checks passed.")
        return "\n".join(lines)

    root = doc[root_key]

    # UUID check
    doc_uuid = root.get("uuid", "")
    _add("Root UUID present", bool(doc_uuid))
    _add("Root UUID format valid", bool(uuid_pattern.match(doc_uuid)) if doc_uuid else False)

    # Metadata checks
    metadata = root.get("metadata", {})
    _add("Metadata present", bool(metadata))
    _add("Metadata: title", bool(metadata.get("title")))
    _add("Metadata: last-modified", bool(metadata.get("last-modified")))
    _add("Metadata: version", bool(metadata.get("version")))

    oscal_ver = metadata.get("oscal-version", "")
    _add(
        "Metadata: oscal-version matches expected",
        oscal_ver == OSCAL_VERSION,
        f"found '{oscal_ver}', expected '{OSCAL_VERSION}'" if oscal_ver != OSCAL_VERSION else "",
    )

    # Document-specific checks
    if doc_type == "SSP":
        sys_chars = root.get("system-characteristics")
        _add("system-characteristics present", sys_chars is not None)

        sys_impl = root.get("system-implementation")
        _add("system-implementation present", sys_impl is not None)

        ctrl_impl = root.get("control-implementation")
        _add("control-implementation present", ctrl_impl is not None)

        if ctrl_impl:
            impl_reqs = ctrl_impl.get("implemented-requirements", [])
            _add(
                "implemented-requirements populated",
                len(impl_reqs) > 0,
                f"{len(impl_reqs)} controls",
            )

            # Check that all have non-empty statements
            empty_stmts = 0
            for req in impl_reqs:
                stmts = req.get("statements", [])
                if not stmts:
                    empty_stmts += 1
                else:
                    for s in stmts:
                        if not s.get("description", "").strip():
                            empty_stmts += 1
            _add(
                "All implemented-requirements have statements",
                empty_stmts == 0,
                f"{empty_stmts} missing/empty" if empty_stmts > 0 else "",
            )

            # Validate UUIDs within requirements
            bad_uuids = 0
            for req in impl_reqs:
                if not uuid_pattern.match(req.get("uuid", "")):
                    bad_uuids += 1
            _add(
                "Implemented-requirement UUIDs valid",
                bad_uuids == 0,
                f"{bad_uuids} invalid" if bad_uuids > 0 else "",
            )

    elif doc_type == "POA&M":
        poam_items = root.get("poam-items", [])
        _add("poam-items present", len(poam_items) > 0, f"{len(poam_items)} items")
        _add("import-ssp present", "import-ssp" in root)

    elif doc_type == "Assessment Results":
        results = root.get("results", [])
        _add("results present", len(results) > 0, f"{len(results)} result sets")
        if results:
            obs = results[0].get("observations", [])
            finds = results[0].get("findings", [])
            _add("observations present", len(obs) > 0, f"{len(obs)} observations")
            _add("findings present", len(finds) > 0, f"{len(finds)} findings")

    # Summary
    passed_count = sum(1 for c in checks if c["passed"])
    total = len(checks)
    all_passed = passed_count == total

    lines = [
        f"OSCAL Validation: {file_path}",
        f"Document type: {doc_type}",
        f"Overall: {'PASS' if all_passed else 'FAIL'} ({passed_count}/{total} checks passed)\n",
    ]
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        detail = f" -- {c['detail']}" if c["detail"] else ""
        lines.append(f"  [{status}] {c['name']}{detail}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 8: Catalog Lookup
# ---------------------------------------------------------------------------


@tool
def oscal_catalog_lookup(control_id: str) -> str:
    """Look up a NIST 800-53 control by ID from the official catalog.

    Returns the control title, description, parameters, and any enhancements.

    Args:
        control_id: The control ID to look up (e.g. "AC-2", "SI-4", "AU-6(1)").
    """
    catalog = _get_nist_catalog()

    # Normalize control ID for lookup: OSCAL uses lowercase with parens for enhancements
    # e.g. "AC-2(4)" -> "ac-2.4" in some catalogs, or "ac-2(4)" in others
    cid = control_id.strip()
    # Try multiple formats
    candidates = [
        cid,
        cid.lower(),
        cid.upper(),
        # Convert "AC-2(4)" to "ac-2.4" format
        re.sub(r"\((\d+)\)", r".\1", cid.lower()),
    ]

    ctrl = None
    for candidate in candidates:
        ctrl = _find_control_in_catalog(catalog, candidate)
        if ctrl:
            break

    if ctrl is None:
        return (
            f"Control '{control_id}' not found in the NIST 800-53 Rev 5 catalog.\n"
            f"Ensure the format is correct (e.g. 'AC-2', 'SI-4', 'AU-6(1)')."
        )

    # Extract information
    title = ctrl.get("title", "N/A")
    ctrl_id_display = ctrl.get("id", control_id).upper()

    # Extract prose from parts
    parts = ctrl.get("parts", [])
    description = ""
    guidance = ""
    for part in parts:
        if part.get("name") == "statement":
            description = _extract_prose([part])
        elif part.get("name") == "guidance":
            guidance = _extract_prose([part])

    # Parameters
    params = ctrl.get("params", [])
    param_lines: list[str] = []
    for p in params:
        pid = p.get("id", "")
        label = p.get("label", "")
        guidelines = p.get("guidelines", [])
        guideline_text = ""
        if guidelines:
            guideline_text = guidelines[0].get("prose", "")
        select = p.get("select", {})
        choices = select.get("choice", []) if select else []
        choice_text = f" (choices: {', '.join(choices)})" if choices else ""
        param_lines.append(f"  {pid}: {label}{choice_text}")
        if guideline_text:
            param_lines.append(f"    Guideline: {guideline_text}")

    # Enhancements (sub-controls)
    enhancements = ctrl.get("controls", [])
    enh_lines: list[str] = []
    for enh in enhancements[:10]:  # Limit to first 10
        enh_id = enh.get("id", "").upper()
        enh_title = enh.get("title", "")
        enh_lines.append(f"  {enh_id}: {enh_title}")

    # Check FedRAMP baseline for this control
    baseline_info = ""
    try:
        profile = _get_fedramp_profile()
        profile_ctrl_ids = _extract_control_ids_from_profile(profile)
        cid_lower = cid.lower()
        if cid_lower in [c.lower() for c in profile_ctrl_ids]:
            baseline_info = "Yes - included in FedRAMP Moderate baseline"
        else:
            baseline_info = "No - not in FedRAMP Moderate baseline"
    except Exception:
        baseline_info = "Unable to check FedRAMP baseline"

    # Build output
    lines = [
        f"NIST 800-53 Rev 5 Control: {ctrl_id_display}",
        f"Title: {title}",
        f"FedRAMP Moderate: {baseline_info}",
        "",
    ]
    if description:
        lines.append(f"Description:\n{description[:1500]}")
        lines.append("")
    if guidance:
        lines.append(f"Guidance:\n{guidance[:1000]}")
        lines.append("")
    if param_lines:
        lines.append("Parameters:")
        lines.extend(param_lines)
        lines.append("")
    if enh_lines:
        lines.append(f"Enhancements ({len(enhancements)} total):")
        lines.extend(enh_lines)
        if len(enhancements) > 10:
            lines.append(f"  ... and {len(enhancements) - 10} more")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 9: Render PDF
# ---------------------------------------------------------------------------


@tool
def oscal_render_pdf(
    document_type: str,
    conversation_id: str,
    user_email: str,
) -> str:
    """Render an OSCAL document as a branded VirtualDojo PDF.

    Reads the specified OSCAL JSON from the FedRAMP repo and produces a
    formatted PDF with cover page, table of contents, and structured content.

    Args:
        document_type: One of "ssp", "poam", or "assessment-results".
        conversation_id: The current conversation ID (provided automatically).
        user_email: The user's email address (provided automatically).
    """
    auth_err = _check_editor(user_email)
    if auth_err:
        return auth_err

    valid_types = {"ssp", "poam", "assessment-results"}
    dt = document_type.lower()
    if dt not in valid_types:
        return f"Invalid document_type '{document_type}'. Must be one of: {', '.join(sorted(valid_types))}"

    file_map = {
        "ssp": "oscal/system-security-plan.json",
        "poam": "oscal/poam.json",
        "assessment-results": "oscal/assessment-results.json",
    }
    file_path = file_map[dt]
    doc = _read_github_json(file_path)
    if doc is None:
        return f"OSCAL document not found: {file_path}. Generate it first."

    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=25)

    # -- Cover page --
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 28)
    pdf.cell(0, 40, "", ln=True)  # top spacing
    pdf.cell(0, 15, "VirtualDojo AI CRM", ln=True, align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "", ln=True)

    title_map = {
        "ssp": "System Security Plan (SSP)",
        "poam": "Plan of Action & Milestones (POA&M)",
        "assessment-results": "Security Assessment Results",
    }
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, title_map[dt], ln=True, align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, "", ln=True)
    pdf.cell(0, 8, f"FedRAMP System ID: {FEDRAMP_SYSTEM_ID}", ln=True, align="C")
    pdf.cell(0, 8, f"OSCAL Version: {OSCAL_VERSION}", ln=True, align="C")
    pdf.cell(
        0, 8,
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        ln=True, align="C",
    )
    pdf.cell(0, 8, "Classification: FedRAMP Moderate", ln=True, align="C")
    pdf.cell(0, 30, "", ln=True)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6, "VirtualDojo Inc. -- Confidential", ln=True, align="C")

    sections_rendered: list[str] = []

    if dt == "ssp":
        ssp_root = doc.get("system-security-plan", {})
        metadata = ssp_root.get("metadata", {})
        sys_chars = ssp_root.get("system-characteristics", {})
        sys_impl = ssp_root.get("system-implementation", {})
        ctrl_impl = ssp_root.get("control-implementation", {})

        # -- Table of Contents --
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "Table of Contents", ln=True)
        pdf.set_font("Helvetica", "", 11)
        toc_items = [
            "1. Document Information",
            "2. System Characteristics",
            "3. System Implementation",
            "4. Control Implementations",
        ]
        for item in toc_items:
            pdf.cell(0, 8, item, ln=True)
        sections_rendered.append("Table of Contents")

        # -- Section 1: Document Information --
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "1. Document Information", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"Title: {metadata.get('title', 'N/A')}", ln=True)
        pdf.cell(0, 7, f"Version: {metadata.get('version', 'N/A')}", ln=True)
        pdf.cell(0, 7, f"Last Modified: {metadata.get('last-modified', 'N/A')}", ln=True)
        pdf.cell(0, 7, f"OSCAL Version: {metadata.get('oscal-version', 'N/A')}", ln=True)
        sections_rendered.append("Document Information")

        # -- Section 2: System Characteristics --
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "2. System Characteristics", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"System Name: {sys_chars.get('system-name', 'N/A')}", ln=True)
        pdf.multi_cell(0, 7, f"Description: {sys_chars.get('description', 'N/A')}")
        pdf.cell(0, 7, f"Sensitivity Level: {sys_chars.get('security-sensitivity-level', 'N/A')}", ln=True)
        pdf.cell(0, 7, f"Status: {sys_chars.get('status', {}).get('state', 'N/A')}", ln=True)
        impact = sys_chars.get("security-impact-level", {})
        pdf.cell(0, 7, f"Confidentiality: {impact.get('security-objective-confidentiality', 'N/A')}", ln=True)
        pdf.cell(0, 7, f"Integrity: {impact.get('security-objective-integrity', 'N/A')}", ln=True)
        pdf.cell(0, 7, f"Availability: {impact.get('security-objective-availability', 'N/A')}", ln=True)
        boundary = sys_chars.get("authorization-boundary", {})
        pdf.multi_cell(0, 7, f"Auth Boundary: {boundary.get('description', 'N/A')}")
        sections_rendered.append("System Characteristics")

        # -- Section 3: System Implementation --
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "3. System Implementation", ln=True)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Components:", ln=True)
        pdf.set_font("Helvetica", "", 11)
        for comp in sys_impl.get("components", []):
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, f"  {comp.get('title', 'N/A')} ({comp.get('type', '')})", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, f"    {comp.get('description', '')}")
            pdf.cell(0, 3, "", ln=True)
        sections_rendered.append("System Implementation")

        # -- Section 4: Control Implementations --
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "4. Control Implementations", ln=True)
        impl_reqs = ctrl_impl.get("implemented-requirements", [])
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, f"Total controls: {len(impl_reqs)}", ln=True)
        pdf.cell(0, 5, "", ln=True)

        for req in impl_reqs:
            cid = req.get("control-id", "N/A").upper()
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, cid, ln=True)
            pdf.set_font("Helvetica", "", 10)
            for stmt in req.get("statements", []):
                desc = stmt.get("description", "Implementation pending.")
                # Truncate very long descriptions for PDF readability
                if len(desc) > 500:
                    desc = desc[:500] + "..."
                pdf.multi_cell(0, 5, f"  {desc}")
            pdf.cell(0, 3, "", ln=True)
        sections_rendered.append(f"Control Implementations ({len(impl_reqs)} controls)")

    elif dt == "poam":
        poam_root = doc.get("plan-of-action-and-milestones", {})
        poam_items = poam_root.get("poam-items", [])

        # -- Table of Contents --
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "Table of Contents", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, "1. Document Information", ln=True)
        pdf.cell(0, 8, "2. POA&M Summary", ln=True)
        pdf.cell(0, 8, "3. POA&M Items Detail", ln=True)
        sections_rendered.append("Table of Contents")

        # -- Section 1: Document Info --
        pdf.add_page()
        metadata = poam_root.get("metadata", {})
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "1. Document Information", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"Title: {metadata.get('title', 'N/A')}", ln=True)
        pdf.cell(0, 7, f"Version: {metadata.get('version', 'N/A')}", ln=True)
        pdf.cell(0, 7, f"Last Modified: {metadata.get('last-modified', 'N/A')}", ln=True)
        sections_rendered.append("Document Information")

        # -- Section 2: Summary Table --
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "2. POA&M Summary", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, f"Total items: {len(poam_items)}", ln=True)
        pdf.cell(0, 5, "", ln=True)

        # Table header
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(60, 7, "Title", border=1)
        pdf.cell(30, 7, "Status", border=1)
        pdf.cell(40, 7, "Risk Level", border=1)
        pdf.cell(50, 7, "Milestone", border=1)
        pdf.cell(0, 7, "", ln=True)

        pdf.set_font("Helvetica", "", 9)
        for item in poam_items:
            title = item.get("title", "N/A")[:30]
            status = item.get("status", "open")
            risk = "N/A"
            risks = item.get("associated-risks", [])
            if risks:
                risk = risks[0].get("risk-level", "N/A")
            ms = ""
            milestones = item.get("milestones", [])
            if milestones:
                ms = milestones[0].get("title", "")[:25]
            pdf.cell(60, 6, title, border=1)
            pdf.cell(30, 6, status, border=1)
            pdf.cell(40, 6, risk, border=1)
            pdf.cell(50, 6, ms, border=1)
            pdf.cell(0, 6, "", ln=True)
        sections_rendered.append(f"POA&M Summary ({len(poam_items)} items)")

        # -- Section 3: Detail --
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "3. POA&M Items Detail", ln=True)
        for item in poam_items:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, item.get("title", "N/A"), ln=True)
            pdf.set_font("Helvetica", "", 10)
            desc = item.get("description", "")
            if len(desc) > 400:
                desc = desc[:400] + "..."
            pdf.multi_cell(0, 5, f"  {desc}")
            pdf.cell(0, 3, "", ln=True)
        sections_rendered.append("POA&M Items Detail")

    elif dt == "assessment-results":
        ar_root = doc.get("assessment-results", {})
        results = ar_root.get("results", [])

        # -- Table of Contents --
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "Table of Contents", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, "1. Document Information", ln=True)
        pdf.cell(0, 8, "2. Assessment Findings", ln=True)
        sections_rendered.append("Table of Contents")

        # -- Section 1 --
        pdf.add_page()
        metadata = ar_root.get("metadata", {})
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "1. Document Information", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"Title: {metadata.get('title', 'N/A')}", ln=True)
        pdf.cell(0, 7, f"Version: {metadata.get('version', 'N/A')}", ln=True)
        pdf.cell(0, 7, f"Last Modified: {metadata.get('last-modified', 'N/A')}", ln=True)
        sections_rendered.append("Document Information")

        # -- Section 2: Findings by control family --
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "2. Assessment Findings", ln=True)

        for result_set in results:
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 9, result_set.get("title", "Assessment Result"), ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, f"Start: {result_set.get('start', 'N/A')}", ln=True)
            pdf.cell(0, 4, "", ln=True)

            # Group findings by control family
            findings_by_family: dict[str, list] = {}
            for finding in result_set.get("findings", []):
                target_id = finding.get("target", {}).get("target-id", "unknown")
                family = target_id.split("-")[0].upper() if "-" in target_id else target_id.upper()
                findings_by_family.setdefault(family, []).append(finding)

            for family in sorted(findings_by_family.keys()):
                family_findings = findings_by_family[family]
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 7, f"Control Family: {family}", ln=True)
                pdf.set_font("Helvetica", "", 9)
                for f in family_findings:
                    status = f.get("target", {}).get("status", {}).get("state", "unknown")
                    status_marker = "[PASS]" if status == "satisfied" else "[FAIL]"
                    desc = f.get("description", "")
                    if len(desc) > 200:
                        desc = desc[:200] + "..."
                    pdf.multi_cell(0, 5, f"  {status_marker} {desc}")
                pdf.cell(0, 3, "", ln=True)

        total_findings = sum(len(r.get("findings", [])) for r in results)
        sections_rendered.append(f"Assessment Findings ({total_findings} findings)")

    # Generate PDF bytes
    pdf_bytes = pdf.output()
    page_count = pdf.pages_count

    # Encode as base64 for the pending upload flow
    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")

    file_name = f"VirtualDojo_{dt.upper().replace('-', '_')}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.pdf"

    _pending_file_uploads[conversation_id] = {
        "file_path": file_name,
        "content": pdf_b64,
        "content_type": "application/pdf",
        "is_binary": True,
        "summary": f"OSCAL {title_map[dt]} PDF ({page_count} pages)",
    }

    _pending_fedramp_cards[conversation_id] = {
        "card_type": "fedramp_file_consent",
        "file_path": file_name,
        "summary": f"OSCAL {title_map[dt]} PDF render ({page_count} pages)",
        "user_email": user_email,
        "conversation_id": conversation_id,
    }

    return (
        f"PDF generated for OSCAL {title_map[dt]}.\n\n"
        f"File: {file_name}\n"
        f"Pages: {page_count}\n"
        f"Sections: {', '.join(sections_rendered)}\n\n"
        f"The PDF will be uploaded to Teams via the file consent flow."
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

FEDRAMP_OSCAL_TOOLS = [
    oscal_generate_ssp,
    oscal_generate_poam,
    oscal_generate_assessment_results,
    oscal_migrate_from_markdown,
    oscal_update_control,
    oscal_link_evidence,
    oscal_validate_package,
    oscal_catalog_lookup,
    oscal_render_pdf,
]
