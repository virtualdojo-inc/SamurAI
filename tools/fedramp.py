"""FedRAMP compliance tools for GCP — NIST 800-53 evidence collection and monitoring."""

import os
import time
from datetime import datetime, timezone, timedelta

from langchain_core.tools import tool

FEDRAMP_PROJECT = "virtualdojo-fedramp-prod"
FEDRAMP_REGION = "us-central1"
EVIDENCE_BUCKET = "virtualdojo-fedramp-evidence"
ALLOYDB_CLUSTER = "quotely-prod"
KMS_KEYRING = "virtualdojo-keyring"
CLOUD_RUN_SERVICE = "quotely"

CONTROL_FAMILIES = {
    "AC": "Access Control",
    "AU": "Audit and Accountability",
    "CM": "Configuration Management",
    "CP": "Contingency Planning",
    "IA": "Identification and Authentication",
    "RA": "Risk Assessment",
    "SC": "System and Communications Protection",
    "SI": "System and Information Integrity",
    "SR": "Supply Chain Risk Management",
}

REMEDIATION_SLAS = {
    "CRITICAL": 15,
    "HIGH": 30,
    "MODERATE": 90,
    "LOW": 180,
}

# ---------------------------------------------------------------------------
# Internal helpers for evidence collection per control family
# ---------------------------------------------------------------------------

def _fmt(results: list[tuple[str, str, str]]) -> str:
    """Format a list of (check_name, status, detail) tuples into readable output."""
    lines = []
    for check_name, status, detail in results:
        lines.append(f"[{status}] {check_name} — {detail}")
    return "\n".join(lines)


def _collect_ac(pid: str) -> list[tuple[str, str, str]]:
    """AC — Access Control evidence."""
    from google.cloud import resourcemanager_v3
    from google.iam.v1 import iam_policy_pb2  # noqa: F811

    client = resourcemanager_v3.ProjectsClient()
    request = iam_policy_pb2.GetIamPolicyRequest(resource=f"projects/{pid}")
    policy = client.get_iam_policy(request=request)

    results: list[tuple[str, str, str]] = []
    sa_count = 0
    broad_roles = []
    for binding in policy.bindings:
        for member in binding.members:
            if member.startswith("serviceAccount:"):
                sa_count += 1
            if binding.role in ("roles/owner", "roles/editor"):
                broad_roles.append(f"{member} has {binding.role}")

    results.append((
        "Service account inventory",
        "INFO",
        f"{sa_count} service account bindings found",
    ))
    if broad_roles:
        results.append((
            "Overly broad roles",
            "FAIL",
            f"{len(broad_roles)} bindings with Owner/Editor: {'; '.join(broad_roles[:5])}",
        ))
    else:
        results.append(("Overly broad roles", "PASS", "No Owner/Editor bindings found"))

    return results


def _collect_au(pid: str) -> list[tuple[str, str, str]]:
    """AU — Audit and Accountability evidence."""
    from google.cloud import logging as cloud_logging

    client = cloud_logging.Client(project=pid)
    sinks = list(client.list_sinks())

    results: list[tuple[str, str, str]] = []
    if sinks:
        sink_names = [s.name for s in sinks]
        results.append(("Log sinks configured", "PASS", f"Sinks: {', '.join(sink_names)}"))
    else:
        results.append(("Log sinks configured", "FAIL", "No log sinks found — logs may not be exported"))

    # Check log buckets via v2 config client
    try:
        from google.cloud.logging_v2 import ConfigServiceV2Client

        config_client = ConfigServiceV2Client()
        parent = f"projects/{pid}/locations/-"
        buckets = list(config_client.list_buckets(parent=parent))
        for bucket in buckets:
            bucket_name = bucket.name.split("/")[-1]
            retention = bucket.retention_days
            status = "PASS" if retention >= 400 else "FAIL"
            results.append((
                f"Bucket '{bucket_name}' retention",
                status,
                f"{retention} days (FedRAMP Moderate requires >= 400)",
            ))
    except Exception as exc:
        results.append(("Log bucket retention check", "INFO", f"Could not check buckets: {exc}"))

    return results


def _collect_cm(pid: str) -> list[tuple[str, str, str]]:
    """CM — Configuration Management evidence."""
    from google.cloud import run_v2

    client = run_v2.ServicesClient()
    name = f"projects/{pid}/locations/{FEDRAMP_REGION}/services/{CLOUD_RUN_SERVICE}"

    results: list[tuple[str, str, str]] = []
    try:
        svc = client.get_service(name=name)
        template = svc.template

        # Environment variables (count, not values for security)
        containers = template.containers
        env_count = sum(len(c.env) for c in containers)
        results.append(("Cloud Run env config", "INFO", f"{env_count} env vars configured across {len(containers)} container(s)"))

        # Scaling
        scaling = template.scaling
        min_inst = scaling.min_instance_count if scaling else 0
        max_inst = scaling.max_instance_count if scaling else "default"
        results.append(("Scaling configuration", "INFO", f"min={min_inst}, max={max_inst}"))

        # Revision
        revision = svc.latest_ready_revision.split("/")[-1] if svc.latest_ready_revision else "unknown"
        results.append(("Active revision", "INFO", f"{revision}"))
    except Exception as exc:
        results.append(("Cloud Run service config", "FAIL", f"Could not read service: {exc}"))

    return results


def _collect_cp(pid: str) -> list[tuple[str, str, str]]:
    """CP — Contingency Planning evidence."""
    results: list[tuple[str, str, str]] = []
    try:
        from google.cloud.alloydb_v1 import AlloyDBAdminClient

        client = AlloyDBAdminClient()
        cluster_name = f"projects/{pid}/locations/{FEDRAMP_REGION}/clusters/{ALLOYDB_CLUSTER}"
        cluster = client.get_cluster(name=cluster_name)

        backup_config = cluster.automated_backup_policy
        if backup_config and backup_config.enabled:
            results.append(("AlloyDB automated backups", "PASS", "Automated backups enabled"))
        else:
            results.append(("AlloyDB automated backups", "FAIL", "Automated backups not enabled"))

        results.append(("AlloyDB cluster state", "INFO", f"Cluster state: {cluster.state.name}"))
    except ImportError:
        results.append(("AlloyDB backup check", "INFO", "google-cloud-alloydb not installed — skipping backup check"))
    except Exception as exc:
        results.append(("AlloyDB backup check", "FAIL", f"Could not read cluster: {exc}"))

    return results


def _collect_ia(pid: str) -> list[tuple[str, str, str]]:
    """IA — Identification and Authentication evidence."""
    from google.cloud import resourcemanager_v3
    from google.iam.v1 import iam_policy_pb2

    client = resourcemanager_v3.ProjectsClient()
    request = iam_policy_pb2.GetIamPolicyRequest(resource=f"projects/{pid}")
    policy = client.get_iam_policy(request=request)

    results: list[tuple[str, str, str]] = []
    sa_key_risk = []
    for binding in policy.bindings:
        for member in binding.members:
            # Flag user-managed service account keys (not default compute SA)
            if member.startswith("serviceAccount:") and "iam.gserviceaccount.com" in member:
                sa_key_risk.append(member)

    if sa_key_risk:
        results.append((
            "Service accounts with key risk",
            "INFO",
            f"{len(sa_key_risk)} custom SA bindings found — verify no long-lived keys exist",
        ))
    else:
        results.append(("Service accounts", "PASS", "No custom service account bindings detected"))

    # Check for conditional (MFA) bindings
    mfa_bindings = [b for b in policy.bindings if b.condition and "request.auth" in str(b.condition)]
    if mfa_bindings:
        results.append(("MFA-conditional bindings", "PASS", f"{len(mfa_bindings)} conditional IAM bindings found"))
    else:
        results.append(("MFA-conditional bindings", "INFO", "No MFA-conditional IAM bindings detected — verify MFA via Cloud Identity"))

    return results


def _collect_ra(pid: str) -> list[tuple[str, str, str]]:
    """RA — Risk Assessment evidence."""
    results: list[tuple[str, str, str]] = []
    try:
        from google.cloud import securitycenter_v1

        client = securitycenter_v1.SecurityCenterClient()
        parent = f"projects/{pid}/sources/-"
        findings = list(client.list_findings(
            request={"parent": parent, "filter": 'state="ACTIVE"', "page_size": 10},
        ))
        count = len(findings)
        results.append(("SCC active findings", "INFO" if count == 0 else "FAIL", f"{count} active findings in Security Command Center"))
    except Exception as exc:
        results.append(("SCC findings check", "INFO", f"Could not query SCC: {exc}"))

    return results


def _collect_sc(pid: str) -> list[tuple[str, str, str]]:
    """SC — System and Communications Protection evidence."""
    from google.cloud import kms_v1

    results: list[tuple[str, str, str]] = []

    # Check KMS keys
    try:
        kms_client = kms_v1.KeyManagementServiceClient()
        parent = kms_client.key_ring_path(pid, FEDRAMP_REGION, KMS_KEYRING)
        keys = list(kms_client.list_crypto_keys(request={"parent": parent}))
        results.append(("KMS keys in keyring", "INFO", f"{len(keys)} crypto keys found in {KMS_KEYRING}"))
        for key in keys:
            key_name = key.name.split("/")[-1]
            rotation = key.rotation_period
            if rotation and rotation.seconds > 0:
                days = rotation.seconds // 86400
                results.append((f"Key '{key_name}' rotation", "PASS", f"Rotation every {days} days"))
            else:
                results.append((f"Key '{key_name}' rotation", "FAIL", "No rotation period configured"))
    except Exception as exc:
        results.append(("KMS keyring check", "FAIL", f"Could not list keys: {exc}"))

    # Check Cloud Run ingress settings
    try:
        from google.cloud import run_v2

        run_client = run_v2.ServicesClient()
        svc_name = f"projects/{pid}/locations/{FEDRAMP_REGION}/services/{CLOUD_RUN_SERVICE}"
        svc = run_client.get_service(name=svc_name)
        ingress = svc.ingress.name if svc.ingress else "UNKNOWN"
        status = "PASS" if "INTERNAL" in ingress else "INFO"
        results.append(("Cloud Run ingress restriction", status, f"Ingress setting: {ingress}"))
    except Exception as exc:
        results.append(("Cloud Run ingress check", "INFO", f"Could not check ingress: {exc}"))

    return results


def _collect_si(pid: str) -> list[tuple[str, str, str]]:
    """SI — System and Information Integrity evidence."""
    from google.cloud import monitoring_v3

    results: list[tuple[str, str, str]] = []
    try:
        client = monitoring_v3.AlertPolicyServiceClient()
        parent = f"projects/{pid}"
        policies = list(client.list_alert_policies(name=parent))
        enabled = [p for p in policies if p.enabled.value]
        results.append((
            "Monitoring alert policies",
            "PASS" if enabled else "FAIL",
            f"{len(enabled)} enabled alert policies ({len(policies)} total)",
        ))
    except Exception as exc:
        results.append(("Alert policies check", "FAIL", f"Could not query alert policies: {exc}"))

    return results


def _collect_sr(pid: str) -> list[tuple[str, str, str]]:
    """SR — Supply Chain Risk Management evidence."""
    return [
        (
            "SBOM / dependency tracking",
            "INFO",
            "Supply chain risk is tracked via GitHub Dependabot alerts and SBOM generation in CI/CD. "
            "Use fedramp_check_dependabot_alerts to view current open vulnerabilities.",
        ),
    ]


_FAMILY_DISPATCH = {
    "AC": _collect_ac,
    "AU": _collect_au,
    "CM": _collect_cm,
    "CP": _collect_cp,
    "IA": _collect_ia,
    "RA": _collect_ra,
    "SC": _collect_sc,
    "SI": _collect_si,
    "SR": _collect_sr,
}


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------


@tool
def fedramp_collect_evidence(
    control_family: str, project_id: str | None = None
) -> str:
    """Collect FedRAMP compliance evidence for a NIST 800-53 control family.

    Runs GCP queries relevant to the specified control family and returns
    PASS/FAIL/INFO results for each check.

    Args:
        control_family: NIST control family code (e.g. 'AC', 'AU', 'SC').
        project_id: GCP project ID. Defaults to GCP_PROJECT_ID env var.
    """
    family = control_family.upper().strip()
    if family not in CONTROL_FAMILIES:
        valid = ", ".join(f"{k} ({v})" for k, v in CONTROL_FAMILIES.items())
        return f"Unknown control family '{control_family}'. Valid families: {valid}"

    pid = project_id or os.environ.get("GCP_PROJECT_ID", FEDRAMP_PROJECT)

    try:
        results = _FAMILY_DISPATCH[family](pid)
    except Exception as exc:
        return f"Error collecting evidence for {family} ({CONTROL_FAMILIES[family]}): {exc}"

    header = f"=== {family}: {CONTROL_FAMILIES[family]} === (project: {pid})"
    return header + "\n" + _fmt(results)


@tool
def fedramp_check_scc_findings(
    project_id: str | None = None, severity: str = "HIGH"
) -> str:
    """Check Security Command Center for active findings at or above a severity level.

    Args:
        project_id: GCP project ID. Defaults to GCP_PROJECT_ID env var.
        severity: Minimum severity to report — CRITICAL, HIGH, MEDIUM, or LOW.
    """
    from google.cloud import securitycenter_v1
    from google.api_core.exceptions import PermissionDenied

    pid = project_id or os.environ.get("GCP_PROJECT_ID", FEDRAMP_PROJECT)
    severity_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    sev_upper = severity.upper()
    if sev_upper not in severity_levels:
        return f"Invalid severity '{severity}'. Must be one of: {', '.join(severity_levels)}"

    min_index = severity_levels.index(sev_upper)
    target_severities = severity_levels[min_index:]

    try:
        client = securitycenter_v1.SecurityCenterClient()
        parent = f"projects/{pid}/sources/-"
        sev_filter = " OR ".join(f'severity="{s}"' for s in target_severities)
        filter_str = f'state="ACTIVE" AND ({sev_filter})'

        findings_iter = client.list_findings(
            request={"parent": parent, "filter": filter_str, "page_size": 20},
        )

        lines = [f"SCC Active Findings (>= {sev_upper}) for project {pid}:"]
        count = 0
        for finding_result in findings_iter:
            finding = finding_result.finding
            lines.append(
                f"  [{finding.severity.name}] {finding.category} — "
                f"resource: {finding.resource_name}\n"
                f"    {finding.description[:200] if finding.description else 'No description'}"
            )
            count += 1
            if count >= 20:
                lines.append("  ... (truncated at 20 results)")
                break

        if count == 0:
            lines.append("  No active findings found at this severity level.")

        lines.append(f"\nTotal: {count} finding(s)")
        return "\n".join(lines)

    except PermissionDenied:
        return (
            f"Permission denied accessing SCC for project {pid}. "
            "The service account needs roles/securitycenter.findingsViewer."
        )
    except Exception as exc:
        return f"Error querying Security Command Center: {exc}"


@tool
def fedramp_daily_log_review(
    project_id: str | None = None, hours: int = 24
) -> str:
    """Run FedRAMP daily log review — checks 7 audit log categories.

    Covers admin activity, data access, policy denied, Cloud Run deployments,
    AlloyDB errors, KMS usage, and Secret Manager access.

    Args:
        project_id: GCP project ID. Defaults to GCP_PROJECT_ID env var.
        hours: Number of hours to look back (default 24).
    """
    from google.cloud import logging as cloud_logging

    from tools.gcp_logging import _message

    pid = project_id or os.environ.get("GCP_PROJECT_ID", FEDRAMP_PROJECT)
    client = cloud_logging.Client(project=pid)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    ts_filter = f'timestamp>="{cutoff.isoformat()}"'

    queries = [
        ("Admin Activity", f'logName:"cloudaudit.googleapis.com%2Factivity" {ts_filter}'),
        ("Data Access", f'logName:"cloudaudit.googleapis.com%2Fdata_access" {ts_filter}'),
        ("Policy Denied", f'logName:"cloudaudit.googleapis.com%2Fpolicy" {ts_filter}'),
        ("Cloud Run Deployments", f'resource.type="cloud_run_revision" {ts_filter}'),
        ("AlloyDB Auth Failures", f'resource.type="alloydb.googleapis.com/Instance" severity>=ERROR {ts_filter}'),
        ("KMS Key Usage", f'resource.type="cloudkms_cryptokey" {ts_filter}'),
        ("Secret Manager Access", f'resource.type="secretmanager.googleapis.com/Secret" {ts_filter}'),
    ]

    lines = [f"FedRAMP Daily Log Review — past {hours}h (project: {pid})", "=" * 60]
    total_events = 0

    for label, filter_str in queries:
        try:
            entries = list(client.list_entries(
                filter_=filter_str, order_by=cloud_logging.DESCENDING, max_results=50
            ))
            count = len(entries)
            total_events += count
            lines.append(f"\n--- {label}: {count} event(s) ---")

            # Compliance evidence: show the 3 most-recent events with a compact
            # one-line message (no noise stripping — audit signal is preserved).
            for entry in entries[:3]:
                ts = entry.timestamp.isoformat() if entry.timestamp else "?"
                msg = _message(entry.payload).replace("\n", " ").strip() if entry.payload else "(empty)"
                lines.append(f"  [{ts}] {entry.severity}: {msg[:200]}")

            if count > 3:
                lines.append(f"  ... and {count - 3} more")
        except Exception as exc:
            lines.append(f"\n--- {label}: ERROR — {exc} ---")

    lines.append(f"\n{'=' * 60}")
    lines.append(f"Summary: {total_events} total events across {len(queries)} categories in {hours}h window.")
    return "\n".join(lines)


@tool
def fedramp_check_failed_logins(
    project_id: str | None = None, hours: int = 24
) -> str:
    """Check for authentication failure patterns in Cloud Logging.

    Aggregates failed auth events by source IP and flags IPs with > 5 failures.

    Args:
        project_id: GCP project ID. Defaults to GCP_PROJECT_ID env var.
        hours: Number of hours to look back (default 24).
    """
    from google.cloud import logging as cloud_logging

    pid = project_id or os.environ.get("GCP_PROJECT_ID", FEDRAMP_PROJECT)
    client = cloud_logging.Client(project=pid)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    ts_filter = f'timestamp>="{cutoff.isoformat()}"'
    filter_str = (
        f'protoPayload.status.code=7 OR protoPayload.status.code=16 '
        f'OR protoPayload.authenticationInfo.principalEmail="" '
        f'OR severity="ERROR" logName:"cloudaudit.googleapis.com" '
        f'{ts_filter}'
    )

    try:
        entries = list(client.list_entries(filter_=filter_str, max_results=500))
    except Exception as exc:
        return f"Error querying failed logins: {exc}"

    # Aggregate by source IP
    ip_counts: dict[str, int] = {}
    for entry in entries:
        payload = entry.payload if isinstance(entry.payload, dict) else {}
        caller_ip = (
            payload.get("requestMetadata", {}).get("callerIp", "unknown")
            if isinstance(payload, dict)
            else "unknown"
        )
        ip_counts[caller_ip] = ip_counts.get(caller_ip, 0) + 1

    lines = [f"Failed Authentication Report — past {hours}h (project: {pid})"]
    lines.append(f"Total failed auth events: {len(entries)}")
    lines.append("")

    if not ip_counts:
        lines.append("No failed authentication events found.")
        return "\n".join(lines)

    # Sort by count descending
    sorted_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)
    for ip, count in sorted_ips:
        alert = " ** ALERT — potential brute force **" if count > 5 else ""
        lines.append(f"  {ip}: {count} failure(s){alert}")

    alert_count = sum(1 for _, c in sorted_ips if c > 5)
    if alert_count:
        lines.append(f"\n{alert_count} IP(s) flagged with > 5 failures. Investigate for potential unauthorized access.")

    return "\n".join(lines)


@tool
def fedramp_scan_container_vulnerabilities(
    project_id: str | None = None,
) -> str:
    """Scan container images in Artifact Registry for known vulnerabilities.

    Queries Container Analysis (Grafeas) for VULNERABILITY occurrences on
    images in the us-central1-docker.pkg.dev registry.

    Args:
        project_id: GCP project ID. Defaults to GCP_PROJECT_ID env var.
    """
    pid = project_id or os.environ.get("GCP_PROJECT_ID", FEDRAMP_PROJECT)

    lines = [f"Container Vulnerability Scan — project: {pid}"]

    # List docker images in Artifact Registry
    try:
        from google.cloud import artifactregistry_v1

        ar_client = artifactregistry_v1.ArtifactRegistryClient()
        parent = f"projects/{pid}/locations/{FEDRAMP_REGION}/repositories/cloud-run-source-deploy"
        images = list(ar_client.list_docker_images(parent=parent))
        lines.append(f"Docker images in registry: {len(images)}")
        for img in images[:5]:
            lines.append(f"  {img.name.split('/')[-1]} — tags: {', '.join(img.tags) if img.tags else 'untagged'}")
        if len(images) > 5:
            lines.append(f"  ... and {len(images) - 5} more")
    except Exception as exc:
        lines.append(f"Could not list Artifact Registry images: {exc}")

    # Query Container Analysis for vulnerabilities
    try:
        from google.devtools.containeranalysis.v1 import GrafeasClient

        grafeas_client = GrafeasClient()
        parent = f"projects/{pid}"
        occurrences = list(grafeas_client.list_occurrences(
            parent=parent,
            filter='kind="VULNERABILITY"',
            page_size=100,
        ))

        severity_counts: dict[str, int] = {}
        critical_high = []
        for occ in occurrences:
            vuln = occ.vulnerability
            sev = vuln.severity.name if vuln.severity else "UNKNOWN"
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            if sev in ("CRITICAL", "HIGH"):
                critical_high.append(occ)

        lines.append(f"\nVulnerability occurrences: {len(occurrences)}")
        for sev, count in sorted(severity_counts.items()):
            lines.append(f"  {sev}: {count}")

        if critical_high:
            lines.append(f"\nCRITICAL/HIGH details ({len(critical_high)}):")
            for occ in critical_high[:10]:
                vuln = occ.vulnerability
                sev = vuln.severity.name if vuln.severity else "?"
                pkg = vuln.package_issue[0].affected_package if vuln.package_issue else "unknown"
                lines.append(f"  [{sev}] {pkg} — {occ.resource_uri[:80]}")
            if len(critical_high) > 10:
                lines.append(f"  ... and {len(critical_high) - 10} more CRITICAL/HIGH")
        else:
            lines.append("No CRITICAL or HIGH vulnerabilities found.")

    except Exception as exc:
        lines.append(f"Could not query Container Analysis: {exc}")

    return "\n".join(lines)


@tool
def fedramp_check_dependabot_alerts(
    repo: str = "virtualdojo-inc/virtualdojo",
) -> str:
    """Check GitHub Dependabot alerts for open dependency vulnerabilities.

    Args:
        repo: Repository in 'owner/repo' format (default: virtualdojo-inc/virtualdojo).
    """
    import httpx
    from tools.github import _github_token

    try:
        token = _github_token()
        resp = httpx.get(
            f"https://api.github.com/repos/{repo}/dependabot/alerts?state=open",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        alerts = resp.json()
    except Exception as exc:
        return f"Error fetching Dependabot alerts for {repo}: {exc}"

    if not alerts:
        return f"No open Dependabot alerts for {repo}."

    # Count by severity
    severity_counts: dict[str, int] = {}
    for alert in alerts:
        sev = alert.get("security_advisory", {}).get("severity", "unknown")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    lines = [f"Dependabot Alerts for {repo}: {len(alerts)} open"]
    lines.append("By severity:")
    for sev, count in sorted(severity_counts.items(), key=lambda x: x[1], reverse=True):
        sla = REMEDIATION_SLAS.get(sev.upper(), "N/A")
        lines.append(f"  {sev.upper()}: {count} (SLA: {sla} days)")

    lines.append("\nDetails:")
    for alert in alerts:
        pkg = alert.get("dependency", {}).get("package", {}).get("name", "?")
        sev = alert.get("security_advisory", {}).get("severity", "?")
        title = alert.get("security_advisory", {}).get("summary", "No summary")
        lines.append(f"  [{sev.upper()}] {pkg} — {title}")

    return "\n".join(lines)


@tool
def fedramp_poam_status() -> str:
    """Read POA&M (Plan of Action and Milestones) status from the FedRAMP OSCAL repo.

    Fetches oscal/poam.json from virtualdojo-inc/Fedramp and formats as a status report.
    """
    import httpx
    from tools.github import _github_token

    try:
        token = _github_token()
        resp = httpx.get(
            "https://api.github.com/repos/virtualdojo-inc/Fedramp/contents/oscal/poam.json",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.raw+json",
            },
            timeout=30,
        )
        if resp.status_code == 404:
            return (
                "POA&M file not found at virtualdojo-inc/Fedramp/oscal/poam.json. "
                "The POA&M has not been initialized yet. Use oscal_generate_poam to create one."
            )
        resp.raise_for_status()
    except Exception as exc:
        return f"Error fetching POA&M: {exc}"

    import json

    try:
        poam = json.loads(resp.text)
    except json.JSONDecodeError as exc:
        return f"Error parsing POA&M JSON: {exc}"

    lines = ["=== POA&M Status Report ==="]

    # Navigate OSCAL POA&M structure
    poam_data = poam.get("plan-of-action-and-milestones", poam)
    metadata = poam_data.get("metadata", {})
    lines.append(f"Title: {metadata.get('title', 'N/A')}")
    lines.append(f"Last modified: {metadata.get('last-modified', 'N/A')}")

    poam_items = poam_data.get("poam-items", [])
    lines.append(f"Total POA&M items: {len(poam_items)}")
    lines.append("")

    status_counts: dict[str, int] = {}
    for item in poam_items:
        title = item.get("title", "Untitled")
        description = item.get("description", "")[:100]
        status = item.get("status", "unknown")
        risk_level = item.get("risk-level", item.get("props", [{}])[0].get("value", "?") if item.get("props") else "?")
        status_counts[status] = status_counts.get(status, 0) + 1
        lines.append(f"  [{status.upper()}] {title}")
        if description:
            lines.append(f"    {description}")

    if status_counts:
        lines.append("\nSummary by status:")
        for status, count in sorted(status_counts.items()):
            lines.append(f"  {status}: {count}")

    return "\n".join(lines)


@tool
def fedramp_check_log_retention(
    project_id: str | None = None,
) -> str:
    """Check Cloud Logging bucket retention against FedRAMP AU-11 requirements.

    FedRAMP Moderate requires at least 400-day log retention.

    Args:
        project_id: GCP project ID. Defaults to GCP_PROJECT_ID env var.
    """
    from google.cloud.logging_v2 import ConfigServiceV2Client

    pid = project_id or os.environ.get("GCP_PROJECT_ID", FEDRAMP_PROJECT)

    try:
        client = ConfigServiceV2Client()
        parent = f"projects/{pid}/locations/-"
        buckets = list(client.list_buckets(parent=parent))
    except Exception as exc:
        return f"Error listing log buckets: {exc}"

    if not buckets:
        return f"No log buckets found in project {pid}."

    lines = [f"Log Bucket Retention Audit — project: {pid}", f"FedRAMP Moderate requirement: >= 400 days (AU-11)", ""]
    pass_count = 0
    fail_count = 0

    for bucket in buckets:
        bucket_name = bucket.name.split("/")[-1]
        retention = bucket.retention_days
        locked = bucket.locked
        if retention >= 400:
            status = "PASS"
            pass_count += 1
        else:
            status = "FAIL"
            fail_count += 1
        lock_str = " (LOCKED)" if locked else ""
        lines.append(f"[{status}] {bucket_name}: {retention} days{lock_str}")

    lines.append("")
    lines.append(f"Results: {pass_count} PASS, {fail_count} FAIL out of {len(buckets)} bucket(s)")
    if fail_count:
        lines.append("ACTION REQUIRED: Update retention to >= 400 days for failing buckets.")

    return "\n".join(lines)


@tool
def fedramp_check_encryption(
    project_id: str | None = None,
) -> str:
    """Check KMS key configuration and CMEK usage for FedRAMP SC-12/SC-13 compliance.

    Reviews crypto key rotation, algorithm, and Cloud Run CMEK settings.

    Args:
        project_id: GCP project ID. Defaults to GCP_PROJECT_ID env var.
    """
    from google.cloud import kms_v1

    pid = project_id or os.environ.get("GCP_PROJECT_ID", FEDRAMP_PROJECT)
    lines = [f"Encryption Compliance Report — project: {pid}", ""]

    # Check KMS keys
    try:
        kms_client = kms_v1.KeyManagementServiceClient()
        parent = kms_client.key_ring_path(pid, FEDRAMP_REGION, KMS_KEYRING)
        keys = list(kms_client.list_crypto_keys(request={"parent": parent}))

        lines.append(f"KMS Keyring: {KMS_KEYRING} ({len(keys)} key(s))")
        for key in keys:
            key_name = key.name.split("/")[-1]
            purpose = key.purpose.name if key.purpose else "UNKNOWN"
            state = key.primary.state.name if key.primary else "NO_PRIMARY"
            algorithm = key.primary.algorithm.name if key.primary else "N/A"
            rotation = key.rotation_period
            if rotation and rotation.seconds > 0:
                days = rotation.seconds // 86400
                rot_status = "PASS"
                rot_detail = f"{days} days"
            else:
                rot_status = "FAIL"
                rot_detail = "not configured"

            lines.append(f"  Key: {key_name}")
            lines.append(f"    Purpose: {purpose} | Algorithm: {algorithm} | State: {state}")
            lines.append(f"    [{rot_status}] Rotation: {rot_detail}")
    except Exception as exc:
        lines.append(f"Could not check KMS keys: {exc}")

    # Check Cloud Run CMEK config
    lines.append("")
    try:
        from google.cloud import run_v2

        run_client = run_v2.ServicesClient()
        svc_name = f"projects/{pid}/locations/{FEDRAMP_REGION}/services/{CLOUD_RUN_SERVICE}"
        svc = run_client.get_service(name=svc_name)

        template = svc.template
        cmek = template.encryption_key_revocation_action if hasattr(template, "encryption_key_revocation_action") else None
        encryption_key = template.encryption_key if hasattr(template, "encryption_key") else None

        if encryption_key:
            lines.append(f"[PASS] Cloud Run CMEK: {encryption_key}")
        else:
            lines.append("[INFO] Cloud Run CMEK: Not configured (using Google-managed encryption)")
    except Exception as exc:
        lines.append(f"Could not check Cloud Run CMEK config: {exc}")

    return "\n".join(lines)


@tool
def fedramp_check_iam_compliance(
    project_id: str | None = None,
) -> str:
    """Audit IAM policy for FedRAMP AC-2/AC-6 compliance issues.

    Checks for overly broad roles, public access, and service accounts with
    admin privileges.

    Args:
        project_id: GCP project ID. Defaults to GCP_PROJECT_ID env var.
    """
    from google.cloud import resourcemanager_v3
    from google.iam.v1 import iam_policy_pb2

    pid = project_id or os.environ.get("GCP_PROJECT_ID", FEDRAMP_PROJECT)

    try:
        client = resourcemanager_v3.ProjectsClient()
        request = iam_policy_pb2.GetIamPolicyRequest(resource=f"projects/{pid}")
        policy = client.get_iam_policy(request=request)
    except Exception as exc:
        return f"Error fetching IAM policy for project {pid}: {exc}"

    lines = [f"IAM Compliance Audit — project: {pid}", ""]
    findings: list[tuple[str, str, str]] = []

    role_member_counts: dict[str, int] = {}
    for binding in policy.bindings:
        role = binding.role
        members = list(binding.members)
        role_member_counts[role] = len(members)

        for member in members:
            # CRITICAL: public access
            if member in ("allUsers", "allAuthenticatedUsers"):
                findings.append(("CRITICAL", f"Public access via {member}", f"role: {role}"))

            # HIGH: overly broad roles
            if role in ("roles/owner", "roles/editor"):
                findings.append(("HIGH", f"Overly broad role {role}", f"member: {member}"))

            # MEDIUM: service accounts with admin roles
            if member.startswith("serviceAccount:") and "admin" in role.lower():
                findings.append(("MEDIUM", f"SA with admin role {role}", f"member: {member}"))

    # Report findings by severity
    if findings:
        for level in ("CRITICAL", "HIGH", "MEDIUM"):
            level_findings = [f for f in findings if f[0] == level]
            if level_findings:
                lines.append(f"{level} ({len(level_findings)}):")
                for _, desc, detail in level_findings:
                    lines.append(f"  [{level}] {desc} — {detail}")
                lines.append("")
    else:
        lines.append("No IAM compliance issues found.")
        lines.append("")

    # Role summary
    lines.append("Role member counts:")
    for role, count in sorted(role_member_counts.items()):
        lines.append(f"  {role}: {count} member(s)")

    lines.append("")
    lines.append(f"Total findings: {len(findings)} ({sum(1 for f in findings if f[0] == 'CRITICAL')} critical, "
                 f"{sum(1 for f in findings if f[0] == 'HIGH')} high, "
                 f"{sum(1 for f in findings if f[0] == 'MEDIUM')} medium)")

    return "\n".join(lines)


@tool
def fedramp_evidence_summary(
    project_id: str | None = None,
) -> str:
    """Quick FedRAMP compliance dashboard — runs lightweight checks for all control families.

    Provides a high-level overview of compliance posture across all NIST 800-53
    control families. Individual check failures do not break the overall summary.

    Args:
        project_id: GCP project ID. Defaults to GCP_PROJECT_ID env var.
    """
    pid = project_id or os.environ.get("GCP_PROJECT_ID", FEDRAMP_PROJECT)
    lines = [
        f"FedRAMP Evidence Summary Dashboard — project: {pid}",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "=" * 60,
    ]

    overall_pass = 0
    overall_fail = 0
    overall_info = 0

    for family_code, family_name in CONTROL_FAMILIES.items():
        try:
            results = _FAMILY_DISPATCH[family_code](pid)
            passes = sum(1 for _, s, _ in results if s == "PASS")
            fails = sum(1 for _, s, _ in results if s == "FAIL")
            infos = sum(1 for _, s, _ in results if s == "INFO")
            overall_pass += passes
            overall_fail += fails
            overall_info += infos

            if fails > 0:
                status = "NEEDS ATTENTION"
            elif passes > 0:
                status = "COMPLIANT"
            else:
                status = "REVIEW"

            lines.append(f"  {family_code} {family_name}: {status} ({passes}P/{fails}F/{infos}I)")
        except Exception as exc:
            lines.append(f"  {family_code} {family_name}: ERROR — {exc}")

    lines.append("=" * 60)
    lines.append(f"Totals: {overall_pass} PASS / {overall_fail} FAIL / {overall_info} INFO")
    if overall_fail > 0:
        lines.append(f"ACTION REQUIRED: {overall_fail} failing check(s) need remediation.")
    else:
        lines.append("All automated checks passing.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

FEDRAMP_TOOLS = [
    fedramp_collect_evidence,
    fedramp_check_scc_findings,
    fedramp_daily_log_review,
    fedramp_check_failed_logins,
    fedramp_scan_container_vulnerabilities,
    fedramp_check_dependabot_alerts,
    fedramp_poam_status,
    fedramp_check_log_retention,
    fedramp_check_encryption,
    fedramp_check_iam_compliance,
    fedramp_evidence_summary,
]
