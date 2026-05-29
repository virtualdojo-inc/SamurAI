"""Tests for tools.fedramp — GCP FedRAMP compliance evidence collection and monitoring."""

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_TOKEN = "ghp_test_token_123"


def _inject_missing_module(full_path: str) -> MagicMock:
    """Ensure a dotted module path exists in sys.modules so local imports work.

    Returns the mock for the *leaf* module.
    """
    parts = full_path.split(".")
    for i in range(1, len(parts) + 1):
        mod_path = ".".join(parts[:i])
        if mod_path not in sys.modules:
            sys.modules[mod_path] = MagicMock()
    return sys.modules[full_path]


# ---------------------------------------------------------------------------
# fedramp_collect_evidence
# ---------------------------------------------------------------------------


class TestCollectEvidence:
    """Tests for fedramp_collect_evidence — dispatches to per-family collectors."""

    def test_invalid_control_family(self):
        from tools.fedramp import fedramp_collect_evidence

        result = fedramp_collect_evidence.invoke({"control_family": "ZZ"})
        assert "Unknown control family" in result
        assert "ZZ" in result
        assert "AC" in result  # should list valid families

    def test_case_insensitive_family(self):
        """Lower-case family codes should be accepted."""
        from tools.fedramp import fedramp_collect_evidence

        with patch("tools.fedramp._FAMILY_DISPATCH", {"SR": lambda pid: [("SBOM", "INFO", "ok")]}):
            result = fedramp_collect_evidence.invoke({"control_family": "sr"})
            assert "SR" in result
            assert "SBOM" in result

    @patch("google.cloud.resourcemanager_v3.ProjectsClient")
    def test_ac_family_pass(self, mock_projects_cls):
        from tools.fedramp import fedramp_collect_evidence

        binding = MagicMock()
        binding.role = "roles/viewer"
        binding.members = ["serviceAccount:sa@proj.iam.gserviceaccount.com"]

        mock_policy = MagicMock()
        mock_policy.bindings = [binding]

        mock_client = MagicMock()
        mock_client.get_iam_policy.return_value = mock_policy
        mock_projects_cls.return_value = mock_client

        result = fedramp_collect_evidence.invoke(
            {"control_family": "AC", "project_id": "test-project"}
        )
        assert "AC" in result
        assert "Access Control" in result
        assert "PASS" in result
        assert "No Owner/Editor bindings" in result

    @patch("google.cloud.resourcemanager_v3.ProjectsClient")
    def test_ac_family_fail_broad_roles(self, mock_projects_cls):
        from tools.fedramp import fedramp_collect_evidence

        binding = MagicMock()
        binding.role = "roles/owner"
        binding.members = ["user:admin@example.com"]

        mock_policy = MagicMock()
        mock_policy.bindings = [binding]

        mock_client = MagicMock()
        mock_client.get_iam_policy.return_value = mock_policy
        mock_projects_cls.return_value = mock_client

        result = fedramp_collect_evidence.invoke(
            {"control_family": "AC", "project_id": "test-project"}
        )
        assert "FAIL" in result
        assert "Owner/Editor" in result

    def test_dispatch_error_handling(self):
        """Exceptions in a family collector should be caught gracefully."""
        from tools.fedramp import fedramp_collect_evidence

        def _boom(pid):
            raise RuntimeError("GCP API down")

        with patch("tools.fedramp._FAMILY_DISPATCH", {"AC": _boom}):
            result = fedramp_collect_evidence.invoke(
                {"control_family": "AC", "project_id": "test-project"}
            )
            assert "Error collecting evidence" in result
            assert "GCP API down" in result


# ---------------------------------------------------------------------------
# fedramp_check_scc_findings
# ---------------------------------------------------------------------------


class TestCheckSccFindings:
    """Tests for fedramp_check_scc_findings — Security Command Center queries."""

    def test_invalid_severity(self):
        from tools.fedramp import fedramp_check_scc_findings

        # The import happens before the severity check, so inject the module
        _inject_missing_module("google.cloud.securitycenter_v1")
        result = fedramp_check_scc_findings.invoke({"severity": "SUPERCRITICAL"})
        assert "Invalid severity" in result

    def test_no_findings(self):
        from tools.fedramp import fedramp_check_scc_findings

        mock_scc_mod = _inject_missing_module("google.cloud.securitycenter_v1")
        mock_client = MagicMock()
        mock_client.list_findings.return_value = iter([])
        mock_scc_mod.SecurityCenterClient.return_value = mock_client

        # Also inject PermissionDenied so the import inside the function works
        _inject_missing_module("google.api_core.exceptions")

        result = fedramp_check_scc_findings.invoke(
            {"project_id": "test-project", "severity": "HIGH"}
        )
        assert "No active findings" in result
        assert "Total: 0" in result

    def test_findings_returned(self):
        from tools.fedramp import fedramp_check_scc_findings

        mock_scc_mod = _inject_missing_module("google.cloud.securitycenter_v1")

        mock_finding = MagicMock()
        mock_finding.finding.severity.name = "CRITICAL"
        mock_finding.finding.category = "SQL_INJECTION"
        mock_finding.finding.resource_name = "//compute.googleapis.com/projects/test"
        mock_finding.finding.description = "SQL injection vulnerability detected"

        mock_client = MagicMock()
        mock_client.list_findings.return_value = iter([mock_finding])
        mock_scc_mod.SecurityCenterClient.return_value = mock_client

        result = fedramp_check_scc_findings.invoke(
            {"project_id": "test-project", "severity": "HIGH"}
        )
        assert "CRITICAL" in result
        assert "SQL_INJECTION" in result
        assert "Total: 1" in result

    def test_permission_denied(self):
        from google.api_core.exceptions import PermissionDenied
        from tools.fedramp import fedramp_check_scc_findings

        mock_scc_mod = _inject_missing_module("google.cloud.securitycenter_v1")
        mock_client = MagicMock()
        mock_client.list_findings.side_effect = PermissionDenied("nope")
        mock_scc_mod.SecurityCenterClient.return_value = mock_client

        result = fedramp_check_scc_findings.invoke({"project_id": "test-project"})
        assert "Permission denied" in result
        assert "findingsViewer" in result


# ---------------------------------------------------------------------------
# fedramp_daily_log_review
# ---------------------------------------------------------------------------


class TestDailyLogReview:
    """Tests for fedramp_daily_log_review — 7-category log audit."""

    @patch("google.cloud.logging.Client")
    def test_normal_operation(self, mock_logging_cls):
        from tools.fedramp import fedramp_daily_log_review

        mock_entry = MagicMock()
        mock_entry.timestamp.isoformat.return_value = "2026-04-06T00:00:00Z"
        mock_entry.severity = "INFO"
        mock_entry.payload = "some log data"

        mock_client = MagicMock()
        mock_client.list_entries.return_value = [mock_entry]
        mock_logging_cls.return_value = mock_client

        result = fedramp_daily_log_review.invoke(
            {"project_id": "test-project", "hours": 24}
        )
        assert "FedRAMP Daily Log Review" in result
        assert "Admin Activity" in result
        assert "Data Access" in result
        assert "Summary:" in result
        assert "7 total events" in result

    @patch("google.cloud.logging.Client")
    def test_empty_logs(self, mock_logging_cls):
        from tools.fedramp import fedramp_daily_log_review

        mock_client = MagicMock()
        mock_client.list_entries.return_value = []
        mock_logging_cls.return_value = mock_client

        result = fedramp_daily_log_review.invoke(
            {"project_id": "test-project", "hours": 24}
        )
        assert "0 total events" in result

    @patch("google.cloud.logging.Client")
    def test_query_error_per_category(self, mock_logging_cls):
        from tools.fedramp import fedramp_daily_log_review

        mock_client = MagicMock()
        mock_client.list_entries.side_effect = RuntimeError("quota exceeded")
        mock_logging_cls.return_value = mock_client

        result = fedramp_daily_log_review.invoke({"project_id": "test-project"})
        assert "ERROR" in result
        assert "quota exceeded" in result


# ---------------------------------------------------------------------------
# fedramp_check_failed_logins
# ---------------------------------------------------------------------------


class TestCheckFailedLogins:
    """Tests for fedramp_check_failed_logins — failed auth aggregation."""

    @patch("google.cloud.logging.Client")
    def test_no_failed_logins(self, mock_logging_cls):
        from tools.fedramp import fedramp_check_failed_logins

        mock_client = MagicMock()
        mock_client.list_entries.return_value = []
        mock_logging_cls.return_value = mock_client

        result = fedramp_check_failed_logins.invoke({"project_id": "test-project"})
        assert "Total failed auth events: 0" in result
        assert "No failed authentication events" in result

    @patch("google.cloud.logging.Client")
    def test_brute_force_detection(self, mock_logging_cls):
        from tools.fedramp import fedramp_check_failed_logins

        entries = []
        for _ in range(6):
            entry = MagicMock()
            entry.payload = {"requestMetadata": {"callerIp": "10.0.0.99"}}
            entries.append(entry)

        mock_client = MagicMock()
        mock_client.list_entries.return_value = entries
        mock_logging_cls.return_value = mock_client

        result = fedramp_check_failed_logins.invoke({"project_id": "test-project"})
        assert "10.0.0.99" in result
        assert "ALERT" in result
        assert "brute force" in result

    @patch("google.cloud.logging.Client")
    def test_query_error(self, mock_logging_cls):
        from tools.fedramp import fedramp_check_failed_logins

        mock_client = MagicMock()
        mock_client.list_entries.side_effect = RuntimeError("timeout")
        mock_logging_cls.return_value = mock_client

        result = fedramp_check_failed_logins.invoke({"project_id": "test-project"})
        assert "Error querying failed logins" in result


# ---------------------------------------------------------------------------
# fedramp_scan_container_vulnerabilities
# ---------------------------------------------------------------------------


class TestScanContainerVulnerabilities:
    """Tests for fedramp_scan_container_vulnerabilities — Artifact Registry + Grafeas."""

    def test_no_vulnerabilities(self):
        from tools.fedramp import fedramp_scan_container_vulnerabilities

        # Inject mock modules for packages that may not be installed
        ar_mod = _inject_missing_module("google.cloud.artifactregistry_v1")
        ca_mod = _inject_missing_module("google.devtools.containeranalysis.v1")

        mock_img = MagicMock()
        mock_img.name = "projects/p/locations/l/repositories/r/dockerImages/img1"
        mock_img.tags = ["latest"]
        mock_ar_client = MagicMock()
        mock_ar_client.list_docker_images.return_value = [mock_img]
        ar_mod.ArtifactRegistryClient.return_value = mock_ar_client

        mock_grafeas = MagicMock()
        mock_grafeas.list_occurrences.return_value = []
        ca_mod.GrafeasClient.return_value = mock_grafeas

        result = fedramp_scan_container_vulnerabilities.invoke(
            {"project_id": "test-project"}
        )
        assert "Docker images in registry: 1" in result
        assert "No CRITICAL or HIGH vulnerabilities" in result

    def test_critical_vulnerability(self):
        from tools.fedramp import fedramp_scan_container_vulnerabilities

        ar_mod = _inject_missing_module("google.cloud.artifactregistry_v1")
        ca_mod = _inject_missing_module("google.devtools.containeranalysis.v1")

        mock_ar_client = MagicMock()
        mock_ar_client.list_docker_images.return_value = []
        ar_mod.ArtifactRegistryClient.return_value = mock_ar_client

        mock_occ = MagicMock()
        mock_occ.vulnerability.severity.name = "CRITICAL"
        mock_occ.vulnerability.package_issue = [MagicMock(affected_package="openssl")]
        mock_occ.resource_uri = "us-central1-docker.pkg.dev/test-project/repo/img@sha256:abc"

        mock_grafeas = MagicMock()
        mock_grafeas.list_occurrences.return_value = [mock_occ]
        ca_mod.GrafeasClient.return_value = mock_grafeas

        result = fedramp_scan_container_vulnerabilities.invoke(
            {"project_id": "test-project"}
        )
        assert "CRITICAL" in result
        assert "openssl" in result

    def test_ar_error(self):
        from tools.fedramp import fedramp_scan_container_vulnerabilities

        ar_mod = _inject_missing_module("google.cloud.artifactregistry_v1")
        mock_ar_client = MagicMock()
        mock_ar_client.list_docker_images.side_effect = RuntimeError("forbidden")
        ar_mod.ArtifactRegistryClient.return_value = mock_ar_client

        result = fedramp_scan_container_vulnerabilities.invoke(
            {"project_id": "test-project"}
        )
        assert "Could not list Artifact Registry images" in result


# ---------------------------------------------------------------------------
# fedramp_check_dependabot_alerts
# ---------------------------------------------------------------------------


class TestCheckDependabotAlerts:
    """Tests for fedramp_check_dependabot_alerts — GitHub Dependabot API."""

    @patch("httpx.get")
    @patch("tools.github._github_token", return_value=MOCK_TOKEN)
    def test_no_alerts(self, mock_token, mock_get):
        from tools.fedramp import fedramp_check_dependabot_alerts

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=[]),
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = fedramp_check_dependabot_alerts.invoke(
            {"repo": "virtualdojo-inc/virtualdojo"}
        )
        assert "No open Dependabot alerts" in result

    @patch("httpx.get")
    @patch("tools.github._github_token", return_value=MOCK_TOKEN)
    def test_alerts_with_severity(self, mock_token, mock_get):
        from tools.fedramp import fedramp_check_dependabot_alerts

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value=[
                    {
                        "security_advisory": {
                            "severity": "high",
                            "summary": "RCE in openssl",
                        },
                        "dependency": {"package": {"name": "openssl"}},
                    },
                    {
                        "security_advisory": {
                            "severity": "critical",
                            "summary": "Buffer overflow in curl",
                        },
                        "dependency": {"package": {"name": "curl"}},
                    },
                ]
            ),
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = fedramp_check_dependabot_alerts.invoke(
            {"repo": "virtualdojo-inc/virtualdojo"}
        )
        assert "2 open" in result
        assert "HIGH" in result
        assert "CRITICAL" in result
        assert "openssl" in result
        assert "curl" in result

    @patch("httpx.get")
    @patch("tools.github._github_token", return_value=MOCK_TOKEN)
    def test_api_error(self, mock_token, mock_get):
        from tools.fedramp import fedramp_check_dependabot_alerts

        mock_get.side_effect = RuntimeError("network timeout")

        result = fedramp_check_dependabot_alerts.invoke(
            {"repo": "virtualdojo-inc/virtualdojo"}
        )
        assert "Error fetching Dependabot alerts" in result


# ---------------------------------------------------------------------------
# fedramp_poam_status
# ---------------------------------------------------------------------------


class TestPoamStatus:
    """Tests for fedramp_poam_status — reads OSCAL POA&M from repo."""

    @patch("httpx.get")
    @patch("tools.github._github_token", return_value=MOCK_TOKEN)
    def test_poam_not_found(self, mock_token, mock_get):
        from tools.fedramp import fedramp_poam_status

        mock_get.return_value = MagicMock(status_code=404)

        result = fedramp_poam_status.invoke({})
        assert "POA&M file not found" in result

    @patch("httpx.get")
    @patch("tools.github._github_token", return_value=MOCK_TOKEN)
    def test_poam_parsed(self, mock_token, mock_get):
        import json
        from tools.fedramp import fedramp_poam_status

        poam_json = json.dumps({
            "plan-of-action-and-milestones": {
                "metadata": {
                    "title": "VirtualDojo POA&M",
                    "last-modified": "2026-04-01T00:00:00Z",
                },
                "poam-items": [
                    {
                        "title": "POA&M Item: AC-2",
                        "description": "Account management finding",
                        "status": "open",
                    }
                ],
            }
        })

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = poam_json
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fedramp_poam_status.invoke({})
        assert "POA&M Status Report" in result
        assert "VirtualDojo POA&M" in result
        assert "OPEN" in result
        assert "AC-2" in result


# ---------------------------------------------------------------------------
# fedramp_check_log_retention
# ---------------------------------------------------------------------------


class TestCheckLogRetention:
    """Tests for fedramp_check_log_retention — AU-11 bucket retention audit."""

    def _make_mock_config_module(self):
        """Inject ConfigServiceV2Client into google.cloud.logging_v2."""
        mock_config_client = MagicMock()
        # Patch the module-level attribute so `from google.cloud.logging_v2 import ConfigServiceV2Client` works
        import google.cloud.logging_v2 as logging_v2
        self._original_attr = getattr(logging_v2, "ConfigServiceV2Client", None)
        logging_v2.ConfigServiceV2Client = MagicMock()
        return logging_v2.ConfigServiceV2Client

    def _restore_config_module(self):
        import google.cloud.logging_v2 as logging_v2
        if self._original_attr is None:
            if hasattr(logging_v2, "ConfigServiceV2Client"):
                delattr(logging_v2, "ConfigServiceV2Client")
        else:
            logging_v2.ConfigServiceV2Client = self._original_attr

    def test_passing_retention(self):
        from tools.fedramp import fedramp_check_log_retention

        mock_cls = self._make_mock_config_module()
        try:
            bucket = MagicMock()
            bucket.name = "projects/p/locations/global/buckets/_Default"
            bucket.retention_days = 400
            bucket.locked = True

            mock_client = MagicMock()
            mock_client.list_buckets.return_value = [bucket]
            mock_cls.return_value = mock_client

            result = fedramp_check_log_retention.invoke({"project_id": "test-project"})
            assert "[PASS]" in result
            assert "400 days" in result
            assert "LOCKED" in result
            assert "1 PASS, 0 FAIL" in result
        finally:
            self._restore_config_module()

    def test_failing_retention(self):
        from tools.fedramp import fedramp_check_log_retention

        mock_cls = self._make_mock_config_module()
        try:
            bucket = MagicMock()
            bucket.name = "projects/p/locations/global/buckets/_Default"
            bucket.retention_days = 30
            bucket.locked = False

            mock_client = MagicMock()
            mock_client.list_buckets.return_value = [bucket]
            mock_cls.return_value = mock_client

            result = fedramp_check_log_retention.invoke({"project_id": "test-project"})
            assert "[FAIL]" in result
            assert "0 PASS, 1 FAIL" in result
            assert "ACTION REQUIRED" in result
        finally:
            self._restore_config_module()

    def test_no_buckets(self):
        from tools.fedramp import fedramp_check_log_retention

        mock_cls = self._make_mock_config_module()
        try:
            mock_client = MagicMock()
            mock_client.list_buckets.return_value = []
            mock_cls.return_value = mock_client

            result = fedramp_check_log_retention.invoke({"project_id": "test-project"})
            assert "No log buckets found" in result
        finally:
            self._restore_config_module()

    def test_api_error(self):
        from tools.fedramp import fedramp_check_log_retention

        mock_cls = self._make_mock_config_module()
        try:
            mock_client = MagicMock()
            mock_client.list_buckets.side_effect = RuntimeError("permission denied")
            mock_cls.return_value = mock_client

            result = fedramp_check_log_retention.invoke({"project_id": "test-project"})
            assert "Error listing log buckets" in result
        finally:
            self._restore_config_module()


# ---------------------------------------------------------------------------
# fedramp_check_encryption
# ---------------------------------------------------------------------------


class TestCheckEncryption:
    """Tests for fedramp_check_encryption — KMS + CMEK SC-12/SC-13."""

    def test_keys_with_rotation(self):
        from tools.fedramp import fedramp_check_encryption

        kms_mod = _inject_missing_module("google.cloud.kms_v1")

        mock_key = MagicMock()
        mock_key.name = "projects/p/locations/l/keyRings/kr/cryptoKeys/my-key"
        mock_key.purpose.name = "ENCRYPT_DECRYPT"
        mock_key.primary.state.name = "ENABLED"
        mock_key.primary.algorithm.name = "GOOGLE_SYMMETRIC_ENCRYPTION"
        mock_key.rotation_period.seconds = 7776000  # 90 days

        mock_kms = MagicMock()
        mock_kms.key_ring_path.return_value = "projects/p/locations/l/keyRings/kr"
        mock_kms.list_crypto_keys.return_value = [mock_key]
        kms_mod.KeyManagementServiceClient.return_value = mock_kms

        # Cloud Run part — already installed, so patch works
        with patch("google.cloud.run_v2.ServicesClient") as mock_run_cls:
            mock_svc = MagicMock()
            mock_svc.template.encryption_key = "projects/p/locations/l/keyRings/kr/cryptoKeys/my-key"
            mock_run_cls.return_value.get_service.return_value = mock_svc

            result = fedramp_check_encryption.invoke({"project_id": "test-project"})
            assert "Encryption Compliance Report" in result
            assert "[PASS] Rotation:" in result
            assert "90 days" in result
            assert "[PASS] Cloud Run CMEK" in result

    def test_key_no_rotation(self):
        from tools.fedramp import fedramp_check_encryption

        kms_mod = _inject_missing_module("google.cloud.kms_v1")

        mock_key = MagicMock()
        mock_key.name = "projects/p/locations/l/keyRings/kr/cryptoKeys/norot-key"
        mock_key.purpose.name = "ENCRYPT_DECRYPT"
        mock_key.primary.state.name = "ENABLED"
        mock_key.primary.algorithm.name = "GOOGLE_SYMMETRIC_ENCRYPTION"
        mock_key.rotation_period.seconds = 0

        mock_kms = MagicMock()
        mock_kms.key_ring_path.return_value = "projects/p/locations/l/keyRings/kr"
        mock_kms.list_crypto_keys.return_value = [mock_key]
        kms_mod.KeyManagementServiceClient.return_value = mock_kms

        result = fedramp_check_encryption.invoke({"project_id": "test-project"})
        assert "[FAIL] Rotation: not configured" in result


# ---------------------------------------------------------------------------
# fedramp_check_iam_compliance
# ---------------------------------------------------------------------------


class TestCheckIamCompliance:
    """Tests for fedramp_check_iam_compliance — AC-2/AC-6 IAM audit."""

    @patch("google.cloud.resourcemanager_v3.ProjectsClient")
    def test_clean_iam(self, mock_projects_cls):
        from tools.fedramp import fedramp_check_iam_compliance

        binding = MagicMock()
        binding.role = "roles/viewer"
        binding.members = ["user:reader@example.com"]

        mock_policy = MagicMock()
        mock_policy.bindings = [binding]

        mock_client = MagicMock()
        mock_client.get_iam_policy.return_value = mock_policy
        mock_projects_cls.return_value = mock_client

        result = fedramp_check_iam_compliance.invoke({"project_id": "test-project"})
        assert "No IAM compliance issues found" in result
        assert "Total findings: 0" in result

    @patch("google.cloud.resourcemanager_v3.ProjectsClient")
    def test_public_access_finding(self, mock_projects_cls):
        from tools.fedramp import fedramp_check_iam_compliance

        binding = MagicMock()
        binding.role = "roles/storage.objectViewer"
        binding.members = ["allUsers"]

        mock_policy = MagicMock()
        mock_policy.bindings = [binding]

        mock_client = MagicMock()
        mock_client.get_iam_policy.return_value = mock_policy
        mock_projects_cls.return_value = mock_client

        result = fedramp_check_iam_compliance.invoke({"project_id": "test-project"})
        assert "CRITICAL" in result
        assert "Public access" in result
        assert "allUsers" in result

    @patch("google.cloud.resourcemanager_v3.ProjectsClient")
    def test_overly_broad_roles(self, mock_projects_cls):
        from tools.fedramp import fedramp_check_iam_compliance

        binding = MagicMock()
        binding.role = "roles/owner"
        binding.members = ["user:dev@example.com"]

        mock_policy = MagicMock()
        mock_policy.bindings = [binding]

        mock_client = MagicMock()
        mock_client.get_iam_policy.return_value = mock_policy
        mock_projects_cls.return_value = mock_client

        result = fedramp_check_iam_compliance.invoke({"project_id": "test-project"})
        assert "HIGH" in result
        assert "Overly broad role" in result

    @patch("google.cloud.resourcemanager_v3.ProjectsClient")
    def test_api_error(self, mock_projects_cls):
        from tools.fedramp import fedramp_check_iam_compliance

        mock_client = MagicMock()
        mock_client.get_iam_policy.side_effect = RuntimeError("permission denied")
        mock_projects_cls.return_value = mock_client

        result = fedramp_check_iam_compliance.invoke({"project_id": "test-project"})
        assert "Error fetching IAM policy" in result


# ---------------------------------------------------------------------------
# fedramp_evidence_summary
# ---------------------------------------------------------------------------


class TestEvidenceSummary:
    """Tests for fedramp_evidence_summary — dashboard across all families."""

    def test_all_families_pass(self):
        from tools.fedramp import fedramp_evidence_summary

        def _pass_collector(pid):
            return [("Check", "PASS", "All good")]

        mock_dispatch = {code: _pass_collector for code in [
            "AC", "AU", "CM", "CP", "IA", "RA", "SC", "SI", "SR",
        ]}
        with patch("tools.fedramp._FAMILY_DISPATCH", mock_dispatch):
            result = fedramp_evidence_summary.invoke({"project_id": "test-project"})
            assert "Evidence Summary Dashboard" in result
            assert "COMPLIANT" in result
            assert "All automated checks passing" in result

    def test_family_with_failures(self):
        from tools.fedramp import fedramp_evidence_summary

        def _fail_collector(pid):
            return [("Check", "FAIL", "Something is wrong")]

        def _pass_collector(pid):
            return [("Check", "PASS", "OK")]

        mock_dispatch = {code: _pass_collector for code in [
            "AC", "AU", "CM", "CP", "IA", "RA", "SC", "SI", "SR",
        ]}
        mock_dispatch["SC"] = _fail_collector

        with patch("tools.fedramp._FAMILY_DISPATCH", mock_dispatch):
            result = fedramp_evidence_summary.invoke({"project_id": "test-project"})
            assert "NEEDS ATTENTION" in result
            assert "ACTION REQUIRED" in result

    def test_family_error_doesnt_crash(self):
        from tools.fedramp import fedramp_evidence_summary

        def _boom(pid):
            raise RuntimeError("API down")

        def _pass_collector(pid):
            return [("Check", "PASS", "OK")]

        mock_dispatch = {code: _pass_collector for code in [
            "AC", "AU", "CM", "CP", "IA", "RA", "SC", "SI", "SR",
        ]}
        mock_dispatch["AU"] = _boom

        with patch("tools.fedramp._FAMILY_DISPATCH", mock_dispatch):
            result = fedramp_evidence_summary.invoke({"project_id": "test-project"})
            assert "AU" in result
            assert "ERROR" in result
            assert "COMPLIANT" in result


# ---------------------------------------------------------------------------
# Audit collector helpers — AU, CM
# ---------------------------------------------------------------------------


class TestCollectAu:
    """Tests for _collect_au — Audit and Accountability evidence."""

    @patch("google.cloud.logging.Client")
    def test_sinks_present(self, mock_logging_cls):
        from tools.fedramp import _collect_au

        mock_sink = MagicMock()
        mock_sink.name = "fedramp-export-sink"

        mock_client = MagicMock()
        mock_client.list_sinks.return_value = [mock_sink]
        mock_logging_cls.return_value = mock_client

        results = _collect_au("test-project")
        statuses = [s for _, s, _ in results]
        assert "PASS" in statuses
        assert any("fedramp-export-sink" in d for _, _, d in results)

    @patch("google.cloud.logging.Client")
    def test_no_sinks(self, mock_logging_cls):
        from tools.fedramp import _collect_au

        mock_client = MagicMock()
        mock_client.list_sinks.return_value = []
        mock_logging_cls.return_value = mock_client

        results = _collect_au("test-project")
        statuses = [s for _, s, _ in results]
        assert "FAIL" in statuses


class TestCollectCm:
    """Tests for _collect_cm — Configuration Management evidence."""

    @patch("google.cloud.run_v2.ServicesClient")
    def test_service_info(self, mock_run_cls):
        from tools.fedramp import _collect_cm

        mock_container = MagicMock()
        mock_container.env = [MagicMock(), MagicMock()]

        mock_scaling = MagicMock()
        mock_scaling.min_instance_count = 1
        mock_scaling.max_instance_count = 5

        mock_svc = MagicMock()
        mock_svc.template.containers = [mock_container]
        mock_svc.template.scaling = mock_scaling
        mock_svc.latest_ready_revision = "projects/p/locations/l/services/s/revisions/rev-001"

        mock_client = MagicMock()
        mock_client.get_service.return_value = mock_svc
        mock_run_cls.return_value = mock_client

        results = _collect_cm("test-project")
        details = [d for _, _, d in results]
        assert any("2 env vars" in d for d in details)
        assert any("min=1" in d for d in details)
        assert any("rev-001" in d for d in details)

    @patch("google.cloud.run_v2.ServicesClient")
    def test_service_not_found(self, mock_run_cls):
        from tools.fedramp import _collect_cm

        mock_client = MagicMock()
        mock_client.get_service.side_effect = RuntimeError("not found")
        mock_run_cls.return_value = mock_client

        results = _collect_cm("test-project")
        statuses = [s for _, s, _ in results]
        assert "FAIL" in statuses
