"""Tests for the deterministic sanitization gate (sanitizer.py)."""

from datetime import datetime, timezone

import pytest

import sanitizer
from sanitizer import (
    GateResult,
    Mode,
    build_audit_record,
    scan_artifacts,
    scan_text,
    validate_branch_name,
)


# --- secrets (both modes) ---------------------------------------------------------

@pytest.mark.parametrize("text", [
    "AKIAIOSFODNN7EXAMPLE",
    "AIza" + "Sy0aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  # AIza + exactly 35 chars
    "ghp_012345678901234567890123456789012345",
    "-----BEGIN RSA PRIVATE KEY-----",
    "api_key = 'sk-verysecretvalue123'",
    "Authorization: Bearer abcdef012345678901234567890",
])
def test_secrets_blocked_in_both_modes(text):
    for mode in (Mode.FULL, Mode.LIGHT):
        res = scan_text(text, mode=mode)
        assert res.blocked, f"{text!r} should block in {mode}"
        assert res.findings[0].category == "secret"


def test_secret_preview_is_masked():
    res = scan_text("ghp_012345678901234567890123456789012345", mode=Mode.LIGHT)
    preview = res.findings[0].match_preview
    assert "012345678901234567890" not in preview
    assert "*" in preview


# --- PII (full mode only) ---------------------------------------------------------

@pytest.mark.parametrize("text,rule", [
    ("contact jane@acme.com about it", "email"),
    ("SSN 123-45-6789 on file", "ssn"),
    ("EIN 12-3456789 for the entity", "ein"),
    ("call (555) 123-4567 today", "phone"),
    ("server at 203.0.113.42 failed", "ipv4"),
])
def test_pii_blocked_in_full_mode(text, rule):
    res = scan_text(text, mode=Mode.FULL)
    assert res.blocked
    assert any(f.rule == rule for f in res.findings)


@pytest.mark.parametrize("text", [
    "contact jane@acme.com",
    "SSN 123-45-6789",
    "server at 203.0.113.42",
])
def test_pii_ignored_in_light_mode(text):
    # Claude Code sessions never touch customer data → PII checks are off.
    res = scan_text(text, mode=Mode.LIGHT)
    assert not res.blocked


def test_private_and_loopback_ips_not_blocked():
    for ip in ("127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.1"):
        res = scan_text(f"bind to {ip}", mode=Mode.FULL)
        assert not any(f.rule == "ipv4" for f in res.findings), ip


# --- tenant denylist (full mode only) ---------------------------------------------

def test_tenant_name_blocked_full_mode():
    res = scan_text(
        "the recurring bug for Acme Corp invoices",
        mode=Mode.FULL,
        tenant_names=["Acme Corp", "Globex"],
    )
    assert res.blocked
    assert any(f.category == "tenant" for f in res.findings)


def test_tenant_name_ignored_light_mode():
    res = scan_text(
        "the recurring bug for Acme Corp invoices",
        mode=Mode.LIGHT,
        tenant_names=["Acme Corp"],
    )
    assert not res.blocked


def test_tenant_substring_does_not_false_positive():
    # "cat" must not match inside "category"
    res = scan_text("update the category field", mode=Mode.FULL, tenant_names=["Cat"])
    assert not any(f.category == "tenant" for f in res.findings)


# --- clean pattern-only skill passes ----------------------------------------------

def test_clean_skill_passes():
    body = ("When a bulk write hits an N+1 per-row flush, batch with a Core multi-row "
            "insert instead of add_all. Verify with a row-count assertion.")
    res = scan_text(body, mode=Mode.FULL, tenant_names=["Acme Corp"])
    assert not res.blocked


# --- artifact-level gate ----------------------------------------------------------

def test_scan_artifacts_flags_pr_body():
    gate = scan_artifacts(
        {
            "name": "n-plus-one-batch-flush-fix",
            "description": "Fix per-row flush N+1 on bulk writes",
            "body": "clean pattern text",
            "pr_body": "seen while debugging jane@acme.com's tenant",
        },
        mode=Mode.FULL,
    )
    assert isinstance(gate, GateResult)
    assert gate.blocked
    assert any(f.field == "pr_body" for f in gate.findings)


def test_scan_artifacts_clean_passes():
    gate = scan_artifacts(
        {
            "name": "clone-clin-status-bulk-write",
            "description": "Consolidate CLIN status writes into one bulk insert",
            "body": "Use a single Core insert; assert affected row count.",
            "pr_title": "Add skill: bulk CLIN status write",
            "commit_message": "skill draft",
        },
        mode=Mode.FULL,
        tenant_names=["Acme Corp"],
    )
    assert not gate.blocked


# --- branch-name slug validation --------------------------------------------------

def test_valid_branch_slug():
    assert validate_branch_name("skill-draft/2f1c8e3a-4b5d-6e7f-8a9b-0c1d2e3f4a5b")


@pytest.mark.parametrize("bad", [
    "skill-draft/acme-invoice-fix",   # content-derived → would leak
    "main",
    "skill-draft/",
    "feature/skill-draft/uuid",
    "",
])
def test_invalid_branch_slugs_rejected(bad):
    assert not validate_branch_name(bad)


# --- audit record -----------------------------------------------------------------

def test_audit_record_hashes_input_and_omits_raw():
    artifacts = {"body": "api_key = 'sk-secret12345'"}
    gate = scan_artifacts(artifacts, mode=Mode.FULL)
    rec = build_audit_record(
        artifacts, gate,
        llm_verdict="pass", llm_reason="looks clean",
        judge_prompt_version="v1",
        now=datetime(2026, 7, 22, tzinfo=timezone.utc),
    )
    assert rec["deterministic_blocked"] is True
    assert rec["ruleset_version"] == sanitizer.RULESET_VERSION
    assert rec["judge_prompt_version"] == "v1"
    assert rec["ts"] == "2026-07-22T00:00:00+00:00"
    assert len(rec["input_sha256"]) == 64
    # raw secret must not appear anywhere in the audit record
    assert "sk-secret12345" not in str(rec)
