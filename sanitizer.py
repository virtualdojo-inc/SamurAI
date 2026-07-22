"""Deterministic sanitization gate for the skills feedback loop.

This is the **first, hard, auditable** layer of the boundary control described in
``docs/skillsfeedbacklooprequirements.md`` and the approved plan. Before any skill
artifact crosses out to GitHub (skill name/description/body, PR title/body/comments,
commit messages, branch names), it is scanned here for private content:

  * PII       — emails, US SSN/EIN, phone numbers, IPv4 addresses
  * secrets   — API-key/token shapes, private-key headers, credential assignments
  * tenants   — a customer/tenant-name denylist (SamurAI/"full" mode only)

A match **blocks** the artifact regardless of what any downstream LLM judge decides.
The LLM pass is *additional* screening layered on top of this, never a substitute.

Two profiles (see the plan's boundary resolution, 2026-07-22):

  * ``Mode.FULL``  — SamurAI: PII + secrets + tenant denylist. This engine can read
                     customer org data via support grants, so it gets the full stack.
  * ``Mode.LIGHT`` — Claude Code: secret-shape scan only. Dev sessions never touch
                     customer data, so PII/tenant checks would be needless plumbing;
                     a skill can still accidentally echo a secret from the codebase.

The module is pure/self-contained: it does not import the CRM or hit the network. The
tenant denylist is *injected* (a plain iterable of names) so callers wire it to the CRM
at their own layer and this module stays trivially testable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable

# Bump when the deterministic ruleset changes materially — recorded in the audit trail
# so a historical gate decision can be reproduced/re-reviewed.
RULESET_VERSION = "1.0.0"


class Mode(str, Enum):
    FULL = "full"    # SamurAI: PII + secrets + tenant denylist
    LIGHT = "light"  # Claude Code: secrets only


# --- Deterministic patterns -------------------------------------------------------
# Kept conservative to limit false positives; a match is a hard block, so over-eager
# rules would jam the review queue. High-recall on real secrets/PII is the priority.

_PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # US SSN: 3-2-4 with separators (bare 9-digit is too FP-prone to block on).
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # US EIN: 2-7 with a hyphen.
    "ein": re.compile(r"\b\d{2}-\d{7}\b"),
    # North-American phone numbers in common formats.
    "phone": re.compile(
        r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
    ),
    "ipv4": re.compile(
        r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)"
    ),
}

_SECRET_PATTERNS: dict[str, re.Pattern] = {
    "private_key_block": re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}\b"),
    # `api_key = "…"`, `password: '…'`, `SECRET_TOKEN=…` with a non-trivial value.
    "credential_assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|passw(?:or)?d|client[_-]?secret|"
        r"private[_-]?key|access[_-]?key)\b\s*[:=]\s*['\"]?[^\s'\"]{8,}",
    ),
}

# The public IPv4 ranges we DON'T want to block on (docs/examples/loopback/private).
_IP_ALLOW_PREFIXES = ("127.", "0.", "10.", "192.168.", "255.")


def _ip_is_ignorable(value: str) -> bool:
    if value.startswith(_IP_ALLOW_PREFIXES):
        return True
    if value.startswith("172."):
        try:
            second = int(value.split(".")[1])
            return 16 <= second <= 31  # 172.16.0.0/12 private range
        except (IndexError, ValueError):
            return False
    return False


@dataclass
class Finding:
    category: str          # "pii" | "secret" | "tenant"
    rule: str              # which pattern/name matched
    match_preview: str     # short, already-masked snippet for the audit log
    field: str = ""        # which artifact field it was found in


@dataclass
class ScanResult:
    findings: list[Finding] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return bool(self.findings)


def _mask(s: str) -> str:
    """Mask a matched value so the finding itself never becomes a new leak."""
    s = s.strip()
    if len(s) <= 4:
        return "*" * len(s)
    return f"{s[:2]}{'*' * max(1, len(s) - 4)}{s[-2:]}"


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def scan_text(
    text: str,
    *,
    mode: Mode = Mode.FULL,
    tenant_names: Iterable[str] | None = None,
    field_name: str = "",
) -> ScanResult:
    """Scan a single string for private content. Pure; no I/O."""
    result = ScanResult()
    if not text:
        return result

    # Secrets — always checked (both modes).
    for rule, pat in _SECRET_PATTERNS.items():
        for m in pat.finditer(text):
            result.findings.append(
                Finding("secret", rule, _mask(m.group(0)), field_name)
            )

    if mode is Mode.FULL:
        # PII — full mode only.
        for rule, pat in _PII_PATTERNS.items():
            for m in pat.finditer(text):
                val = m.group(0)
                if rule == "ipv4" and _ip_is_ignorable(val):
                    continue
                result.findings.append(Finding("pii", rule, _mask(val), field_name))

        # Tenant/customer-name denylist — full mode only, injected list.
        if tenant_names:
            haystack = _normalize(text)
            for name in tenant_names:
                needle = _normalize(name)
                if not needle:
                    continue
                # whole-token match to avoid substring false positives
                if re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack):
                    result.findings.append(
                        Finding("tenant", name, _mask(name), field_name)
                    )

    return result


# Every artifact field that crosses out to GitHub. Branch names are validated
# separately (they must be a constrained slug, not content-derived) — see
# ``validate_branch_name`` below.
CROSSING_FIELDS = (
    "name",
    "description",
    "body",
    "pr_title",
    "pr_body",
    "pr_comment",
    "commit_message",
)


@dataclass
class GateResult:
    blocked: bool
    findings: list[Finding]
    mode: Mode
    ruleset_version: str = RULESET_VERSION


def scan_artifacts(
    artifacts: dict[str, str],
    *,
    mode: Mode = Mode.FULL,
    tenant_names: Iterable[str] | None = None,
) -> GateResult:
    """Scan every crossing artifact field. Unknown keys are scanned too (fail-safe)."""
    tenant_names = list(tenant_names or [])
    all_findings: list[Finding] = []
    for fname, value in artifacts.items():
        if not isinstance(value, str):
            continue
        res = scan_text(value, mode=mode, tenant_names=tenant_names, field_name=fname)
        all_findings.extend(res.findings)
    return GateResult(blocked=bool(all_findings), findings=all_findings, mode=mode)


_BRANCH_SLUG_RE = re.compile(r"^skill-draft/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                             r"[0-9a-f]{4}-[0-9a-f]{12}$")


def validate_branch_name(branch: str) -> bool:
    """Branch names must be a fixed ``skill-draft/<uuid4>`` slug — never derived from
    content — so they cannot themselves leak a tenant/customer name."""
    return bool(_BRANCH_SLUG_RE.match(branch or ""))


def build_audit_record(
    artifacts: dict[str, str],
    gate: GateResult,
    *,
    llm_verdict: str | None = None,
    llm_reason: str | None = None,
    judge_prompt_version: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Assemble the AU-2/AU-3-style audit record for one gate invocation.

    Records a hash of the input (not the input itself — the audit log must not become
    a leak), the deterministic result, the LLM verdict, and the ruleset/prompt versions
    so a historical decision can be reproduced. ``now`` is injectable for testing.
    """
    joined = "\x1e".join(
        f"{k}={artifacts[k]}" for k in sorted(artifacts) if isinstance(artifacts[k], str)
    )
    input_hash = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    ts = (now or datetime.now(timezone.utc)).isoformat()
    return {
        "ts": ts,
        "input_sha256": input_hash,
        "deterministic_blocked": gate.blocked,
        "deterministic_findings": [
            {"category": f.category, "rule": f.rule, "field": f.field,
             "preview": f.match_preview}
            for f in gate.findings
        ],
        "mode": gate.mode.value,
        "ruleset_version": gate.ruleset_version,
        "llm_verdict": llm_verdict,
        "llm_reason": llm_reason,
        "judge_prompt_version": judge_prompt_version,
    }
