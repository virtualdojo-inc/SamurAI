"""Ingest GitHub issues into the knowledge base (support scope).

Refreshes ``support/raw/github-issues/`` with new/changed issues from
``virtualdojo-inc/virtualdojo`` (one markdown file per issue, with provenance
frontmatter). Incremental via a stored ``since`` watermark. Runs IN-BOUNDARY on
samurai-bot; reuses ``tools/github.py`` auth. No LLM involved — purely mechanical.

raw/ is treated as immutable source: bucket object versioning (ON, no auto-purge)
preserves prior snapshots when an issue is re-fetched after it changes.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from kb import storage
from tools.github import _github

REPO = "virtualdojo-inc/virtualdojo"
RAW_PREFIX = "support/raw/github-issues/"
STATE_PATH = "support/raw/.state/github_last_sync.txt"

# High-signal secret patterns scrubbed from raw on the way in (README rule:
# "clean/secret-scan inputs on the way in").
_SECRET_RE = re.compile(
    r"(AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|gho_[A-Za-z0-9]{30,}|"
    r"github_pat_[A-Za-z0-9_]{30,}|sk-ant-[A-Za-z0-9-]{20,}|AIza[0-9A-Za-z_-]{30,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|xox[baprs]-[A-Za-z0-9-]{10,})"
)


def _scrub(text: str) -> tuple[str, int]:
    """Redact obvious secrets; return (clean_text, num_redacted)."""
    n = len(_SECRET_RE.findall(text or ""))
    return _SECRET_RE.sub("[REDACTED-SECRET]", text or ""), n


def _issue_md(issue) -> str:
    labels = ", ".join(l.name for l in issue.labels)
    body, redacted = _scrub(issue.body or "")
    fm = [
        "---",
        "source: github",
        f"repo: {REPO}",
        f"issue: {issue.number}",
        f"state: {issue.state}",
        f"title: {issue.title!r}",
        f"labels: [{labels}]",
        f"url: {issue.html_url}",
        f"created_at: {issue.created_at.isoformat() if issue.created_at else ''}",
        f"updated_at: {issue.updated_at.isoformat() if issue.updated_at else ''}",
        f"ingested_at: {datetime.now(timezone.utc).isoformat()}",
        f"secrets_redacted: {redacted}",
        "---",
        "",
        f"# {issue.title}",
        "",
        body,
    ]
    return "\n".join(fm)


def refresh_github_issues(limit: int | None = None) -> dict:
    """Refresh raw github-issues since the last watermark. Returns content-free stats."""
    last = storage.read_text(STATE_PATH)
    since = None
    if last:
        try:
            since = datetime.fromisoformat(last.strip())
        except ValueError:
            since = None

    repo = _github().get_repo(REPO)
    kwargs = {"state": "all", "sort": "updated", "direction": "desc"}
    if since:
        kwargs["since"] = since

    written = skipped_prs = redacted_total = 0
    newest_seen = since
    for issue in repo.get_issues(**kwargs):
        # get_issues returns PRs too; skip them.
        if issue.pull_request is not None:
            skipped_prs += 1
            continue
        md = _issue_md(issue)
        redacted_total += md.count("[REDACTED-SECRET]")
        storage.write_text(f"{RAW_PREFIX}issue-{issue.number}.md", md)
        written += 1
        if issue.updated_at and (newest_seen is None or issue.updated_at > newest_seen):
            newest_seen = issue.updated_at
        if limit and written >= limit:
            break

    if newest_seen:
        storage.write_text(STATE_PATH, newest_seen.isoformat(), content_type="text/plain")

    return {
        "source": "github",
        "issues_written": written,
        "prs_skipped": skipped_prs,
        "secrets_redacted": redacted_total,
        "since": since.isoformat() if since else None,
    }
