"""Cold-start backfill for the troubleshooting DB.

Lists closed bug issues on a GitHub repo, uses Flash-lite to extract a
structured TroubleshootingStep from each, and saves to the
("troubleshooting", "virtualdojo") namespace in the shared LangMem store.

Run once locally after the embeddings path is verified:

    GCP_PROJECT_ID=virtualdojo-samurai \\
    SAMURAI_DATA_DIR=/path/to/local/data \\
    python scripts/backfill_troubleshooting.py --repo virtualdojo-inc/virtualdojo --limit 30

Idempotent: skips issues that already have a saved step (matched by
github_issue number in the stored value).

Requires:
- `gcloud auth application-default login` (ADC for Vertex embeddings)
- GITHUB_APP_ID and GITHUB_APP_PRIVATE_KEY env vars (or whatever _github_token() uses)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional

# Make `tools.*` and `memory.*` importable when this script is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill")


EXTRACTION_PROMPT = """You are extracting a reusable troubleshooting pattern from a closed GitHub bug issue.

Read the issue title, body, and the most recent comments. Return a SINGLE JSON object with these fields — no prose, no markdown, just the JSON:

{{
  "symptom": "one-line user-facing description of the bug",
  "winning_hypothesis": "the actual diagnosis / root cause",
  "discriminating_evidence": "what evidence or tool call proved the diagnosis (file:line, log snippet, etc.)",
  "fix_location": "file:line, config setting, or infrastructure location where the fix was applied",
  "fix_description": "one-line description of the change",
  "hypotheses_ruled_out": ["dead-end hypotheses that were investigated and ruled out, as a list"],
  "skip": false
}}

If the issue is too vague, has no clear root cause, or describes something other than a bug (e.g. a feature request), return:
{{"skip": true, "reason": "why you're skipping"}}

Issue:
Title: {title}
Labels: {labels}
Body:
{body}

Top comments:
{comments}
"""


def _extract_step_from_issue(issue, llm) -> Optional[dict]:
    """Use Flash-lite to extract a structured troubleshooting step from an issue.

    Returns None if the LLM decides to skip this issue or extraction fails.
    """
    from langchain_core.messages import HumanMessage

    labels = ", ".join(l.name for l in issue.labels) if issue.labels else "none"
    body = (issue.body or "(no body)")[:4000]
    try:
        comments = list(issue.get_comments()[:5])
    except Exception:
        comments = []
    comment_text = "\n".join(
        f"- {c.user.login}: {(c.body or '')[:500]}" for c in comments
    ) or "(no comments)"

    prompt = EXTRACTION_PROMPT.format(
        title=issue.title,
        labels=labels,
        body=body,
        comments=comment_text,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
    except Exception as e:
        logger.warning("Extraction call failed for issue #%s: %s", issue.number, e)
        return None

    # Gemini 3.x returns content as a list of blocks (thinking + text).
    # Mirror agent._extract_text so we pull the actual text out.
    content = response.content
    if isinstance(content, list):
        text = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    elif isinstance(content, str):
        text = content
    else:
        text = ""
    # Strip common code-fence wrappers if the model added them
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        # Remove a possible leading "json" marker
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: find the first { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            logger.warning(
                "Could not parse JSON from extraction for #%s. Raw response (first 400 chars): %r",
                issue.number,
                text[:400],
            )
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            logger.warning(
                "JSON salvage failed for #%s: %s. Raw response (first 400 chars): %r",
                issue.number,
                e,
                text[:400],
            )
            return None

    if not isinstance(data, dict):
        return None
    if data.get("skip"):
        logger.info(
            "Skipping issue #%s: %s",
            issue.number,
            data.get("reason", "(no reason given)"),
        )
        return None

    required = (
        "symptom",
        "winning_hypothesis",
        "discriminating_evidence",
        "fix_location",
        "fix_description",
    )
    for key in required:
        if not data.get(key):
            logger.info("Skipping issue #%s: missing %s", issue.number, key)
            return None

    data.setdefault("hypotheses_ruled_out", [])
    return data


def _existing_issue_numbers(namespace) -> set[int]:
    """Return github_issue numbers already present in the store so we can skip them."""
    from memory import get_memory_store

    store = get_memory_store()
    existing: set[int] = set()
    # Search by a neutral query; LangMem InMemoryStore supports iterating via search
    # with a broad query and large limit. The store doesn't expose a list() directly.
    results = store.search(namespace, query="bug fix issue troubleshooting", limit=10000)
    for r in results:
        num = r.value.get("github_issue")
        if isinstance(num, int):
            existing.add(num)
    return existing


def backfill(
    repo: str,
    limit: int,
    state: str = "closed",
    labels: str = "bug",
    dry_run: bool = False,
) -> dict:
    from langchain_google_genai import ChatGoogleGenerativeAI

    from memory import persist_memories
    from tools.github import _github
    from tools.troubleshooting import TROUBLESHOOTING_NAMESPACE, _save_step

    # Use the same model + location as the main agent's flash path — known to work
    # on this project's Vertex config. Flash-lite is not reliably available at
    # location=global across all Vertex projects.
    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("BACKFILL_MODEL", "gemini-3.6-flash"),
        vertexai=True,
        project=os.environ.get("GCP_PROJECT_ID"),
        location=os.environ.get("GCP_LOCATION", "global"),
    )

    logger.info("Listing %s issues on %s (labels=%s, limit=%d)", state, repo, labels, limit)
    gh = _github().get_repo(repo)
    issue_iter = gh.get_issues(state=state, labels=[labels] if labels else [], sort="updated")

    already = _existing_issue_numbers(TROUBLESHOOTING_NAMESPACE)
    logger.info("Found %d existing troubleshooting entries linked to issues", len(already))

    stats = {"examined": 0, "saved": 0, "skipped_existing": 0, "skipped_by_llm": 0, "failed": 0}

    for issue in issue_iter:
        if stats["examined"] >= limit:
            break
        # Filter PRs (GitHub returns PRs as issues)
        if issue.pull_request is not None:
            continue
        stats["examined"] += 1

        if issue.number in already:
            logger.info("Skipping issue #%s (already backfilled)", issue.number)
            stats["skipped_existing"] += 1
            continue

        logger.info("Extracting from issue #%s: %s", issue.number, issue.title[:80])
        step = _extract_step_from_issue(issue, llm)
        if step is None:
            stats["skipped_by_llm"] += 1
            continue

        if dry_run:
            logger.info(
                "DRY-RUN would save for #%s: symptom=%r fix=%r @ %r",
                issue.number,
                step["symptom"][:80],
                step["fix_description"][:80],
                step["fix_location"][:80],
            )
            stats["saved"] += 1
            continue

        try:
            _save_step(
                symptom=step["symptom"],
                winning_hypothesis=step["winning_hypothesis"],
                discriminating_evidence=step["discriminating_evidence"],
                fix_location=step["fix_location"],
                fix_description=step["fix_description"],
                hypotheses_ruled_out=step.get("hypotheses_ruled_out") or [],
                repo=repo,
                github_issue=issue.number,
                source="from_issue",
            )
            stats["saved"] += 1
            time.sleep(0.5)  # Gentle pacing for Vertex quotas
        except Exception as e:
            logger.error("Save failed for #%s: %s", issue.number, e)
            stats["failed"] += 1

    if not dry_run:
        logger.info("Persisting memories to SQLite")
        persist_memories()

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="virtualdojo-inc/virtualdojo")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--state", default="closed", choices=["closed", "open", "all"])
    parser.add_argument("--labels", default="bug")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be saved without persisting.",
    )
    args = parser.parse_args()

    if not os.environ.get("GCP_PROJECT_ID"):
        logger.error("GCP_PROJECT_ID env var is required (for Vertex embeddings + LLM).")
        sys.exit(1)

    stats = backfill(
        repo=args.repo,
        limit=args.limit,
        state=args.state,
        labels=args.labels,
        dry_run=args.dry_run,
    )

    print()
    print("=" * 60)
    print("Backfill complete.")
    print(f"  Repo:            {args.repo}")
    print(f"  Examined:        {stats['examined']}")
    print(f"  Saved:           {stats['saved']}{' (DRY RUN)' if args.dry_run else ''}")
    print(f"  Skipped (existing): {stats['skipped_existing']}")
    print(f"  Skipped (LLM):   {stats['skipped_by_llm']}")
    print(f"  Failed:          {stats['failed']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
