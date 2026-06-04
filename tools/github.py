"""Tools for interacting with GitHub repositories via GitHub App auth."""

import os
import time
from itertools import islice

import httpx
from langchain_core.tools import tool

GITHUB_ORG = "virtualdojo-inc"

_issue_type_cache: dict[str, str] = {}


def _get_issue_type_id(name: str) -> str:
    """Resolve an org-level Issue Type name to its GraphQL ID. Cached per process."""
    if name in _issue_type_cache:
        return _issue_type_cache[name]
    data = _graphql(
        """query($org: String!) {
          organization(login: $org) {
            issueTypes(first: 20) { nodes { id name isEnabled } }
          }
        }""",
        {"org": GITHUB_ORG},
    )
    nodes = (data.get("organization") or {}).get("issueTypes", {}).get("nodes", [])
    for it in nodes:
        if it.get("isEnabled"):
            _issue_type_cache[it["name"]] = it["id"]
    if name not in _issue_type_cache:
        raise ValueError(
            f"Issue Type '{name}' is not enabled on org '{GITHUB_ORG}'. "
            f"Available: {sorted(_issue_type_cache)}"
        )
    return _issue_type_cache[name]


def _get_issue_node_id(repo: str, issue_number: int) -> str:
    """Resolve a repo+number to the issue's GraphQL node ID."""
    owner, name = repo.split("/", 1)
    data = _graphql(
        """query($owner: String!, $name: String!, $num: Int!) {
          repository(owner: $owner, name: $name) { issue(number: $num) { id } }
        }""",
        {"owner": owner, "name": name, "num": issue_number},
    )
    issue = (data.get("repository") or {}).get("issue")
    if not issue:
        raise ValueError(f"Issue #{issue_number} not found in {repo}")
    return issue["id"]

# Cache the token to avoid re-generating on every tool call within a request
_token_cache: dict = {"token": None, "expires_at": 0}


def _github_token() -> str:
    """Get a GitHub App installation access token (cached)."""
    from github import GithubIntegration

    now = time.time()
    if _token_cache["token"] and _token_cache["expires_at"] > now + 60:
        return _token_cache["token"]

    app_id = os.environ["GITHUB_APP_ID"]
    private_key = os.environ["GITHUB_APP_PRIVATE_KEY"]

    integration = GithubIntegration(app_id, private_key)
    installations = integration.get_installations()
    if not installations:
        raise RuntimeError("GitHub App is not installed on any organization.")

    access = integration.get_access_token(installations[0].id)
    _token_cache["token"] = access.token
    _token_cache["expires_at"] = access.expires_at.timestamp() if access.expires_at else now + 3600
    return access.token


def _github():
    """Authenticate as a GitHub App installation and return a Github client."""
    from github import Github

    return Github(_github_token())


def _graphql(query: str, variables: dict | None = None) -> dict:
    """Execute a GitHub GraphQL query."""
    resp = httpx.post(
        "https://api.github.com/graphql",
        json={"query": query, "variables": variables or {}},
        headers={"Authorization": f"Bearer {_github_token()}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        msgs = "; ".join(e.get("message", str(e)) for e in data["errors"])
        raise RuntimeError(f"GraphQL error: {msgs}")
    return data["data"]


@tool
def github_list_prs(repo: str, state: str = "open") -> str:
    """List pull requests for a GitHub repository.

    Args:
        repo: Repository in 'owner/repo' format.
        state: PR state — 'open', 'closed', or 'all'.
    """
    pulls = list(
        islice(_github().get_repo(repo).get_pulls(state=state, sort="updated"), 10)
    )

    if not pulls:
        return f"No {state} PRs found in {repo}."

    lines = []
    for p in pulls:
        lines.append(f"#{p.number} {p.title} ({p.state}) by {p.user.login}")
    return "\n".join(lines)


@tool
def github_get_pr_details(repo: str, pr_number: int) -> str:
    """Get details of a specific pull request including changed files.

    Args:
        repo: Repository in 'owner/repo' format.
        pr_number: The PR number.
    """
    pr = _github().get_repo(repo).get_pull(pr_number)
    files = [f.filename for f in pr.get_files()]
    return (
        f"Title: {pr.title}\n"
        f"Author: {pr.user.login}\n"
        f"State: {pr.state}\n"
        f"Branch: {pr.head.ref} -> {pr.base.ref}\n"
        f"Files changed ({len(files)}): {', '.join(files)}"
    )


@tool
def github_list_recent_commits(
    repo: str, branch: str = "main", count: int = 10
) -> str:
    """List recent commits on a branch.

    Args:
        repo: Repository in 'owner/repo' format.
        branch: Branch name (default 'main').
        count: Number of commits to return (default 10).
    """
    commits = list(islice(_github().get_repo(repo).get_commits(sha=branch), count))

    lines = []
    for c in commits:
        short_sha = c.sha[:7]
        msg = c.commit.message.splitlines()[0]
        author = c.commit.author.name
        lines.append(f"{short_sha} {msg} — {author}")
    return "\n".join(lines)


@tool
def github_get_commit_diff(repo: str, sha: str) -> str:
    """Get the diff (changed files and patches) for a specific commit.

    Useful for understanding what changed in a commit without syncing the full repo.

    Args:
        repo: Repository in 'owner/repo' format.
        sha: Full or short commit SHA.
    """
    commit = _github().get_repo(repo).get_commit(sha)
    lines = [
        f"**{commit.sha[:7]}** {commit.commit.message.splitlines()[0]}",
        f"Author: {commit.commit.author.name}",
        f"Date: {commit.commit.author.date.isoformat()}",
        f"Files changed: {len(commit.files)}  (+{commit.stats.additions} -{commit.stats.deletions})",
        "",
    ]
    for f in commit.files:
        lines.append(f"### {f.filename} ({f.status}, +{f.additions} -{f.deletions})")
        if f.patch:
            # Truncate large patches to avoid blowing up context
            patch = f.patch if len(f.patch) <= 2000 else f.patch[:2000] + "\n... (truncated)"
            lines.append(f"```diff\n{patch}\n```")
        lines.append("")

    result = "\n".join(lines)
    # Hard cap to avoid overwhelming the LLM
    if len(result) > 15000:
        result = result[:15000] + "\n\n... (output truncated, too many changes)"
    return result


@tool
def github_list_issues(repo: str, state: str = "open", count: int = 10) -> str:
    """List issues for a GitHub repository.

    Args:
        repo: Repository in 'owner/repo' format.
        state: Issue state — 'open', 'closed', or 'all'.
        count: Number of issues to return (default 10).
    """
    issues = _github().get_repo(repo).get_issues(state=state, sort="updated")
    # Filter out pull requests (GitHub API returns PRs as issues too)
    results = []
    for issue in issues:
        if issue.pull_request is not None:
            continue
        labels = ", ".join(l.name for l in issue.labels) if issue.labels else "none"
        results.append(
            f"#{issue.number} {issue.title} ({issue.state}) "
            f"by {issue.user.login} | labels: {labels}"
        )
        if len(results) >= count:
            break

    return "\n".join(results) if results else f"No {state} issues found in {repo}."


@tool
def github_search_issues(
    query: str,
    repo: str = "virtualdojo-inc/virtualdojo",
    state: str = "all",
    count: int = 10,
) -> str:
    """Full-text search GitHub issues and PRs in a repository.

    Use this as the FIRST step in troubleshooting: if a closed bug issue
    already matches the symptom, you can cite the fix directly instead of
    re-investigating. This uses GitHub's /search/issues endpoint which
    indexes title, body, and comments.

    Args:
        query: Free-text search (e.g. "api key activities endpoint",
            "500 error auth", "token expired entra"). Do NOT include "repo:"
            — it is added automatically.
        repo: Repository in 'owner/repo' format. Defaults to the main data
            service.
        state: 'open', 'closed', or 'all' (default). Prefer 'all' for
            troubleshooting — closed issues often contain the fix.
        count: Max results to return (default 10).
    """
    # Strip any repo: qualifier the model might have included
    clean_query = " ".join(
        tok for tok in query.split() if not tok.lower().startswith("repo:")
    )
    full_query = f"repo:{repo} {clean_query}"
    if state in ("open", "closed"):
        full_query += f" state:{state}"

    try:
        results = _github().search_issues(query=full_query)
    except Exception as e:
        return f"Search failed: {type(e).__name__}: {e}"

    # PyGitHub's PaginatedList raises IndexError on the empty case instead
    # of returning []. Catch it and treat as "no matches" — otherwise every
    # zero-result search crashes with "list index out of range".
    try:
        issues = list(results[:count])
    except IndexError:
        issues = []

    lines = []
    for issue in issues:
        labels = ", ".join(l.name for l in issue.labels) if issue.labels else "none"
        kind = "PR" if issue.pull_request is not None else "issue"
        lines.append(
            f"#{issue.number} [{kind}, {issue.state}] {issue.title} "
            f"(labels: {labels})"
        )
    if not lines:
        return f"No issues or PRs matched '{clean_query}' in {repo}."
    header = f"Top {len(lines)} matches for '{clean_query}' in {repo}:\n"
    return header + "\n".join(lines)


@tool
def github_get_issue_details(repo: str, issue_number: int) -> str:
    """Get details of a specific GitHub issue including comments.

    Args:
        repo: Repository in 'owner/repo' format.
        issue_number: The issue number.
    """
    issue = _github().get_repo(repo).get_issue(issue_number)
    labels = ", ".join(l.name for l in issue.labels) if issue.labels else "none"
    assignees = ", ".join(a.login for a in issue.assignees) if issue.assignees else "unassigned"

    result = (
        f"Title: {issue.title}\n"
        f"State: {issue.state}\n"
        f"Author: {issue.user.login}\n"
        f"Labels: {labels}\n"
        f"Assignees: {assignees}\n"
        f"Created: {issue.created_at}\n"
        f"Body:\n{issue.body or '(empty)'}"
    )

    comments = list(islice(issue.get_comments(), 5))
    if comments:
        result += "\n\nRecent comments:"
        for c in comments:
            result += f"\n- {c.user.login}: {c.body[:200]}"

    return result


VALID_ISSUE_TYPES = ("Bug", "Feature", "Task")

# Project #2 "VirtualDojo Development" is the org-wide backlog. Every new
# issue gets added here so triage queries (github_get_project_items) see
# them. Priority short codes map to the project's full option labels.
DEFAULT_PROJECT_NUMBER = 2
PRIORITY_LABELS = {
    "P0": "P0 - Critical",
    "P1": "P1 - High",
    "P2": "P2 - Medium",
    "P3": "P3 - Low",
}


def _add_issue_to_dev_project(issue_node_id: str, priority: str) -> str:
    """Add an issue to Project #2 and set its Priority field. Returns the
    project item id. Raises on any GraphQL failure so the caller can
    surface a warning."""
    proj = _graphql(
        """query($org: String!, $num: Int!) {
          organization(login: $org) {
            projectV2(number: $num) {
              id
              fields(first: 30) {
                nodes {
                  ... on ProjectV2SingleSelectField {
                    id name options { id name }
                  }
                }
              }
            }
          }
        }""",
        {"org": GITHUB_ORG, "num": DEFAULT_PROJECT_NUMBER},
    )
    project = proj["organization"]["projectV2"]
    project_id = project["id"]
    priority_field = next(
        (f for f in project["fields"]["nodes"] if f.get("name") == "Priority"),
        None,
    )
    if not priority_field:
        raise RuntimeError("Priority field not found on Project #2")
    target_label = PRIORITY_LABELS[priority]
    option = next(
        (o for o in priority_field["options"] if o["name"] == target_label),
        None,
    )
    if not option:
        available = [o["name"] for o in priority_field["options"]]
        raise RuntimeError(
            f"Priority option {target_label!r} not found. Available: {available}"
        )

    add_result = _graphql(
        """mutation($projectId: ID!, $contentId: ID!) {
          addProjectV2ItemById(input: {projectId: $projectId, contentId: $contentId}) {
            item { id }
          }
        }""",
        {"projectId": project_id, "contentId": issue_node_id},
    )
    item_id = add_result["addProjectV2ItemById"]["item"]["id"]

    _graphql(
        """mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
          updateProjectV2ItemFieldValue(input: {
            projectId: $projectId
            itemId: $itemId
            fieldId: $fieldId
            value: { singleSelectOptionId: $optionId }
          }) { projectV2Item { id } }
        }""",
        {
            "projectId": project_id,
            "itemId": item_id,
            "fieldId": priority_field["id"],
            "optionId": option["id"],
        },
    )
    return item_id


@tool
def github_create_issue(
    repo: str,
    title: str,
    body: str,
    issue_type: str,
    priority: str,
    labels: str = "",
) -> str:
    """Create a new GitHub issue, classify it, and add it to the org backlog.

    Every new issue is auto-added to Project #2 ('VirtualDojo Development')
    and assigned a Priority. This is the org standard — there is no path
    to create an untyped or un-prioritized issue.

    Args:
        repo: Repository in 'owner/repo' format.
        title: The issue title.
        body: The issue body/description (supports markdown).
        issue_type: REQUIRED. One of 'Bug', 'Feature', or 'Task'. Pick
            'Bug' for unexpected behavior or regressions, 'Feature' for
            new functionality requests, 'Task' for specific pieces of
            work that aren't bugs or features.
        priority: REQUIRED. One of 'P0', 'P1', 'P2', 'P3' — maps to
            P0-Critical / P1-High / P2-Medium / P3-Low on Project #2.
            Use P0 for incidents and outages, P1 for clear regressions
            blocking a workflow, P2 for the default backlog priority,
            P3 for nice-to-haves.
        labels: Comma-separated label names to apply (optional).
    """
    if issue_type not in VALID_ISSUE_TYPES:
        raise ValueError(
            f"issue_type must be one of {VALID_ISSUE_TYPES}, got {issue_type!r}"
        )
    if priority not in PRIORITY_LABELS:
        raise ValueError(
            f"priority must be one of {tuple(PRIORITY_LABELS)}, got {priority!r}"
        )

    repo_obj = _github().get_repo(repo)
    label_list = [l.strip() for l in labels.split(",") if l.strip()] if labels else []
    issue = repo_obj.create_issue(title=title, body=body, labels=label_list)
    issue_node_id = issue.raw_data["node_id"]

    warnings: list[str] = []

    try:
        type_id = _get_issue_type_id(issue_type)
        _graphql(
            """mutation($issueId: ID!, $typeId: ID!) {
              updateIssueIssueType(input: {issueId: $issueId, issueTypeId: $typeId}) {
                issue { number }
              }
            }""",
            {"issueId": issue_node_id, "typeId": type_id},
        )
    except Exception as e:
        warnings.append(
            f"failed to set Issue Type '{issue_type}': {e} "
            "(call github_set_issue_type to fix)"
        )

    try:
        _add_issue_to_dev_project(issue_node_id, priority)
    except Exception as e:
        warnings.append(
            f"failed to add to Project #{DEFAULT_PROJECT_NUMBER} with "
            f"priority {priority}: {e} "
            "(call github_add_item_to_project + github_update_item_field to fix)"
        )

    warning_block = ""
    if warnings:
        warning_block = "\nWARNINGS:\n" + "\n".join(f"  - {w}" for w in warnings)

    return (
        f"Created issue #{issue.number} (type: {issue_type}, "
        f"priority: {priority}, project: #{DEFAULT_PROJECT_NUMBER}): "
        f"{issue.title}\nURL: {issue.html_url}{warning_block}"
    )


@tool
def github_get_issue_type(repo: str, issue_number: int) -> str:
    """Get the current Issue Type set on a GitHub issue.

    Returns 'Bug', 'Feature', 'Task', or 'none' (when no type is set). Use this during
    triage to check whether an issue already has a type before deciding to set one.

    Args:
        repo: Repository in 'owner/repo' format.
        issue_number: The issue number.
    """
    owner, name = repo.split("/", 1)
    data = _graphql(
        """query($owner: String!, $name: String!, $num: Int!) {
          repository(owner: $owner, name: $name) {
            issue(number: $num) { issueType { name } }
          }
        }""",
        {"owner": owner, "name": name, "num": issue_number},
    )
    issue = (data.get("repository") or {}).get("issue")
    if issue is None:
        raise ValueError(f"Issue #{issue_number} not found in {repo}")
    issue_type = issue.get("issueType")
    return issue_type["name"] if issue_type else "none"


@tool
def github_set_issue_type(repo: str, issue_number: int, issue_type: str) -> str:
    """Set the Issue Type on an existing GitHub issue.

    Issue Type is an org-level classification (Bug / Feature / Task) that travels with
    the issue across all projects, search results, and views. Use this for triage —
    it's the canonical signal for whether something is a bug, not the project Status field.

    Args:
        repo: Repository in 'owner/repo' format.
        issue_number: The issue number.
        issue_type: One of 'Bug', 'Feature', or 'Task'.
    """
    issue_id = _get_issue_node_id(repo, issue_number)
    type_id = _get_issue_type_id(issue_type)
    _graphql(
        """mutation($issueId: ID!, $typeId: ID!) {
          updateIssueIssueType(input: {issueId: $issueId, issueTypeId: $typeId}) {
            issue { number }
          }
        }""",
        {"issueId": issue_id, "typeId": type_id},
    )
    return f"Set Issue Type to '{issue_type}' on {repo}#{issue_number}"


@tool
def github_list_workflow_runs(
    repo: str, status: str = "", count: int = 10
) -> str:
    """List recent GitHub Actions workflow runs for a repository.

    Args:
        repo: Repository in 'owner/repo' format.
        status: Filter by status — 'completed', 'in_progress', 'queued', 'failure', 'success', or '' for all.
        count: Number of runs to return (default 10).
    """
    repo_obj = _github().get_repo(repo)
    kwargs = {}
    if status:
        kwargs["status"] = status
    runs = list(islice(repo_obj.get_workflow_runs(**kwargs), count))

    if not runs:
        return f"No workflow runs found in {repo}."

    lines = []
    for run in runs:
        duration = ""
        if run.created_at and run.updated_at:
            delta = run.updated_at - run.created_at
            duration = f" ({delta.total_seconds():.0f}s)"
        lines.append(
            f"{run.name} | {run.conclusion or run.status} | "
            f"run_id={run.id} #{run.run_number} on {run.head_branch}{duration} | "
            f"{run.created_at.strftime('%Y-%m-%d %H:%M')}"
        )
    return "\n".join(lines)


@tool
def github_get_workflow_run_details(repo: str, run_id: int) -> str:
    """Get details of a specific GitHub Actions workflow run, including failed job info.

    Args:
        repo: Repository in 'owner/repo' format.
        run_id: The workflow run ID (a large number like 14358032881, NOT the run number). Use the run_id from github_list_workflow_runs output.
    """
    repo_obj = _github().get_repo(repo)
    run = repo_obj.get_workflow_run(run_id)

    result = (
        f"Workflow: {run.name}\n"
        f"Status: {run.status} | Conclusion: {run.conclusion}\n"
        f"Branch: {run.head_branch}\n"
        f"Triggered by: {run.event} ({run.triggering_actor.login if run.triggering_actor else 'unknown'})\n"
        f"Run #: {run.run_number}\n"
        f"URL: {run.html_url}\n"
    )

    jobs = list(run.jobs())
    if jobs:
        result += "\nJobs:"
        for job in jobs:
            result += f"\n  {job.name}: {job.conclusion or job.status}"
            if job.conclusion == "failure":
                # Show failed steps
                for step in job.steps:
                    if step.conclusion == "failure":
                        result += f"\n    FAILED step: {step.name}"

    return result


# ---------------------------------------------------------------------------
# GitHub Projects V2 tools (GraphQL)
# ---------------------------------------------------------------------------

@tool
def github_list_projects() -> str:
    """List all GitHub Projects in the virtualdojo-inc organization."""
    data = _graphql(
        """
        query($org: String!) {
          organization(login: $org) {
            projectsV2(first: 20, orderBy: {field: UPDATED_AT, direction: DESC}) {
              nodes { number title shortDescription closed url }
            }
          }
        }
        """,
        {"org": GITHUB_ORG},
    )
    projects = data["organization"]["projectsV2"]["nodes"]
    if not projects:
        return "No projects found."
    lines = []
    for p in projects:
        status = "closed" if p["closed"] else "open"
        desc = f" — {p['shortDescription']}" if p.get("shortDescription") else ""
        lines.append(f"#{p['number']} {p['title']} ({status}){desc}")
    return "\n".join(lines)


@tool
def github_get_project_items(project_number: int, count: int = 20) -> str:
    """List items (issues, PRs, drafts) in a GitHub Project with their field values.

    Args:
        project_number: The project number (e.g. 1, 2).
        count: Number of items to return (default 20, max 100).
    """
    count = min(count, 100)
    data = _graphql(
        """
        query($org: String!, $num: Int!, $count: Int!) {
          organization(login: $org) {
            projectV2(number: $num) {
              title
              fields(first: 30) {
                nodes {
                  ... on ProjectV2SingleSelectField { id name options { id name } }
                  ... on ProjectV2Field { id name }
                  ... on ProjectV2IterationField { id name }
                }
              }
              items(first: $count, orderBy: {field: POSITION, direction: ASC}) {
                nodes {
                  id
                  fieldValues(first: 15) {
                    nodes {
                      ... on ProjectV2ItemFieldSingleSelectValue {
                        name
                        field { ... on ProjectV2SingleSelectField { name } }
                      }
                      ... on ProjectV2ItemFieldTextValue {
                        text
                        field { ... on ProjectV2Field { name } }
                      }
                      ... on ProjectV2ItemFieldNumberValue {
                        number
                        field { ... on ProjectV2Field { name } }
                      }
                      ... on ProjectV2ItemFieldDateValue {
                        date
                        field { ... on ProjectV2Field { name } }
                      }
                      ... on ProjectV2ItemFieldIterationValue {
                        title
                        field { ... on ProjectV2IterationField { name } }
                      }
                    }
                  }
                  content {
                    ... on Issue { title number state url }
                    ... on PullRequest { title number state url }
                    ... on DraftIssue { title body }
                  }
                }
              }
            }
          }
        }
        """,
        {"org": GITHUB_ORG, "num": project_number, "count": count},
    )
    project = data["organization"]["projectV2"]
    items = project["items"]["nodes"]
    if not items:
        return f"No items in project '{project['title']}'."

    lines = [f"Project: {project['title']} ({len(items)} items)\n"]
    for item in items:
        content = item.get("content") or {}
        title = content.get("title", "(draft)")
        number = content.get("number")
        state = content.get("state", "")

        label = f"#{number} " if number else ""
        state_str = f" [{state}]" if state else ""

        # Collect field values
        fields = []
        for fv in item["fieldValues"]["nodes"]:
            fname = ""
            fval = ""
            if "name" in fv and "field" in fv:
                fname = fv["field"].get("name", "")
                fval = fv["name"]
            elif "text" in fv and "field" in fv:
                fname = fv["field"].get("name", "")
                fval = fv["text"]
            elif "number" in fv and "field" in fv:
                fname = fv["field"].get("name", "")
                fval = str(fv["number"])
            elif "date" in fv and "field" in fv:
                fname = fv["field"].get("name", "")
                fval = fv["date"]
            elif "title" in fv and "field" in fv:
                fname = fv["field"].get("name", "")
                fval = fv["title"]
            if fname and fval:
                fields.append(f"{fname}: {fval}")

        field_str = f" | {', '.join(fields)}" if fields else ""
        lines.append(f"- {label}{title}{state_str}{field_str}")
        lines.append(f"  item_id: {item['id']}")

    return "\n".join(lines)


@tool
def github_create_draft_issue(
    project_number: int, title: str, body: str = ""
) -> str:
    """Create a new draft issue in a GitHub Project.

    Args:
        project_number: The project number.
        title: Title for the draft issue.
        body: Optional body/description.
    """
    # First get the project node ID
    data = _graphql(
        """
        query($org: String!, $num: Int!) {
          organization(login: $org) {
            projectV2(number: $num) { id title }
          }
        }
        """,
        {"org": GITHUB_ORG, "num": project_number},
    )
    project_id = data["organization"]["projectV2"]["id"]

    result = _graphql(
        """
        mutation($projectId: ID!, $title: String!, $body: String) {
          addProjectV2DraftIssue(input: {
            projectId: $projectId
            title: $title
            body: $body
          }) {
            projectItem { id }
          }
        }
        """,
        {"projectId": project_id, "title": title, "body": body or None},
    )
    item_id = result["addProjectV2DraftIssue"]["projectItem"]["id"]
    return f"Created draft issue '{title}' in project (item_id: {item_id})"


@tool
def github_add_item_to_project(project_number: int, repo: str, issue_number: int) -> str:
    """Add an existing issue or PR to a GitHub Project.

    Args:
        project_number: The project number.
        repo: Repository in 'owner/repo' format.
        issue_number: The issue or PR number to add.
    """
    # Get project ID
    data = _graphql(
        """
        query($org: String!, $num: Int!) {
          organization(login: $org) {
            projectV2(number: $num) { id title }
          }
        }
        """,
        {"org": GITHUB_ORG, "num": project_number},
    )
    project_id = data["organization"]["projectV2"]["id"]

    # Get issue/PR node ID
    owner, name = repo.split("/")
    data = _graphql(
        """
        query($owner: String!, $name: String!, $number: Int!) {
          repository(owner: $owner, name: $name) {
            issueOrPullRequest(number: $number) {
              ... on Issue { id title }
              ... on PullRequest { id title }
            }
          }
        }
        """,
        {"owner": owner, "name": name, "number": issue_number},
    )
    content = data["repository"]["issueOrPullRequest"]
    if not content:
        return f"Issue/PR #{issue_number} not found in {repo}."

    result = _graphql(
        """
        mutation($projectId: ID!, $contentId: ID!) {
          addProjectV2ItemById(input: {
            projectId: $projectId
            contentId: $contentId
          }) {
            item { id }
          }
        }
        """,
        {"projectId": project_id, "contentId": content["id"]},
    )
    item_id = result["addProjectV2ItemById"]["item"]["id"]
    return f"Added '{content['title']}' to project (item_id: {item_id})"


@tool
def github_update_item_field(
    project_number: int, item_id: str, field_name: str, value: str
) -> str:
    """Update a field value on a project item (e.g. Status, Priority).

    Args:
        project_number: The project number.
        item_id: The project item ID (from github_get_project_items output).
        field_name: The field name to update (e.g. 'Status', 'Priority').
        value: The value to set (must match an existing option for single-select fields).
    """
    # Get project ID and field definitions
    data = _graphql(
        """
        query($org: String!, $num: Int!) {
          organization(login: $org) {
            projectV2(number: $num) {
              id
              fields(first: 30) {
                nodes {
                  ... on ProjectV2SingleSelectField {
                    id name options { id name }
                  }
                  ... on ProjectV2Field { id name dataType }
                }
              }
            }
          }
        }
        """,
        {"org": GITHUB_ORG, "num": project_number},
    )
    project = data["organization"]["projectV2"]
    project_id = project["id"]

    # Find the field
    target_field = None
    for field in project["fields"]["nodes"]:
        if field.get("name", "").lower() == field_name.lower():
            target_field = field
            break

    if not target_field:
        available = [f["name"] for f in project["fields"]["nodes"] if f.get("name")]
        return f"Field '{field_name}' not found. Available fields: {', '.join(available)}"

    field_id = target_field["id"]
    data_type = target_field.get("dataType")

    # For single-select fields, find the option ID
    if "options" in target_field:
        option = next(
            (o for o in target_field["options"] if o["name"].lower() == value.lower()),
            None,
        )
        if not option:
            available = [o["name"] for o in target_field["options"]]
            return f"Value '{value}' not found for '{field_name}'. Options: {', '.join(available)}"

        _graphql(
            """
            mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
              updateProjectV2ItemFieldValue(input: {
                projectId: $projectId
                itemId: $itemId
                fieldId: $fieldId
                value: { singleSelectOptionId: $optionId }
              }) { projectV2Item { id } }
            }
            """,
            {
                "projectId": project_id,
                "itemId": item_id,
                "fieldId": field_id,
                "optionId": option["id"],
            },
        )
        return f"Updated '{field_name}' to '{option['name']}'"

    if data_type == "DATE":
        import re

        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return (
                f"Date '{value}' for field '{field_name}' must be in YYYY-MM-DD format"
            )
        _graphql(
            """
            mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $date: Date!) {
              updateProjectV2ItemFieldValue(input: {
                projectId: $projectId
                itemId: $itemId
                fieldId: $fieldId
                value: { date: $date }
              }) { projectV2Item { id } }
            }
            """,
            {
                "projectId": project_id,
                "itemId": item_id,
                "fieldId": field_id,
                "date": value,
            },
        )
        return f"Updated '{field_name}' to '{value}'"

    if data_type == "NUMBER":
        try:
            number = float(value)
        except ValueError:
            return f"Value '{value}' for field '{field_name}' is not a number"
        _graphql(
            """
            mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $number: Float!) {
              updateProjectV2ItemFieldValue(input: {
                projectId: $projectId
                itemId: $itemId
                fieldId: $fieldId
                value: { number: $number }
              }) { projectV2Item { id } }
            }
            """,
            {
                "projectId": project_id,
                "itemId": item_id,
                "fieldId": field_id,
                "number": number,
            },
        )
        return f"Updated '{field_name}' to '{value}'"

    # For text fields
    _graphql(
        """
        mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $text: String!) {
          updateProjectV2ItemFieldValue(input: {
            projectId: $projectId
            itemId: $itemId
            fieldId: $fieldId
            value: { text: $text }
          }) { projectV2Item { id } }
        }
        """,
        {
            "projectId": project_id,
            "itemId": item_id,
            "fieldId": field_id,
            "text": value,
        },
    )
    return f"Updated '{field_name}' to '{value}'"


@tool
def github_close_issue(repo: str, issue_number: int, reason: str = "") -> str:
    """Close a GitHub issue. Only use this to clean up duplicates or issues created in error.

    Args:
        repo: Repository in 'owner/repo' format.
        issue_number: The issue number to close.
        reason: Brief reason for closing (added as a comment before closing).
    """
    repo_obj = _github().get_repo(repo)
    issue = repo_obj.get_issue(number=issue_number)
    if reason:
        issue.create_comment(f"Closing: {reason}")
    issue.edit(state="closed")
    return f"Closed issue #{issue_number}: {issue.title}"


@tool
def github_edit_issue(
    repo: str, issue_number: int, title: str = "", body: str = ""
) -> str:
    """Edit the title and/or body of an existing GitHub issue.

    Use this to fix a typo, clarify the description, or add detail to an issue
    that already exists. Leave a field blank ('') to keep it unchanged. This
    edits the issue text only — to change the Bug/Feature/Task classification
    use github_set_issue_type, and to close an issue use github_close_issue.

    Args:
        repo: Repository in 'owner/repo' format.
        issue_number: The issue number to edit.
        title: New title. Leave blank to keep the current title.
        body: New body/description (supports markdown). Leave blank to keep
            the current body.
    """
    if not title and not body:
        return "Nothing to edit: provide a new title and/or body."
    repo_obj = _github().get_repo(repo)
    issue = repo_obj.get_issue(number=issue_number)
    kwargs = {}
    if title:
        kwargs["title"] = title
    if body:
        kwargs["body"] = body
    issue.edit(**kwargs)
    changed = " and ".join(k for k in ("title", "body") if k in kwargs)
    return f"Edited {changed} on {repo}#{issue_number}: {issue.title}"


# All project tools for easy import
PROJECT_TOOLS = [
    github_list_projects,
    github_get_project_items,
    github_create_draft_issue,
    github_add_item_to_project,
    github_update_item_field,
    github_get_issue_type,
    github_set_issue_type,
]
