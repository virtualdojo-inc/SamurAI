"""Tests for tools.github — PRs, PR details, and commits."""

from unittest.mock import MagicMock, patch


def _make_pr(number, title, state, login):
    pr = MagicMock()
    pr.number = number
    pr.title = title
    pr.state = state
    pr.user.login = login
    return pr


def _make_commit(sha, message, author_name):
    c = MagicMock()
    c.sha = sha
    c.commit.message = message
    c.commit.author.name = author_name
    return c


# --- github_list_prs ---


@patch("tools.github._github")
def test_list_prs_formats_output(mock_gh):
    from tools.github import github_list_prs

    prs = [
        _make_pr(42, "Fix login bug", "open", "alice"),
        _make_pr(43, "Add dashboard", "open", "bob"),
    ]
    mock_gh.return_value.get_repo.return_value.get_pulls.return_value.__getitem__ = lambda s, k: prs[k] if isinstance(k, int) else prs
    mock_gh.return_value.get_repo.return_value.get_pulls.return_value.__iter__ = lambda s: iter(prs)
    # Handle the [:10] slice
    mock_gh.return_value.get_repo.return_value.get_pulls.return_value.__getitem__ = lambda s, k: prs

    result = github_list_prs.invoke({"repo": "org/repo"})

    assert "#42" in result
    assert "Fix login bug" in result
    assert "alice" in result
    assert "#43" in result


@patch("tools.github._github")
def test_list_prs_no_results(mock_gh):
    from tools.github import github_list_prs

    mock_gh.return_value.get_repo.return_value.get_pulls.return_value.__getitem__ = lambda s, k: []
    mock_gh.return_value.get_repo.return_value.get_pulls.return_value.__bool__ = lambda s: False
    mock_gh.return_value.get_repo.return_value.get_pulls.return_value.__iter__ = lambda s: iter([])

    result = github_list_prs.invoke({"repo": "owner/repo"})
    assert "No open PRs" in result


@patch("tools.github._github")
def test_list_prs_state_forwarded(mock_gh):
    from tools.github import github_list_prs

    mock_gh.return_value.get_repo.return_value.get_pulls.return_value.__getitem__ = lambda s, k: []

    github_list_prs.invoke({"repo": "org/repo", "state": "closed"})
    mock_gh.return_value.get_repo.return_value.get_pulls.assert_called_with(
        state="closed", sort="updated"
    )


# --- github_get_pr_details ---


@patch("tools.github._github")
def test_pr_details_output_format(mock_gh):
    from tools.github import github_get_pr_details

    pr = MagicMock()
    pr.title = "Add auth"
    pr.user.login = "alice"
    pr.state = "open"
    pr.head.ref = "feature/auth"
    pr.base.ref = "main"
    file1 = MagicMock()
    file1.filename = "auth.py"
    file2 = MagicMock()
    file2.filename = "tests/test_auth.py"
    pr.get_files.return_value = [file1, file2]
    mock_gh.return_value.get_repo.return_value.get_pull.return_value = pr

    result = github_get_pr_details.invoke({"repo": "org/repo", "pr_number": 10})

    assert "Title: Add auth" in result
    assert "Author: alice" in result
    assert "State: open" in result
    assert "feature/auth -> main" in result


@patch("tools.github._github")
def test_pr_details_file_list(mock_gh):
    from tools.github import github_get_pr_details

    pr = MagicMock()
    pr.title = "X"
    pr.user.login = "x"
    pr.state = "open"
    pr.head.ref = "a"
    pr.base.ref = "b"
    f1, f2 = MagicMock(), MagicMock()
    f1.filename = "foo.py"
    f2.filename = "bar.py"
    pr.get_files.return_value = [f1, f2]
    mock_gh.return_value.get_repo.return_value.get_pull.return_value = pr

    result = github_get_pr_details.invoke({"repo": "org/repo", "pr_number": 5})

    assert "foo.py" in result
    assert "bar.py" in result
    assert "Files changed (2)" in result


# --- github_list_recent_commits ---


@patch("tools.github._github")
def test_list_commits_format(mock_gh):
    from tools.github import github_list_recent_commits

    commits = [
        _make_commit("abc1234567890", "Fix typo in readme", "alice"),
        _make_commit("def4567890123", "Add CI pipeline\n\ndetails here", "bob"),
    ]
    mock_gh.return_value.get_repo.return_value.get_commits.return_value.__iter__ = lambda s: iter(commits)

    result = github_list_recent_commits.invoke({"repo": "org/repo"})

    assert "abc1234" in result
    assert "Fix typo in readme" in result
    assert "alice" in result
    assert "def4567" in result
    # Multi-line message should only show first line
    assert "Add CI pipeline" in result
    assert "details here" not in result


@patch("tools.github._github")
def test_list_commits_default_branch_main(mock_gh):
    from tools.github import github_list_recent_commits

    mock_gh.return_value.get_repo.return_value.get_commits.return_value.__getitem__ = lambda s, k: []

    github_list_recent_commits.invoke({"repo": "org/repo"})
    mock_gh.return_value.get_repo.return_value.get_commits.assert_called_with(sha="main")


@patch("tools.github._github")
def test_list_commits_custom_branch(mock_gh):
    from tools.github import github_list_recent_commits

    mock_gh.return_value.get_repo.return_value.get_commits.return_value.__getitem__ = lambda s, k: []

    github_list_recent_commits.invoke({"repo": "org/repo", "branch": "develop"})
    mock_gh.return_value.get_repo.return_value.get_commits.assert_called_with(sha="develop")


# --- github_get_commit_diff ---


def _make_commit_detail(sha, message, author_name, files):
    c = MagicMock()
    c.sha = sha
    c.commit.message = message
    c.commit.author.name = author_name
    c.commit.author.date.isoformat.return_value = "2026-04-10T18:00:00"
    c.stats.additions = sum(f["additions"] for f in files)
    c.stats.deletions = sum(f["deletions"] for f in files)
    mock_files = []
    for f in files:
        mf = MagicMock()
        mf.filename = f["filename"]
        mf.status = f.get("status", "modified")
        mf.additions = f["additions"]
        mf.deletions = f["deletions"]
        mf.patch = f.get("patch", "")
        mock_files.append(mf)
    c.files = mock_files
    return c


@patch("tools.github._github")
def test_commit_diff_format(mock_gh):
    from tools.github import github_get_commit_diff

    commit = _make_commit_detail(
        "abc1234567890",
        "Fix login bug",
        "alice",
        [
            {"filename": "auth.py", "additions": 5, "deletions": 2, "patch": "+new line\n-old line"},
            {"filename": "tests/test_auth.py", "additions": 10, "deletions": 0, "patch": "+test code"},
        ],
    )
    mock_gh.return_value.get_repo.return_value.get_commit.return_value = commit

    result = github_get_commit_diff.invoke({"repo": "org/repo", "sha": "abc1234"})

    assert "abc1234" in result
    assert "Fix login bug" in result
    assert "alice" in result
    assert "auth.py" in result
    assert "tests/test_auth.py" in result
    assert "+new line" in result
    assert "Files changed: 2" in result
    assert "+15 -2" in result


@patch("tools.github._github")
def test_commit_diff_truncates_large_patch(mock_gh):
    from tools.github import github_get_commit_diff

    commit = _make_commit_detail(
        "def4567890123",
        "Big refactor",
        "bob",
        [{"filename": "big.py", "additions": 500, "deletions": 300, "patch": "x" * 3000}],
    )
    mock_gh.return_value.get_repo.return_value.get_commit.return_value = commit

    result = github_get_commit_diff.invoke({"repo": "org/repo", "sha": "def4567"})

    assert "truncated" in result
    assert "big.py" in result


@patch("tools.github._github")
def test_commit_diff_no_patch(mock_gh):
    from tools.github import github_get_commit_diff

    commit = _make_commit_detail(
        "aaa1111222233",
        "Binary file update",
        "carol",
        [{"filename": "image.png", "additions": 0, "deletions": 0, "status": "modified", "patch": None}],
    )
    mock_gh.return_value.get_repo.return_value.get_commit.return_value = commit

    result = github_get_commit_diff.invoke({"repo": "org/repo", "sha": "aaa1111"})

    assert "image.png" in result
    assert "```diff" not in result  # No patch block for None patch


# --- github_search_issues ---


def _make_issue(number, title, state="closed", labels=None, is_pr=False):
    issue = MagicMock()
    issue.number = number
    issue.title = title
    issue.state = state
    lbls = []
    for name in labels or []:
        lbl = MagicMock()
        lbl.name = name
        lbls.append(lbl)
    issue.labels = lbls
    issue.pull_request = MagicMock() if is_pr else None
    return issue


@patch("tools.github._github")
def test_search_issues_scopes_query_to_repo(mock_gh):
    """The tool must auto-inject repo: scope so the agent doesn't accidentally
    search across all of GitHub."""
    from tools.github import github_search_issues

    mock_gh.return_value.search_issues.return_value = []
    github_search_issues.invoke(
        {"query": "api key activities", "repo": "virtualdojo-inc/virtualdojo"}
    )

    query_arg = mock_gh.return_value.search_issues.call_args.kwargs.get(
        "query"
    ) or mock_gh.return_value.search_issues.call_args.args[0]
    assert "repo:virtualdojo-inc/virtualdojo" in query_arg
    assert "api key activities" in query_arg


@patch("tools.github._github")
def test_search_issues_strips_repo_qualifier_if_model_adds_it(mock_gh):
    """If the model includes `repo:...` in the query we strip it — otherwise
    GitHub sees two repo: qualifiers and errors / narrows incorrectly."""
    from tools.github import github_search_issues

    mock_gh.return_value.search_issues.return_value = []
    github_search_issues.invoke(
        {"query": "repo:foo/bar some symptom", "repo": "virtualdojo-inc/virtualdojo"}
    )

    query_arg = mock_gh.return_value.search_issues.call_args.kwargs.get(
        "query"
    ) or mock_gh.return_value.search_issues.call_args.args[0]
    # Exactly one repo: qualifier, and it's the tool's, not the model's
    assert query_arg.count("repo:") == 1
    assert "repo:virtualdojo-inc/virtualdojo" in query_arg
    assert "some symptom" in query_arg


@patch("tools.github._github")
def test_search_issues_formats_results(mock_gh):
    from tools.github import github_search_issues

    results = [
        _make_issue(522, "Activities endpoints reject API key", "closed", ["bug", "P2"]),
        _make_issue(480, "Login PR fix", "closed", ["bug"], is_pr=True),
    ]
    # PyGithub's PaginatedList supports slicing with [:N]
    paginated = MagicMock()
    paginated.__iter__ = lambda s: iter(results)
    paginated.__getitem__ = lambda s, k: results[k]
    mock_gh.return_value.search_issues.return_value = paginated

    out = github_search_issues.invoke(
        {"query": "api key", "repo": "virtualdojo-inc/virtualdojo"}
    )

    assert "#522" in out
    assert "Activities endpoints reject API key" in out
    assert "bug" in out
    assert "[issue, closed]" in out
    assert "[PR, closed]" in out


@patch("tools.github._github")
def test_search_issues_no_matches(mock_gh):
    from tools.github import github_search_issues

    paginated = MagicMock()
    paginated.__iter__ = lambda s: iter([])
    paginated.__getitem__ = lambda s, k: []
    mock_gh.return_value.search_issues.return_value = paginated

    out = github_search_issues.invoke({"query": "nonexistent", "repo": "org/repo"})
    assert "No issues" in out


@patch("tools.github._github")
def test_search_issues_handles_exception(mock_gh):
    """Rate-limit or auth errors should return a string, not raise."""
    from tools.github import github_search_issues

    mock_gh.return_value.search_issues.side_effect = RuntimeError("rate limited")
    out = github_search_issues.invoke({"query": "anything", "repo": "org/repo"})
    assert "Search failed" in out
    assert "rate limited" in out


@patch("tools.github._github")
def test_search_issues_state_filter(mock_gh):
    from tools.github import github_search_issues

    mock_gh.return_value.search_issues.return_value = []
    github_search_issues.invoke(
        {"query": "foo", "repo": "org/repo", "state": "closed"}
    )

    query_arg = mock_gh.return_value.search_issues.call_args.kwargs.get(
        "query"
    ) or mock_gh.return_value.search_issues.call_args.args[0]
    assert "state:closed" in query_arg


# --- Issue Types: _get_issue_type_id, _get_issue_node_id, github_set_issue_type ---


import pytest


@pytest.fixture(autouse=False)
def clear_issue_type_cache():
    """Reset the module-level issue type cache between tests."""
    from tools import github as gh_mod

    gh_mod._issue_type_cache.clear()
    yield
    gh_mod._issue_type_cache.clear()


@patch("tools.github._graphql")
def test_get_issue_type_id_resolves_and_caches(mock_graphql, clear_issue_type_cache):
    from tools.github import _get_issue_type_id

    mock_graphql.return_value = {
        "organization": {
            "issueTypes": {
                "nodes": [
                    {"id": "IT_BUG", "name": "Bug", "isEnabled": True},
                    {"id": "IT_FEAT", "name": "Feature", "isEnabled": True},
                    {"id": "IT_TASK", "name": "Task", "isEnabled": True},
                    {"id": "IT_OFF", "name": "Disabled", "isEnabled": False},
                ]
            }
        }
    }

    assert _get_issue_type_id("Bug") == "IT_BUG"
    assert _get_issue_type_id("Feature") == "IT_FEAT"
    # Second + third lookups must not refetch.
    assert mock_graphql.call_count == 1
    # Disabled types must not be exposed.
    with pytest.raises(ValueError, match="Disabled"):
        _get_issue_type_id("Disabled")


@patch("tools.github._graphql")
def test_get_issue_type_id_raises_for_unknown(mock_graphql, clear_issue_type_cache):
    from tools.github import _get_issue_type_id

    mock_graphql.return_value = {
        "organization": {
            "issueTypes": {
                "nodes": [{"id": "IT_BUG", "name": "Bug", "isEnabled": True}]
            }
        }
    }
    with pytest.raises(ValueError, match="Nonsense"):
        _get_issue_type_id("Nonsense")


@patch("tools.github._graphql")
def test_get_issue_node_id_returns_id(mock_graphql):
    from tools.github import _get_issue_node_id

    mock_graphql.return_value = {"repository": {"issue": {"id": "I_NODE"}}}
    assert _get_issue_node_id("virtualdojo-inc/virtualdojo", 423) == "I_NODE"

    sent_vars = mock_graphql.call_args.args[1]
    assert sent_vars == {"owner": "virtualdojo-inc", "name": "virtualdojo", "num": 423}


@patch("tools.github._graphql")
def test_get_issue_node_id_raises_when_missing(mock_graphql):
    from tools.github import _get_issue_node_id

    mock_graphql.return_value = {"repository": {"issue": None}}
    with pytest.raises(ValueError, match="not found"):
        _get_issue_node_id("virtualdojo-inc/virtualdojo", 9999)


@patch("tools.github._graphql")
def test_get_issue_type_returns_name_when_set(mock_graphql):
    from tools.github import github_get_issue_type

    mock_graphql.return_value = {
        "repository": {"issue": {"issueType": {"name": "Bug"}}}
    }
    result = github_get_issue_type.invoke(
        {"repo": "virtualdojo-inc/virtualdojo", "issue_number": 423}
    )
    assert result == "Bug"


@patch("tools.github._graphql")
def test_get_issue_type_returns_none_when_unset(mock_graphql):
    from tools.github import github_get_issue_type

    mock_graphql.return_value = {"repository": {"issue": {"issueType": None}}}
    result = github_get_issue_type.invoke(
        {"repo": "virtualdojo-inc/virtualdojo", "issue_number": 999}
    )
    assert result == "none"


@patch("tools.github._graphql")
def test_get_issue_type_raises_when_issue_missing(mock_graphql):
    from tools.github import github_get_issue_type

    mock_graphql.return_value = {"repository": {"issue": None}}
    with pytest.raises(ValueError, match="not found"):
        github_get_issue_type.invoke(
            {"repo": "virtualdojo-inc/virtualdojo", "issue_number": 99999}
        )


@patch("tools.github._graphql")
def test_set_issue_type_happy_path(mock_graphql, clear_issue_type_cache):
    from tools.github import github_set_issue_type

    # Three calls: node id lookup, type id lookup, the mutation.
    mock_graphql.side_effect = [
        {"repository": {"issue": {"id": "I_NODE"}}},
        {
            "organization": {
                "issueTypes": {
                    "nodes": [{"id": "IT_BUG", "name": "Bug", "isEnabled": True}]
                }
            }
        },
        {"updateIssueIssueType": {"issue": {"number": 423}}},
    ]

    result = github_set_issue_type.invoke(
        {"repo": "virtualdojo-inc/virtualdojo", "issue_number": 423, "issue_type": "Bug"}
    )

    assert "Bug" in result
    assert "#423" in result
    # Mutation should receive the resolved IDs.
    mutation_call = mock_graphql.call_args_list[2]
    assert mutation_call.args[1] == {"issueId": "I_NODE", "typeId": "IT_BUG"}


# --- github_create_issue with issue_type ---


def _project_lookup_response():
    """GraphQL stub for the Project #2 + Priority field lookup."""
    return {
        "organization": {
            "projectV2": {
                "id": "PRJ_2",
                "fields": {
                    "nodes": [
                        {
                            "id": "FIELD_PRIORITY",
                            "name": "Priority",
                            "options": [
                                {"id": "OPT_P0", "name": "P0 - Critical"},
                                {"id": "OPT_P1", "name": "P1 - High"},
                                {"id": "OPT_P2", "name": "P2 - Medium"},
                                {"id": "OPT_P3", "name": "P3 - Low"},
                            ],
                        }
                    ]
                },
            }
        }
    }


def test_create_issue_rejects_missing_type_at_schema_level():
    """The tool's schema makes issue_type required — calls without it must fail."""
    import pytest
    from pydantic_core import ValidationError
    from tools.github import github_create_issue

    with pytest.raises(ValidationError) as exc:
        github_create_issue.invoke(
            {"repo": "foo/bar", "title": "X", "body": "...", "priority": "P2"}
        )
    assert "issue_type" in str(exc.value)


def test_create_issue_rejects_missing_priority_at_schema_level():
    """The tool's schema makes priority required — calls without it must fail."""
    import pytest
    from pydantic_core import ValidationError
    from tools.github import github_create_issue

    with pytest.raises(ValidationError) as exc:
        github_create_issue.invoke(
            {"repo": "foo/bar", "title": "X", "body": "...", "issue_type": "Bug"}
        )
    assert "priority" in str(exc.value)


@patch("tools.github._graphql")
@patch("tools.github._github")
def test_create_issue_rejects_invalid_type_value(
    mock_gh, mock_graphql, clear_issue_type_cache
):
    import pytest
    from tools.github import github_create_issue

    with pytest.raises(ValueError, match="issue_type must be one of"):
        github_create_issue.invoke(
            {
                "repo": "foo/bar",
                "title": "X",
                "body": "...",
                "issue_type": "Story",
                "priority": "P2",
            }
        )


@patch("tools.github._graphql")
@patch("tools.github._github")
def test_create_issue_rejects_invalid_priority_value(
    mock_gh, mock_graphql, clear_issue_type_cache
):
    import pytest
    from tools.github import github_create_issue

    with pytest.raises(ValueError, match="priority must be one of"):
        github_create_issue.invoke(
            {
                "repo": "foo/bar",
                "title": "X",
                "body": "...",
                "issue_type": "Bug",
                "priority": "URGENT",
            }
        )


@patch("tools.github._graphql")
@patch("tools.github._github")
def test_create_issue_sets_type_and_adds_to_project(
    mock_gh, mock_graphql, clear_issue_type_cache
):
    from tools.github import github_create_issue

    issue = MagicMock()
    issue.number = 8
    issue.title = "Crash on load"
    issue.html_url = "https://github.com/foo/bar/issues/8"
    issue.raw_data = {"node_id": "I_NEW"}
    mock_gh.return_value.get_repo.return_value.create_issue.return_value = issue

    # Sequence: issue type lookup, set type, project lookup, add to project, set priority.
    mock_graphql.side_effect = [
        {
            "organization": {
                "issueTypes": {
                    "nodes": [{"id": "IT_BUG", "name": "Bug", "isEnabled": True}]
                }
            }
        },
        {"updateIssueIssueType": {"issue": {"number": 8}}},
        _project_lookup_response(),
        {"addProjectV2ItemById": {"item": {"id": "ITEM_8"}}},
        {"updateProjectV2ItemFieldValue": {"projectV2Item": {"id": "ITEM_8"}}},
    ]

    result = github_create_issue.invoke(
        {
            "repo": "foo/bar",
            "title": "Crash on load",
            "body": "stack trace",
            "issue_type": "Bug",
            "priority": "P1",
        }
    )

    assert "#8" in result
    assert "type: Bug" in result
    assert "priority: P1" in result
    assert "project: #2" in result
    assert "WARNING" not in result

    # Project add mutation got the issue node id and project id.
    add_call_vars = mock_graphql.call_args_list[3].args[1]
    assert add_call_vars == {"projectId": "PRJ_2", "contentId": "I_NEW"}
    # Priority update got the right field + option ids.
    set_priority_vars = mock_graphql.call_args_list[4].args[1]
    assert set_priority_vars["fieldId"] == "FIELD_PRIORITY"
    assert set_priority_vars["optionId"] == "OPT_P1"


@patch("tools.github._graphql")
@patch("tools.github._github")
def test_create_issue_warns_when_type_set_fails(
    mock_gh, mock_graphql, clear_issue_type_cache
):
    """If issue creation succeeds but type-setting fails, the issue must not be lost
    and the project tagging should still be attempted."""
    from tools.github import github_create_issue

    issue = MagicMock()
    issue.number = 9
    issue.title = "Crash"
    issue.html_url = "https://github.com/foo/bar/issues/9"
    issue.raw_data = {"node_id": "I_NEW"}
    mock_gh.return_value.get_repo.return_value.create_issue.return_value = issue

    # Type lookup fails, but project tagging should still run and succeed.
    mock_graphql.side_effect = [
        RuntimeError("GraphQL boom"),
        _project_lookup_response(),
        {"addProjectV2ItemById": {"item": {"id": "ITEM_9"}}},
        {"updateProjectV2ItemFieldValue": {"projectV2Item": {"id": "ITEM_9"}}},
    ]

    result = github_create_issue.invoke(
        {
            "repo": "foo/bar",
            "title": "Crash",
            "body": "...",
            "issue_type": "Bug",
            "priority": "P2",
        }
    )

    assert "#9" in result
    assert "WARNING" in result
    assert "GraphQL boom" in result
    # Project tagging still succeeded — no warning about that.
    assert "failed to add to Project" not in result


@patch("tools.github._graphql")
@patch("tools.github._github")
def test_create_issue_warns_when_project_add_fails(
    mock_gh, mock_graphql, clear_issue_type_cache
):
    """If project tagging fails, the issue is still created and typed — warn only."""
    from tools.github import github_create_issue

    issue = MagicMock()
    issue.number = 10
    issue.title = "X"
    issue.html_url = "https://github.com/foo/bar/issues/10"
    issue.raw_data = {"node_id": "I_NEW"}
    mock_gh.return_value.get_repo.return_value.create_issue.return_value = issue

    mock_graphql.side_effect = [
        {
            "organization": {
                "issueTypes": {
                    "nodes": [{"id": "IT_BUG", "name": "Bug", "isEnabled": True}]
                }
            }
        },
        {"updateIssueIssueType": {"issue": {"number": 10}}},
        RuntimeError("project lookup failed"),
    ]

    result = github_create_issue.invoke(
        {
            "repo": "foo/bar",
            "title": "X",
            "body": "...",
            "issue_type": "Bug",
            "priority": "P2",
        }
    )

    assert "#10" in result
    assert "type: Bug" in result
    assert "WARNING" in result
    assert "failed to add to Project #2" in result
    assert "project lookup failed" in result


@patch("tools.github._github")
def test_search_issues_handles_paginated_list_index_error(mock_gh):
    """PyGitHub's PaginatedList raises IndexError on the empty case when sliced
    (not [])  — caught Devin's Monday session where 10 parallel searches all
    crashed with 'list index out of range' for queries returning zero results."""
    from tools.github import github_search_issues

    paginated = MagicMock()
    paginated.__getitem__ = MagicMock(side_effect=IndexError("list index out of range"))
    mock_gh.return_value.search_issues.return_value = paginated

    out = github_search_issues.invoke(
        {"query": "nothing matches this", "repo": "virtualdojo-inc/virtualdojo"}
    )
    assert "No issues" in out
    assert "IndexError" not in out
    assert "list index out of range" not in out
