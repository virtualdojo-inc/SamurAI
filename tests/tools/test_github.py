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
    mock_gh.return_value.get_repo.return_value.get_commits.return_value.__getitem__ = lambda s, k: commits

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
        {"query": "api key activities", "repo": "Quote-ly/quotely-data-service"}
    )

    query_arg = mock_gh.return_value.search_issues.call_args.kwargs.get(
        "query"
    ) or mock_gh.return_value.search_issues.call_args.args[0]
    assert "repo:Quote-ly/quotely-data-service" in query_arg
    assert "api key activities" in query_arg


@patch("tools.github._github")
def test_search_issues_strips_repo_qualifier_if_model_adds_it(mock_gh):
    """If the model includes `repo:...` in the query we strip it — otherwise
    GitHub sees two repo: qualifiers and errors / narrows incorrectly."""
    from tools.github import github_search_issues

    mock_gh.return_value.search_issues.return_value = []
    github_search_issues.invoke(
        {"query": "repo:foo/bar some symptom", "repo": "Quote-ly/quotely-data-service"}
    )

    query_arg = mock_gh.return_value.search_issues.call_args.kwargs.get(
        "query"
    ) or mock_gh.return_value.search_issues.call_args.args[0]
    # Exactly one repo: qualifier, and it's the tool's, not the model's
    assert query_arg.count("repo:") == 1
    assert "repo:Quote-ly/quotely-data-service" in query_arg
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
        {"query": "api key", "repo": "Quote-ly/quotely-data-service"}
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


@patch("tools.github._github")
def test_search_issues_handles_paginated_list_index_error(mock_gh):
    """PyGitHub's PaginatedList raises IndexError on the empty case when sliced
    (not []) — caught Devin's Monday session where 10 parallel searches all
    crashed with 'list index out of range' for queries returning zero results."""
    from tools.github import github_search_issues

    paginated = MagicMock()
    paginated.__getitem__ = MagicMock(side_effect=IndexError("list index out of range"))
    mock_gh.return_value.search_issues.return_value = paginated

    out = github_search_issues.invoke(
        {"query": "nothing matches this", "repo": "Quote-ly/quotely-data-service"}
    )
    assert "No issues" in out
    assert "IndexError" not in out
    assert "list index out of range" not in out
