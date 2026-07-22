"""Tests for tools/repo_sync.py — local repo sync and code reading tools."""

import os
import shutil
import subprocess
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_repo_sync_state():
    """Reset module-level state between tests so the last-synced cache and
    the per-(repo, branch) lock dict don't leak across tests."""
    import tools.repo_sync as rs

    rs._last_synced_branch.clear()
    with rs._sync_locks_guard:
        rs._sync_locks.clear()
    yield
    rs._last_synced_branch.clear()
    with rs._sync_locks_guard:
        rs._sync_locks.clear()


# --- sync_repo ---


def test_sync_repo_rejects_unlisted_repo():
    from tools.repo_sync import sync_repo

    result = sync_repo.invoke({"repo": "Evil/hacker-repo", "branch": "main"})
    assert "not a whitelisted repo" in result


@patch("tools.github._github_token", return_value="fake-token")
@patch("tools.repo_sync._get_remote_sha", return_value=None)
def test_sync_repo_handles_unreachable_branch(mock_sha, mock_token):
    from tools.repo_sync import sync_repo

    result = sync_repo.invoke(
        {"repo": "virtualdojo-inc/virtualdojo", "branch": "nonexistent"}
    )
    assert "Could not reach" in result


@patch("tools.github._github_token", return_value="fake-token")
@patch("tools.repo_sync._get_local_sha", return_value="abc12345")
@patch("tools.repo_sync._get_remote_sha", return_value="abc12345")
def test_sync_repo_skips_when_up_to_date(mock_remote, mock_local, mock_token):
    from tools.repo_sync import sync_repo

    result = sync_repo.invoke(
        {"repo": "virtualdojo-inc/virtualdojo", "branch": "main"}
    )
    assert "Already up to date" in result
    assert "abc12345" in result


@patch("subprocess.run")
@patch("tools.repo_sync._get_local_sha", return_value=None)
@patch("tools.repo_sync._get_remote_sha", return_value="def67890")
def test_sync_repo_clones_when_missing(mock_remote, mock_local, mock_run):
    from tools.repo_sync import sync_repo

    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    with patch("tools.github._github_token", return_value="fake-token"):
        result = sync_repo.invoke(
            {"repo": "virtualdojo-inc/virtualdojo", "branch": "main"}
        )

    assert "Synced successfully" in result
    assert "def67890" in result
    # Verify git clone was called with --depth 1
    clone_call = [c for c in mock_run.call_args_list if "clone" in str(c)]
    assert len(clone_call) > 0


@patch("subprocess.run")
@patch("tools.repo_sync._get_local_sha", return_value="old111")
@patch("tools.repo_sync._get_remote_sha", return_value="new222")
def test_sync_repo_reclones_when_stale(mock_remote, mock_local, mock_run):
    from tools.repo_sync import sync_repo

    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    with patch("tools.github._github_token", return_value="fake-token"):
        result = sync_repo.invoke(
            {"repo": "virtualdojo-inc/virtualdojo", "branch": "development"}
        )

    assert "Synced successfully" in result


@patch("subprocess.run")
@patch("tools.repo_sync._get_local_sha", return_value=None)
@patch("tools.repo_sync._get_remote_sha", return_value="abc123")
def test_sync_repo_handles_clone_failure(mock_remote, mock_local, mock_run):
    from tools.repo_sync import sync_repo

    mock_run.return_value = MagicMock(
        returncode=128, stdout="", stderr="fatal: repository not found"
    )

    with patch("tools.github._github_token", return_value="fake-token"):
        result = sync_repo.invoke(
            {"repo": "virtualdojo-inc/virtualdojo", "branch": "main"}
        )

    assert "Error cloning" in result


# --- Repo map ---


def _make_fake_repo(local_dir):
    os.makedirs(os.path.join(local_dir, "app", "services"), exist_ok=True)
    os.makedirs(os.path.join(local_dir, "node_modules", "junk"), exist_ok=True)
    with open(os.path.join(local_dir, "main.py"), "w") as f:
        f.write("# entry\n")
    with open(os.path.join(local_dir, "app", "config.py"), "w") as f:
        f.write("X = 1\n")
    with open(os.path.join(local_dir, "app", "services", "svc.py"), "w") as f:
        f.write("Y = 2\n")
    with open(os.path.join(local_dir, "node_modules", "junk", "big.js"), "w") as f:
        f.write("junk\n")


def test_build_repo_map_two_levels_with_counts(tmp_path):
    from tools.repo_sync import _build_repo_map

    _make_fake_repo(str(tmp_path))
    result = _build_repo_map(str(tmp_path))

    assert "app/ (2 files)" in result
    assert "app/services/ (1 files)" in result
    assert "Top-level files: main.py" in result
    # Noise dirs are skipped entirely
    assert "node_modules" not in result


def test_build_repo_map_missing_dir_returns_empty():
    from tools.repo_sync import _build_repo_map

    assert _build_repo_map("/tmp/definitely/not/a/dir") == ""


def test_sync_repo_up_to_date_includes_repo_map():
    """The most common sync result ('Already up to date') carries the repo map
    so every investigator gets navigation hints without extra tool calls."""
    import tools.repo_sync as rs

    repo = "virtualdojo-inc/virtualdojo"
    local_dir = rs._repo_dir(repo, "main")
    _make_fake_repo(local_dir)
    rs._repo_map_cache.clear()

    try:
        with (
            patch("tools.github._github_token", return_value="fake-token"),
            patch("tools.repo_sync._get_remote_sha", return_value="same-sha"),
            patch("tools.repo_sync._get_local_sha", return_value="same-sha"),
        ):
            result = rs.sync_repo.invoke({"repo": repo, "branch": "main"})

        assert "Already up to date" in result
        assert "Repo map" in result
        assert "app/ (2 files)" in result
    finally:
        shutil.rmtree(local_dir)
        rs._repo_map_cache.clear()


def test_repo_map_cached_by_sha():
    import tools.repo_sync as rs

    repo = "virtualdojo-inc/virtualdojo"
    local_dir = rs._repo_dir(repo, "main")
    _make_fake_repo(local_dir)
    rs._repo_map_cache.clear()

    try:
        first = rs._repo_map_text(repo, "main", "sha-1")
        assert "app/" in first
        # Change the tree; same SHA must return the cached map (no rebuild)
        with open(os.path.join(local_dir, "app", "new.py"), "w") as f:
            f.write("Z = 3\n")
        assert rs._repo_map_text(repo, "main", "sha-1") == first
        # New SHA rebuilds
        assert "app/ (3 files)" in rs._repo_map_text(repo, "main", "sha-2")
    finally:
        shutil.rmtree(local_dir)
        rs._repo_map_cache.clear()


# --- read_repo_file ---


def test_read_repo_file_rejects_unlisted_repo():
    from tools.repo_sync import read_repo_file

    result = read_repo_file.invoke(
        {"file_path": "main.py", "repo": "Evil/repo", "branch": "main"}
    )
    assert "not a whitelisted repo" in result


def test_read_repo_file_not_synced():
    from tools.repo_sync import read_repo_file

    result = read_repo_file.invoke(
        {"file_path": "main.py", "repo": "virtualdojo-inc/virtualdojo", "branch": "nonexistent-branch-xyz"}
    )
    assert "not synced yet" in result.lower() or "Call sync_repo" in result


def test_read_repo_file_reads_content(tmp_path):
    from tools.repo_sync import read_repo_file, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)

    # Create a fake repo with a file
    os.makedirs(local_dir, exist_ok=True)
    test_file = os.path.join(local_dir, "test.py")
    with open(test_file, "w") as f:
        f.write("print('hello world')")

    try:
        result = read_repo_file.invoke(
            {"file_path": "test.py", "repo": repo, "branch": branch}
        )
        assert "hello world" in result
    finally:
        os.remove(test_file)
        os.rmdir(local_dir)


def test_read_repo_file_not_found(tmp_path):
    from tools.repo_sync import read_repo_file, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    try:
        result = read_repo_file.invoke(
            {"file_path": "nonexistent.py", "repo": repo, "branch": branch}
        )
        assert "File not found" in result
    finally:
        os.rmdir(local_dir)


def test_read_repo_file_truncates_large_files(tmp_path):
    from tools.repo_sync import read_repo_file, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    test_file = os.path.join(local_dir, "big.txt")
    with open(test_file, "w") as f:
        f.write("x" * 60000)

    try:
        result = read_repo_file.invoke(
            {"file_path": "big.txt", "repo": repo, "branch": branch}
        )
        assert "truncated" in result
        assert len(result) < 55000
    finally:
        os.remove(test_file)
        os.rmdir(local_dir)


# --- read_repo_file_range ---


def test_read_repo_file_range_rejects_unlisted_repo():
    from tools.repo_sync import read_repo_file_range

    result = read_repo_file_range.invoke(
        {
            "file_path": "main.py",
            "start_line": 1,
            "end_line": 5,
            "repo": "Evil/repo",
            "branch": "main",
        }
    )
    assert "not a whitelisted repo" in result


def test_read_repo_file_range_rejects_invalid_range():
    from tools.repo_sync import read_repo_file_range

    # start_line < 1
    result_zero = read_repo_file_range.invoke(
        {
            "file_path": "x.py",
            "start_line": 0,
            "end_line": 5,
            "repo": "virtualdojo-inc/virtualdojo",
            "branch": "main",
        }
    )
    assert "invalid range" in result_zero

    # end < start
    result_backwards = read_repo_file_range.invoke(
        {
            "file_path": "x.py",
            "start_line": 10,
            "end_line": 5,
            "repo": "virtualdojo-inc/virtualdojo",
            "branch": "main",
        }
    )
    assert "invalid range" in result_backwards


def test_read_repo_file_range_not_synced():
    from tools.repo_sync import read_repo_file_range

    result = read_repo_file_range.invoke(
        {
            "file_path": "x.py",
            "start_line": 1,
            "end_line": 5,
            "repo": "virtualdojo-inc/virtualdojo",
            "branch": "nonexistent-branch-xyz",
        }
    )
    assert "not synced yet" in result.lower() or "Call sync_repo" in result


def test_read_repo_file_range_file_not_found(tmp_path):
    from tools.repo_sync import read_repo_file_range, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    try:
        result = read_repo_file_range.invoke(
            {
                "file_path": "missing.py",
                "start_line": 1,
                "end_line": 5,
                "repo": repo,
                "branch": branch,
            }
        )
        assert "File not found" in result
    finally:
        os.rmdir(local_dir)


def test_read_repo_file_range_rejects_directory(tmp_path):
    from tools.repo_sync import read_repo_file_range, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(os.path.join(local_dir, "subdir"), exist_ok=True)

    try:
        result = read_repo_file_range.invoke(
            {
                "file_path": "subdir",
                "start_line": 1,
                "end_line": 5,
                "repo": repo,
                "branch": branch,
            }
        )
        assert "is a directory" in result
    finally:
        import shutil
        shutil.rmtree(local_dir)


def test_read_repo_file_range_reads_requested_lines(tmp_path):
    from tools.repo_sync import read_repo_file_range, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    test_file = os.path.join(local_dir, "numbered.py")
    with open(test_file, "w") as f:
        f.write(
            "line one\n"
            "line two\n"
            "line three\n"
            "line four\n"
            "line five\n"
        )

    try:
        result = read_repo_file_range.invoke(
            {
                "file_path": "numbered.py",
                "start_line": 2,
                "end_line": 4,
                "repo": repo,
                "branch": branch,
            }
        )
        # Header shows selected range and total
        assert "numbered.py lines 2-4" in result
        assert "(of 5)" in result
        # Line numbers are 1-indexed and match the source
        assert "2: line two" in result
        assert "3: line three" in result
        assert "4: line four" in result
        # Lines outside the range are excluded
        assert "line one" not in result
        assert "line five" not in result
    finally:
        os.remove(test_file)
        os.rmdir(local_dir)


def test_read_repo_file_range_clamps_end_to_eof(tmp_path):
    """end_line beyond EOF should clamp, not error."""
    from tools.repo_sync import read_repo_file_range, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    test_file = os.path.join(local_dir, "short.py")
    with open(test_file, "w") as f:
        f.write("only line\n")

    try:
        result = read_repo_file_range.invoke(
            {
                "file_path": "short.py",
                "start_line": 1,
                "end_line": 9999,
                "repo": repo,
                "branch": branch,
            }
        )
        # Clamped to the actual 1 line available
        assert "lines 1-1" in result
        assert "(of 1)" in result
        assert "1: only line" in result
    finally:
        os.remove(test_file)
        os.rmdir(local_dir)


def test_read_repo_file_range_handles_open_exception(tmp_path):
    """Unexpected IO errors (e.g. permissions) should return an error string, not raise."""
    from tools.repo_sync import read_repo_file_range, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    test_file = os.path.join(local_dir, "boom.py")
    with open(test_file, "w") as f:
        f.write("hello\n")

    try:
        with patch("builtins.open", side_effect=PermissionError("nope")):
            result = read_repo_file_range.invoke(
                {
                    "file_path": "boom.py",
                    "start_line": 1,
                    "end_line": 5,
                    "repo": repo,
                    "branch": branch,
                }
            )
        assert "Error reading" in result
        assert "nope" in result
    finally:
        os.remove(test_file)
        os.rmdir(local_dir)


def test_read_repo_file_range_start_past_eof(tmp_path):
    """start_line past EOF should report that, not crash."""
    from tools.repo_sync import read_repo_file_range, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    test_file = os.path.join(local_dir, "tiny.py")
    with open(test_file, "w") as f:
        f.write("just one line\n")

    try:
        result = read_repo_file_range.invoke(
            {
                "file_path": "tiny.py",
                "start_line": 500,
                "end_line": 510,
                "repo": repo,
                "branch": branch,
            }
        )
        assert "past end of file" in result
        assert "1 lines" in result
    finally:
        os.remove(test_file)
        os.rmdir(local_dir)


# --- search_repo_code ---


def test_search_repo_code_rejects_unlisted_repo():
    from tools.repo_sync import search_repo_code

    result = search_repo_code.invoke(
        {"query": "test", "repo": "Evil/repo", "branch": "main"}
    )
    assert "not a whitelisted repo" in result


def test_search_repo_code_not_synced():
    from tools.repo_sync import search_repo_code

    result = search_repo_code.invoke(
        {"query": "test", "repo": "virtualdojo-inc/virtualdojo", "branch": "nonexistent-xyz"}
    )
    assert "not synced yet" in result.lower() or "Call sync_repo" in result


def test_search_repo_code_finds_matches(tmp_path):
    from tools.repo_sync import search_repo_code, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    test_file = os.path.join(local_dir, "app.py")
    with open(test_file, "w") as f:
        f.write("allow_origins=['*']\nother_line\n")

    try:
        result = search_repo_code.invoke(
            {"query": "allow_origins", "repo": repo, "branch": branch}
        )
        assert "allow_origins" in result
        assert "app.py" in result
    finally:
        os.remove(test_file)
        os.rmdir(local_dir)


def test_search_repo_code_includes_context_lines_by_default(tmp_path):
    """Default context_lines=2 should include 2 lines before/after each match (grep -C 2)."""
    from tools.repo_sync import search_repo_code, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    test_file = os.path.join(local_dir, "context.py")
    with open(test_file, "w") as f:
        f.write(
            "before_2 = 1\n"
            "before_1 = 2\n"
            "MATCH_HERE = 3\n"
            "after_1 = 4\n"
            "after_2 = 5\n"
            "far_away = 6\n"
        )

    try:
        result = search_repo_code.invoke(
            {"query": "MATCH_HERE", "repo": repo, "branch": branch}
        )
        assert "MATCH_HERE" in result
        # Default context_lines=2 pulls in neighbors
        assert "before_1" in result
        assert "before_2" in result
        assert "after_1" in result
        assert "after_2" in result
        # But not lines past the context window
        assert "far_away" not in result
    finally:
        os.remove(test_file)
        os.rmdir(local_dir)


def test_search_repo_code_context_lines_zero_disables_context(tmp_path):
    """context_lines=0 reproduces the old match-only output."""
    from tools.repo_sync import search_repo_code, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    test_file = os.path.join(local_dir, "nocontext.py")
    with open(test_file, "w") as f:
        f.write(
            "above_line = 1\n"
            "MATCH_HERE = 2\n"
            "below_line = 3\n"
        )

    try:
        result = search_repo_code.invoke(
            {
                "query": "MATCH_HERE",
                "repo": repo,
                "branch": branch,
                "context_lines": 0,
            }
        )
        assert "MATCH_HERE" in result
        assert "above_line" not in result
        assert "below_line" not in result
    finally:
        os.remove(test_file)
        os.rmdir(local_dir)


def test_search_repo_code_file_pattern_with_context(tmp_path):
    """file_pattern still works when context_lines is set."""
    from tools.repo_sync import search_repo_code, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    py_file = os.path.join(local_dir, "thing.py")
    txt_file = os.path.join(local_dir, "thing.txt")
    with open(py_file, "w") as f:
        f.write("TARGET_TOKEN = 1\n")
    with open(txt_file, "w") as f:
        f.write("TARGET_TOKEN in text\n")

    try:
        result = search_repo_code.invoke(
            {
                "query": "TARGET_TOKEN",
                "repo": repo,
                "branch": branch,
                "file_pattern": "*.py",
            }
        )
        assert "thing.py" in result
        assert "thing.txt" not in result
    finally:
        os.remove(py_file)
        os.remove(txt_file)
        os.rmdir(local_dir)


def test_search_repo_code_truncates_long_lines(tmp_path):
    """A single matched line >500 chars must be truncated, not returned whole.

    Reproduces the OOM root cause: a giant generated .md file blew memory
    because grep returned a multi-hundred-KB line in one shot.
    """
    from tools.repo_sync import (
        search_repo_code,
        _repo_dir,
        SEARCH_MAX_LINE_BYTES,
    )

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    test_file = os.path.join(local_dir, "huge.md")
    huge_line = "TARGET_X" + ("y" * 100_000)
    with open(test_file, "w") as f:
        f.write(huge_line + "\n")

    try:
        result = search_repo_code.invoke(
            {"query": "TARGET_X", "repo": repo, "branch": branch, "context_lines": 0}
        )
        assert "TARGET_X" in result
        assert "line truncated" in result
        # Result must be far smaller than the source line
        assert len(result) < SEARCH_MAX_LINE_BYTES + 5_000
    finally:
        os.remove(test_file)
        os.rmdir(local_dir)


def test_search_repo_code_byte_cap(tmp_path):
    """Total result is capped at ~50 KB even when many lines match."""
    from tools.repo_sync import (
        search_repo_code,
        _repo_dir,
        SEARCH_MAX_RESULT_BYTES,
    )

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    # 2000 matching lines, each ~250 chars after path prefix → comfortably > 50 KB
    test_file = os.path.join(local_dir, "many.txt")
    with open(test_file, "w") as f:
        for i in range(2000):
            f.write(f"NEEDLE_{i} " + ("z" * 240) + "\n")

    try:
        result = search_repo_code.invoke(
            {
                "query": "NEEDLE_",
                "repo": repo,
                "branch": branch,
                "context_lines": 0,
                "head_limit": 2000,
            }
        )
        assert len(result) <= SEARCH_MAX_RESULT_BYTES + 1_000  # plus the trailing note
        assert "Output truncated" in result
        assert "files_with_matches" in result
    finally:
        os.remove(test_file)
        os.rmdir(local_dir)


def test_search_repo_code_head_limit_and_offset(tmp_path):
    """head_limit + offset together provide pagination."""
    from tools.repo_sync import search_repo_code, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    test_file = os.path.join(local_dir, "page.txt")
    with open(test_file, "w") as f:
        for i in range(10):
            f.write(f"HIT_{i:02d}\n")

    try:
        page1 = search_repo_code.invoke(
            {
                "query": "HIT_",
                "repo": repo,
                "branch": branch,
                "context_lines": 0,
                "head_limit": 4,
                "offset": 0,
            }
        )
        assert "HIT_00" in page1
        assert "HIT_03" in page1
        assert "HIT_04" not in page1
        assert "offset=4" in page1  # pagination hint

        page2 = search_repo_code.invoke(
            {
                "query": "HIT_",
                "repo": repo,
                "branch": branch,
                "context_lines": 0,
                "head_limit": 4,
                "offset": 4,
            }
        )
        assert "HIT_00" not in page2
        assert "HIT_04" in page2
        assert "HIT_07" in page2
        assert "HIT_08" not in page2
        assert "offset=8" in page2
    finally:
        os.remove(test_file)
        os.rmdir(local_dir)


def test_search_repo_code_files_with_matches_mode(tmp_path):
    """output_mode='files_with_matches' returns paths only, no content."""
    from tools.repo_sync import search_repo_code, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    a = os.path.join(local_dir, "a.py")
    b = os.path.join(local_dir, "b.py")
    with open(a, "w") as f:
        f.write("WANTED secret content here\n")
    with open(b, "w") as f:
        f.write("WANTED other content\n")

    try:
        result = search_repo_code.invoke(
            {
                "query": "WANTED",
                "repo": repo,
                "branch": branch,
                "output_mode": "files_with_matches",
            }
        )
        assert "a.py" in result
        assert "b.py" in result
        # Content lines must NOT appear in paths-only mode
        assert "secret content" not in result
        assert "other content" not in result
    finally:
        os.remove(a)
        os.remove(b)
        os.rmdir(local_dir)


def test_search_repo_code_count_mode(tmp_path):
    """output_mode='count' returns per-file match counts; zero-match files filtered."""
    from tools.repo_sync import search_repo_code, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    hot = os.path.join(local_dir, "hot.py")
    cold = os.path.join(local_dir, "cold.py")
    with open(hot, "w") as f:
        f.write("FOO\nFOO\nFOO\n")
    with open(cold, "w") as f:
        f.write("nothing relevant\n")

    try:
        result = search_repo_code.invoke(
            {
                "query": "FOO",
                "repo": repo,
                "branch": branch,
                "output_mode": "count",
            }
        )
        assert "hot.py:3" in result
        # Files with 0 matches must be filtered out
        assert "cold.py" not in result
    finally:
        os.remove(hot)
        os.remove(cold)
        os.rmdir(local_dir)


def test_search_repo_code_rejects_invalid_output_mode():
    from tools.repo_sync import search_repo_code

    result = search_repo_code.invoke(
        {
            "query": "x",
            "repo": "virtualdojo-inc/virtualdojo",
            "branch": "main",
            "output_mode": "bogus",
        }
    )
    assert "invalid output_mode" in result


def test_search_repo_code_file_pattern_with_directory_prefix(tmp_path):
    """A file_pattern containing a path ('sub/dir/*.py') must scope the search
    to that directory instead of silently matching nothing.

    grep --include matches basenames only, so the old code turned any
    path-qualified pattern into a guaranteed 'No matches found' — observed in
    prod as false negatives for terms that existed in the repo.
    """
    from tools.repo_sync import search_repo_code, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    sub = os.path.join(local_dir, "alembic", "versions")
    os.makedirs(sub, exist_ok=True)

    inside = os.path.join(sub, "migration_a.py")
    outside = os.path.join(local_dir, "other.py")
    with open(inside, "w") as f:
        f.write("SCOPED_TOKEN = 1\n")
    with open(outside, "w") as f:
        f.write("SCOPED_TOKEN = 2\n")

    try:
        result = search_repo_code.invoke(
            {
                "query": "SCOPED_TOKEN",
                "repo": repo,
                "branch": branch,
                "file_pattern": "alembic/versions/*.py",
            }
        )
        assert "migration_a.py" in result
        assert "other.py" not in result
    finally:
        shutil.rmtree(local_dir)


def test_search_repo_code_file_pattern_with_missing_directory(tmp_path):
    """A path prefix pointing at a nonexistent directory returns a corrective
    error, not a misleading 'No matches found'."""
    from tools.repo_sync import search_repo_code, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    try:
        result = search_repo_code.invoke(
            {
                "query": "anything",
                "repo": repo,
                "branch": branch,
                "file_pattern": "no/such/dir/*.py",
            }
        )
        assert "directory 'no/such/dir' not found" in result
        assert "list_repo_files" in result
    finally:
        shutil.rmtree(local_dir)


def test_search_repo_code_file_pattern_rejects_parent_traversal(tmp_path):
    from tools.repo_sync import search_repo_code, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    try:
        result = search_repo_code.invoke(
            {
                "query": "x",
                "repo": repo,
                "branch": branch,
                "file_pattern": "../../../etc/*.conf",
            }
        )
        assert "must not contain '..'" in result
    finally:
        shutil.rmtree(local_dir)


def test_search_repo_code_glob_directory_segment_falls_back_to_root(tmp_path):
    """'app/**/*.py' — the '**' segment can't be used as a path scope; the
    search must still run (scoped to 'app', recursing) and find matches."""
    from tools.repo_sync import search_repo_code, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    deep = os.path.join(local_dir, "app", "services")
    os.makedirs(deep, exist_ok=True)

    with open(os.path.join(deep, "deep.py"), "w") as f:
        f.write("DEEP_TOKEN = 1\n")

    try:
        result = search_repo_code.invoke(
            {
                "query": "DEEP_TOKEN",
                "repo": repo,
                "branch": branch,
                "file_pattern": "app/**/*.py",
            }
        )
        assert "deep.py" in result
    finally:
        shutil.rmtree(local_dir)


def test_search_repo_code_no_match_mentions_directory_scope(tmp_path):
    """When the search was directory-scoped, the no-match message says so, so
    the model doesn't conclude the term is absent from the whole repo."""
    from tools.repo_sync import search_repo_code, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    sub = os.path.join(local_dir, "docs")
    os.makedirs(sub, exist_ok=True)
    with open(os.path.join(sub, "a.md"), "w") as f:
        f.write("nothing\n")

    try:
        result = search_repo_code.invoke(
            {
                "query": "ABSENT_TOKEN",
                "repo": repo,
                "branch": branch,
                "file_pattern": "docs/*.md",
            }
        )
        assert "No matches found" in result
        assert "under 'docs/'" in result
    finally:
        shutil.rmtree(local_dir)


def test_search_repo_code_no_matches(tmp_path):
    from tools.repo_sync import search_repo_code, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    test_file = os.path.join(local_dir, "empty.py")
    with open(test_file, "w") as f:
        f.write("# nothing here\n")

    try:
        result = search_repo_code.invoke(
            {"query": "ZZZZNOTFOUND", "repo": repo, "branch": branch}
        )
        assert "No matches found" in result
    finally:
        os.remove(test_file)
        os.rmdir(local_dir)


# --- list_repo_files ---


def test_list_repo_files_rejects_unlisted_repo():
    from tools.repo_sync import list_repo_files

    result = list_repo_files.invoke(
        {"path": "", "repo": "Evil/repo", "branch": "main"}
    )
    assert "not a whitelisted repo" in result


def test_list_repo_files_not_synced():
    from tools.repo_sync import list_repo_files

    result = list_repo_files.invoke(
        {"path": "", "repo": "virtualdojo-inc/virtualdojo", "branch": "nonexistent-xyz"}
    )
    assert "not synced yet" in result.lower() or "Call sync_repo" in result


def test_list_repo_files_shows_contents(tmp_path):
    from tools.repo_sync import list_repo_files, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(os.path.join(local_dir, "app"), exist_ok=True)

    with open(os.path.join(local_dir, "main.py"), "w") as f:
        f.write("# entry point")
    with open(os.path.join(local_dir, "requirements.txt"), "w") as f:
        f.write("flask\n")

    try:
        result = list_repo_files.invoke(
            {"path": "", "repo": repo, "branch": branch}
        )
        assert "main.py" in result
        assert "requirements.txt" in result
        assert "app/" in result
    finally:
        import shutil
        shutil.rmtree(local_dir)


def test_list_repo_files_hides_dotfiles(tmp_path):
    from tools.repo_sync import list_repo_files, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    with open(os.path.join(local_dir, ".git"), "w") as f:
        f.write("")
    with open(os.path.join(local_dir, "visible.py"), "w") as f:
        f.write("")

    try:
        result = list_repo_files.invoke(
            {"path": "", "repo": repo, "branch": branch}
        )
        assert ".git" not in result
        assert "visible.py" in result
    finally:
        import shutil
        shutil.rmtree(local_dir)


# --- Concurrency / branch-default regression tests (May 2026 Jason incident) ---


def test_sync_repo_serializes_concurrent_calls_to_same_branch(tmp_path, monkeypatch):
    """Concurrent sync_repo calls to the same (repo, branch) must not overlap.

    Without the lock, three parallel investigate() calls each fired their own
    sync_repo and raced on rm -rf + git clone, producing 'could not open
    .git/objects/pack/tmp_pack_...' and 'could not lock config file' errors.
    """
    import tools.repo_sync as rs

    monkeypatch.setattr(rs, "REPO_BASE_DIR", str(tmp_path))

    concurrent = 0
    max_concurrent = 0
    counter_lock = threading.Lock()

    def fake_run(cmd, *args, **kwargs):
        nonlocal concurrent, max_concurrent
        if cmd and cmd[0] in ("git", "rm"):
            with counter_lock:
                concurrent += 1
                max_concurrent = max(max_concurrent, concurrent)
            time.sleep(0.05)
            with counter_lock:
                concurrent -= 1
        return MagicMock(returncode=0, stdout="", stderr="")

    with (
        patch("tools.github._github_token", return_value="fake-token"),
        patch("tools.repo_sync._get_remote_sha", return_value="sha-abc"),
        patch("tools.repo_sync._get_local_sha", return_value=None),
        patch("subprocess.run", side_effect=fake_run),
    ):
        threads = [
            threading.Thread(
                target=lambda: rs.sync_repo.invoke(
                    {"repo": "virtualdojo-inc/virtualdojo", "branch": "development"}
                )
            )
            for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert max_concurrent == 1, (
        f"sync_repo on same (repo, branch) ran {max_concurrent}-way concurrently; "
        f"lock did not serialize"
    )


def test_sync_repo_allows_concurrent_calls_on_different_branches(tmp_path, monkeypatch):
    """The lock keys on (repo, branch) — different branches must not block each other."""
    import tools.repo_sync as rs

    monkeypatch.setattr(rs, "REPO_BASE_DIR", str(tmp_path))

    in_flight: set[str] = set()
    saw_overlap = False
    counter_lock = threading.Lock()

    def fake_run(cmd, *args, **kwargs):
        nonlocal saw_overlap
        branch_arg = None
        if cmd and cmd[0] == "git" and "clone" in cmd:
            try:
                branch_arg = cmd[cmd.index("--branch") + 1]
            except (ValueError, IndexError):
                branch_arg = "?"
        if branch_arg:
            with counter_lock:
                in_flight.add(branch_arg)
                if len(in_flight) >= 2:
                    saw_overlap = True
            time.sleep(0.05)
            with counter_lock:
                in_flight.discard(branch_arg)
        return MagicMock(returncode=0, stdout="", stderr="")

    with (
        patch("tools.github._github_token", return_value="fake-token"),
        patch("tools.repo_sync._get_remote_sha", return_value="sha-xyz"),
        patch("tools.repo_sync._get_local_sha", return_value=None),
        patch("subprocess.run", side_effect=fake_run),
    ):
        t_main = threading.Thread(
            target=lambda: rs.sync_repo.invoke(
                {"repo": "virtualdojo-inc/virtualdojo", "branch": "main"}
            )
        )
        t_dev = threading.Thread(
            target=lambda: rs.sync_repo.invoke(
                {"repo": "virtualdojo-inc/virtualdojo", "branch": "development"}
            )
        )
        t_main.start()
        t_dev.start()
        t_main.join()
        t_dev.join()

    assert saw_overlap, (
        "sync_repo on different branches did not overlap; lock keys may be too coarse"
    )


def test_sync_repo_records_last_synced_branch_on_clone(tmp_path, monkeypatch):
    import tools.repo_sync as rs

    monkeypatch.setattr(rs, "REPO_BASE_DIR", str(tmp_path))

    with (
        patch("tools.github._github_token", return_value="fake-token"),
        patch("tools.repo_sync._get_remote_sha", return_value="sha-new"),
        patch("tools.repo_sync._get_local_sha", return_value=None),
        patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="", stderr="")),
    ):
        rs.sync_repo.invoke(
            {"repo": "virtualdojo-inc/virtualdojo", "branch": "development"}
        )

    assert rs._last_synced_branch.get("virtualdojo-inc/virtualdojo") == "development"


def test_sync_repo_records_last_synced_branch_when_up_to_date():
    import tools.repo_sync as rs

    with (
        patch("tools.github._github_token", return_value="fake-token"),
        patch("tools.repo_sync._get_remote_sha", return_value="same"),
        patch("tools.repo_sync._get_local_sha", return_value="same"),
    ):
        result = rs.sync_repo.invoke(
            {"repo": "virtualdojo-inc/virtualdojo", "branch": "development"}
        )

    assert "Already up to date" in result
    assert rs._last_synced_branch.get("virtualdojo-inc/virtualdojo") == "development"


def test_search_repo_code_falls_back_to_last_synced_branch():
    """When branch is omitted, readers must default to the most-recently-synced branch.

    This was the proximate cause of Jason's 'no access to development' report:
    sync_repo(branch='development') and search_repo_code (no branch kwarg)
    fired in the same parallel batch, and search defaulted to 'main' which
    wasn't synced.
    """
    from tools.repo_sync import search_repo_code, _repo_dir
    import tools.repo_sync as rs

    repo = "virtualdojo-inc/virtualdojo"
    branch = "development"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    test_file = os.path.join(local_dir, "marker.py")
    with open(test_file, "w") as f:
        f.write("DEV_ONLY_TOKEN = 1\n")

    rs._last_synced_branch[repo] = branch

    try:
        result = search_repo_code.invoke({"query": "DEV_ONLY_TOKEN", "repo": repo})
        assert "DEV_ONLY_TOKEN" in result
        assert "marker.py" in result
    finally:
        shutil.rmtree(local_dir)


def test_read_repo_file_falls_back_to_last_synced_branch():
    from tools.repo_sync import read_repo_file, _repo_dir
    import tools.repo_sync as rs

    repo = "virtualdojo-inc/virtualdojo"
    branch = "development"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    with open(os.path.join(local_dir, "x.py"), "w") as f:
        f.write("dev_marker\n")

    rs._last_synced_branch[repo] = branch

    try:
        result = read_repo_file.invoke({"file_path": "x.py", "repo": repo})
        assert "dev_marker" in result
    finally:
        shutil.rmtree(local_dir)


def test_not_synced_message_lists_actually_synced_branches():
    """The error must reflect filesystem state, not parrot the caller's branch.

    Previously it said "Call sync_repo(... branch='main') first" even when the
    caller had just synced 'development', which led the model to chase the
    wrong branch.
    """
    from tools.repo_sync import search_repo_code, _repo_dir

    repo = "virtualdojo-inc/virtualdojo"
    dev_dir = _repo_dir(repo, "development")
    # Make sure 'main' is NOT present for this repo, then create 'development'.
    main_dir = _repo_dir(repo, "main")
    if os.path.isdir(main_dir):
        shutil.rmtree(main_dir)
    os.makedirs(dev_dir, exist_ok=True)

    try:
        result = search_repo_code.invoke(
            {"query": "anything", "repo": repo, "branch": "main"}
        )
        assert "not synced" in result.lower()
        assert "development" in result, (
            f"not-synced message should list locally synced branches; got: {result!r}"
        )
    finally:
        shutil.rmtree(dev_dir)


def test_not_synced_message_handles_repo_with_no_local_branches():
    from tools.repo_sync import read_repo_file, _repo_dir

    repo = "virtualdojo-inc/Fedramp"
    local_dir = _repo_dir(repo, "main")
    repo_root = os.path.dirname(local_dir)
    if os.path.isdir(repo_root):
        shutil.rmtree(repo_root)

    result = read_repo_file.invoke(
        {"file_path": "README.md", "repo": repo, "branch": "main"}
    )
    assert "not synced" in result.lower()
    assert "none" in result
