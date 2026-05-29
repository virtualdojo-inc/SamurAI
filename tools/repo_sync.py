"""Local repo sync tools — shallow clone repos to /tmp for fast code reading and search."""

import logging
import os
import subprocess
import threading

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

REPO_BASE_DIR = "/tmp/repos"

ALLOWED_REPOS = {
    "virtualdojo-inc/virtualdojo",
    "virtualdojo-inc/virtualdojo_cli",
    "virtualdojo-inc/SamurAI",
    "virtualdojo-inc/Fedramp",
}

# Per-(repo, branch) lock. LangGraph's ToolNode runs sync tools in worker
# threads, so concurrent sync_repo calls to the same path race on rm -rf +
# git clone — the loser sees "could not open .git/objects/pack/tmp_pack_..."
# or "could not lock config file". threading.Lock serializes them.
_sync_locks: dict[tuple[str, str], threading.Lock] = {}
_sync_locks_guard = threading.Lock()

# Most-recently-synced branch per repo. Lets readers default to the branch
# the caller just synced when the model omits the kwarg in a parallel batch.
_last_synced_branch: dict[str, str] = {}


def _get_sync_lock(repo: str, branch: str) -> threading.Lock:
    key = (repo, branch)
    with _sync_locks_guard:
        lock = _sync_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _sync_locks[key] = lock
        return lock


def _resolve_branch(repo: str, branch: str | None) -> str:
    """Default to the most-recently-synced branch for this repo, else 'main'."""
    if branch is not None:
        return branch
    return _last_synced_branch.get(repo, "main")


def _not_synced_message(repo: str, branch: str) -> str:
    """Tell the model what's actually synced, not a hardcoded 'main' suggestion."""
    repo_root = os.path.join(REPO_BASE_DIR, repo.split("/")[-1])
    have = []
    if os.path.isdir(repo_root):
        try:
            have = sorted(
                e for e in os.listdir(repo_root)
                if os.path.isdir(os.path.join(repo_root, e))
            )
        except OSError:
            have = []
    have_str = ", ".join(have) if have else "none"
    return (
        f"Repo not synced yet: branch '{branch}' of {repo}. "
        f"Locally synced branches: {have_str}. "
        f"Call sync_repo(repo='{repo}', branch='{branch}') first."
    )


def _repo_dir(repo: str, branch: str) -> str:
    """Return the local directory path for a repo+branch."""
    repo_name = repo.split("/")[-1]
    return os.path.join(REPO_BASE_DIR, repo_name, branch)


def _get_remote_sha(repo: str, branch: str) -> str | None:
    """Get the latest commit SHA for a branch from GitHub via git ls-remote."""
    from tools.github import _github_token

    token = _github_token()
    url = f"https://x-access-token:{token}@github.com/{repo}.git"
    try:
        result = subprocess.run(
            ["git", "ls-remote", url, f"refs/heads/{branch}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split()[0]
    except Exception as e:
        logger.error("ls-remote failed for %s/%s: %s", repo, branch, e)
    return None


def _get_local_sha(repo_dir: str) -> str | None:
    """Get the HEAD SHA of a local repo clone."""
    head_file = os.path.join(repo_dir, ".git", "HEAD")
    if not os.path.exists(head_file):
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


@tool
def sync_repo(
    repo: str = "virtualdojo-inc/virtualdojo",
    branch: str = "main",
) -> str:
    """Sync a GitHub repo branch to a local copy for code reading and search.

    Performs a shallow clone if no local copy exists, or pulls latest if
    the remote has new commits. Skips if already up to date.

    Args:
        repo: Repository in 'owner/repo' format. Must be a whitelisted repo.
        branch: Branch name to sync (e.g. 'main', 'development').
    """
    if repo not in ALLOWED_REPOS:
        allowed = ", ".join(sorted(ALLOWED_REPOS))
        return f"Error: '{repo}' is not a whitelisted repo. Allowed: {allowed}"

    from tools.github import _github_token

    token = _github_token()
    clone_url = f"https://x-access-token:{token}@github.com/{repo}.git"
    local_dir = _repo_dir(repo, branch)

    # Serialize concurrent callers on the same (repo, branch). Without this,
    # parallel sync_repo invocations (e.g. from multiple investigate() calls
    # in the same turn) race on rm -rf + git clone into the same directory.
    with _get_sync_lock(repo, branch):
        # Check if we need to sync
        remote_sha = _get_remote_sha(repo, branch)
        if not remote_sha:
            return f"Error: Could not reach {repo} branch '{branch}'. Check the branch name."

        local_sha = _get_local_sha(local_dir)

        if local_sha == remote_sha:
            _last_synced_branch[repo] = branch
            return (
                f"Already up to date.\n"
                f"Repo: {repo} ({branch})\n"
                f"SHA: {remote_sha[:8]}\n"
                f"Local: {local_dir}"
            )

        # Clone or re-clone
        try:
            if os.path.exists(local_dir):
                # Remove stale copy and re-clone (shallow repos can't pull cleanly)
                subprocess.run(["rm", "-rf", local_dir], check=True, timeout=30)

            os.makedirs(os.path.dirname(local_dir), exist_ok=True)

            result = subprocess.run(
                [
                    "git", "clone",
                    "--depth", "1",
                    "--branch", branch,
                    "--single-branch",
                    clone_url,
                    local_dir,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                err = result.stderr[:200]
                return f"Error cloning {repo} ({branch}): {err}"

            _last_synced_branch[repo] = branch
            logger.info("Synced %s/%s to %s (SHA: %s)", repo, branch, local_dir, remote_sha[:8])
            return (
                f"Synced successfully.\n"
                f"Repo: {repo} ({branch})\n"
                f"SHA: {remote_sha[:8]}\n"
                f"Local: {local_dir}"
            )

        except subprocess.TimeoutExpired:
            return f"Error: Clone timed out for {repo} ({branch}). The repo may be too large."
        except Exception as e:
            return f"Error syncing {repo} ({branch}): {e}"


@tool
def read_repo_file(
    file_path: str,
    repo: str = "virtualdojo-inc/virtualdojo",
    branch: str | None = None,
) -> str:
    """Read a file from a locally synced repo.

    Call sync_repo first if you haven't already synced this repo+branch.

    Args:
        file_path: Path relative to repo root (e.g. 'main.py', 'app/config.py').
        repo: Repository in 'owner/repo' format.
        branch: Branch name. Defaults to the most-recently-synced branch for
            this repo, or 'main' if none has been synced this session.
    """
    if repo not in ALLOWED_REPOS:
        return f"Error: '{repo}' is not a whitelisted repo."

    branch = _resolve_branch(repo, branch)
    local_dir = _repo_dir(repo, branch)
    full_path = os.path.join(local_dir, file_path)

    if not os.path.exists(local_dir):
        return _not_synced_message(repo, branch)

    if not os.path.exists(full_path):
        return f"File not found: {file_path} in {repo} ({branch})"

    if not os.path.isfile(full_path):
        return f"'{file_path}' is a directory, not a file. Use list_repo_files to browse."

    try:
        with open(full_path, "r", errors="replace") as f:
            content = f.read()

        if len(content) > 50000:
            content = content[:50000] + f"\n\n... [truncated at 50,000 chars, full file is {len(content)} chars]"

        return content
    except Exception as e:
        return f"Error reading {file_path}: {e}"


@tool
def read_repo_file_range(
    file_path: str,
    start_line: int,
    end_line: int,
    repo: str = "virtualdojo-inc/virtualdojo",
    branch: str | None = None,
) -> str:
    """Read a specific line range from a file in a locally synced repo.

    Use this instead of read_repo_file when you already know which lines matter
    (e.g. from a search_repo_code hit). Returns only the requested lines, each
    prefixed with its 1-indexed line number.

    Call sync_repo first if you haven't already synced this repo+branch.

    Args:
        file_path: Path relative to repo root.
        start_line: 1-indexed start line (inclusive).
        end_line: 1-indexed end line (inclusive). Clamped to EOF.
        repo: Repository in 'owner/repo' format.
        branch: Branch name. Defaults to the most-recently-synced branch for
            this repo, or 'main' if none has been synced this session.
    """
    if repo not in ALLOWED_REPOS:
        return f"Error: '{repo}' is not a whitelisted repo."

    if start_line < 1 or end_line < start_line:
        return (
            f"Error: invalid range {start_line}-{end_line} "
            f"(start_line must be >= 1 and <= end_line)."
        )

    branch = _resolve_branch(repo, branch)
    local_dir = _repo_dir(repo, branch)
    full_path = os.path.join(local_dir, file_path)

    if not os.path.exists(local_dir):
        return _not_synced_message(repo, branch)

    if not os.path.exists(full_path):
        return f"File not found: {file_path} in {repo} ({branch})"

    if not os.path.isfile(full_path):
        return f"'{file_path}' is a directory, not a file. Use list_repo_files to browse."

    try:
        with open(full_path, "r", errors="replace") as f:
            lines = f.readlines()

        total = len(lines)
        if start_line > total:
            return (
                f"start_line {start_line} is past end of file "
                f"({file_path} has {total} lines)."
            )

        end = min(end_line, total)
        selected = lines[start_line - 1 : end]
        numbered = [
            f"{start_line + i}: {line.rstrip()}" for i, line in enumerate(selected)
        ]
        header = f"{file_path} lines {start_line}-{end} (of {total}):\n"
        return header + "\n".join(numbered)
    except Exception as e:
        return f"Error reading {file_path} lines {start_line}-{end_line}: {e}"


# Caps on search_repo_code output. A single tool result that exceeds the byte
# cap once OOM-killed the bot when grep returned a huge generated .md file.
SEARCH_MAX_RESULT_BYTES = 50_000
SEARCH_MAX_LINE_BYTES = 500
SEARCH_DEFAULT_HEAD_LIMIT = 50
SEARCH_VALID_OUTPUT_MODES = ("content", "files_with_matches", "count")


@tool
def search_repo_code(
    query: str,
    repo: str = "virtualdojo-inc/virtualdojo",
    branch: str | None = None,
    file_pattern: str = "",
    context_lines: int = 2,
    output_mode: str = "content",
    head_limit: int = SEARCH_DEFAULT_HEAD_LIMIT,
    offset: int = 0,
) -> str:
    """Search for a pattern in a locally synced repo using grep.

    Call sync_repo first if you haven't already synced this repo+branch.

    Output is hard-capped at ~50 KB. If you hit the cap, narrow the query,
    set a tighter file_pattern, or start with output_mode='files_with_matches'
    to scan paths first and then drill in with read_repo_file_range.

    Args:
        query: Search pattern (regex supported).
        repo: Repository in 'owner/repo' format.
        branch: Branch name. Defaults to the most-recently-synced branch for
            this repo, or 'main' if none has been synced this session.
        file_pattern: Optional glob to filter files (e.g. '*.py', '*.vue').
        context_lines: Lines of surrounding context per match (grep -C).
            Only applies when output_mode='content'. Default 2.
        output_mode: 'content' (default — matched lines + context),
            'files_with_matches' (paths only, cheapest), or
            'count' (match count per file).
        head_limit: Max output lines to return after offset. Default 50.
        offset: Skip this many output lines before head_limit. Use to
            paginate through large result sets.
    """
    if repo not in ALLOWED_REPOS:
        return f"Error: '{repo}' is not a whitelisted repo."

    if output_mode not in SEARCH_VALID_OUTPUT_MODES:
        return (
            f"Error: invalid output_mode '{output_mode}'. "
            f"Must be one of: {', '.join(SEARCH_VALID_OUTPUT_MODES)}."
        )

    if head_limit < 1:
        return "Error: head_limit must be >= 1."
    if offset < 0:
        return "Error: offset must be >= 0."

    branch = _resolve_branch(repo, branch)
    local_dir = _repo_dir(repo, branch)

    if not os.path.exists(local_dir):
        return _not_synced_message(repo, branch)

    if output_mode == "files_with_matches":
        cmd = ["grep", "-rl"]
    elif output_mode == "count":
        cmd = ["grep", "-rc"]
    else:
        cmd = ["grep", "-rn"]
        if context_lines and context_lines > 0:
            cmd += ["-C", str(context_lines)]

    if file_pattern:
        cmd += ["--include", file_pattern]
    cmd += [query, local_dir]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 1:
            return f"No matches found for '{query}' in {repo} ({branch})."

        if result.returncode != 0:
            return f"Search error: {result.stderr[:200]}"

        raw_lines = result.stdout.strip().split("\n")

        # grep -rc returns "<path>:0" for files with no matches; filter those.
        if output_mode == "count":
            raw_lines = [ln for ln in raw_lines if not ln.endswith(":0")]
            if not raw_lines:
                return f"No matches found for '{query}' in {repo} ({branch})."

        total = len(raw_lines)
        windowed = raw_lines[offset : offset + head_limit]

        formatted: list[str] = []
        running_bytes = 0
        byte_capped = False
        for line in windowed:
            cleaned = line.replace(local_dir + "/", "")
            if len(cleaned) > SEARCH_MAX_LINE_BYTES:
                cleaned = (
                    cleaned[:SEARCH_MAX_LINE_BYTES]
                    + f" ... [line truncated, was {len(line)} chars]"
                )
            # +1 for the join newline
            if running_bytes + len(cleaned) + 1 > SEARCH_MAX_RESULT_BYTES:
                byte_capped = True
                break
            formatted.append(cleaned)
            running_bytes += len(cleaned) + 1

        output = "\n".join(formatted)
        shown = len(formatted)
        next_offset = offset + shown

        notes: list[str] = []
        if byte_capped:
            notes.append(
                f"Output truncated at {SEARCH_MAX_RESULT_BYTES // 1000} KB after "
                f"{shown} of {total} lines. Narrow the query, set file_pattern, "
                f"or call again with output_mode='files_with_matches'."
            )
        elif total > offset + shown:
            notes.append(
                f"{total} total lines, showing {offset + 1}-{next_offset}. "
                f"Call again with offset={next_offset} to continue."
            )
        elif offset > 0:
            notes.append(f"{total} total lines, showing {offset + 1}-{next_offset}.")

        if notes:
            output = output + "\n\n... [" + " ".join(notes) + "]"

        return output

    except subprocess.TimeoutExpired:
        return f"Search timed out. Try a more specific query or file_pattern."
    except Exception as e:
        return f"Error searching: {e}"


@tool
def list_repo_files(
    path: str = "",
    repo: str = "virtualdojo-inc/virtualdojo",
    branch: str | None = None,
) -> str:
    """List files and directories in a locally synced repo.

    Call sync_repo first if you haven't already synced this repo+branch.

    Args:
        path: Directory path relative to repo root. Empty for root.
        repo: Repository in 'owner/repo' format.
        branch: Branch name. Defaults to the most-recently-synced branch for
            this repo, or 'main' if none has been synced this session.
    """
    if repo not in ALLOWED_REPOS:
        return f"Error: '{repo}' is not a whitelisted repo."

    branch = _resolve_branch(repo, branch)
    local_dir = _repo_dir(repo, branch)
    target = os.path.join(local_dir, path) if path else local_dir

    if not os.path.exists(local_dir):
        return _not_synced_message(repo, branch)

    if not os.path.exists(target):
        return f"Path not found: {path} in {repo} ({branch})"

    if not os.path.isdir(target):
        return f"'{path}' is a file, not a directory. Use read_repo_file to read it."

    try:
        entries = sorted(os.listdir(target))
        lines = []
        for entry in entries:
            if entry.startswith("."):
                continue
            full = os.path.join(target, entry)
            if os.path.isdir(full):
                lines.append(f"  {entry}/")
            else:
                size = os.path.getsize(full)
                if size > 1024 * 1024:
                    size_str = f"{size / (1024*1024):.1f} MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size} B"
                lines.append(f"  {entry} ({size_str})")

        if not lines:
            return f"Empty directory: {path or '/'}"

        header = f"{repo} ({branch}) — {path or '/'}\n"
        return header + "\n".join(lines)

    except Exception as e:
        return f"Error listing {path}: {e}"


REPO_SYNC_TOOLS = [
    sync_repo,
    read_repo_file,
    read_repo_file_range,
    search_repo_code,
    list_repo_files,
]
