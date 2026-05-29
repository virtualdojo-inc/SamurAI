"""Tests for tools.fedramp_docs — FedRAMP documentation and code review tools."""

import base64
from unittest.mock import MagicMock, patch

import pytest

from tools.fedramp_docs import (
    _pending_file_uploads,
    _pending_fedramp_cards,
    _uploaded_files,
)

AUTHORIZED_EMAIL = "devin@virtualdojo.com"
UNAUTHORIZED_EMAIL = "hacker@example.com"
CONV_ID = "test-conv-fedramp-123"
MOCK_TOKEN = "ghp_test_token_123"


@pytest.fixture(autouse=True)
def clear_pending():
    """Clear pending state before and after each test."""
    _pending_file_uploads.clear()
    _pending_fedramp_cards.clear()
    _uploaded_files.clear()
    yield
    _pending_file_uploads.clear()
    _pending_fedramp_cards.clear()
    _uploaded_files.clear()


# ---------------------------------------------------------------------------
# fedramp_read_document
# ---------------------------------------------------------------------------


class TestReadDocument:
    """Tests for fedramp_read_document — read file from FedRAMP repo."""

    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_read_success(self, mock_token, mock_get):
        from tools.fedramp_docs import fedramp_read_document

        content_b64 = base64.b64encode(b"# AC-2 Account Management\nAccess control policy.").decode()

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"content": content_b64}),
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = fedramp_read_document.invoke({"file_path": "policies/AC-Policy.md"})
        assert "AC-2 Account Management" in result
        assert "Access control policy" in result

    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_read_not_found(self, mock_token, mock_get):
        from tools.fedramp_docs import fedramp_read_document

        mock_get.return_value = MagicMock(status_code=404)

        result = fedramp_read_document.invoke({"file_path": "nonexistent.md"})
        assert "File not found" in result
        assert "nonexistent.md" in result

    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_read_truncation(self, mock_token, mock_get):
        from tools.fedramp_docs import fedramp_read_document

        big_content = "x" * 12000
        content_b64 = base64.b64encode(big_content.encode()).decode()

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"content": content_b64}),
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = fedramp_read_document.invoke({"file_path": "big-file.md"})
        assert "truncated" in result
        assert len(result) < 12000


# ---------------------------------------------------------------------------
# fedramp_list_documents
# ---------------------------------------------------------------------------


class TestListDocuments:
    """Tests for fedramp_list_documents — list repo directory contents."""

    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_list_root(self, mock_token, mock_get):
        from tools.fedramp_docs import fedramp_list_documents

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value=[
                    {"name": "policies", "type": "dir", "size": 0},
                    {"name": "README.md", "type": "file", "size": 1234},
                ]
            ),
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = fedramp_list_documents.invoke({"path": ""})
        assert "(root)" in result
        assert "[dir] policies" in result
        assert "[file] README.md" in result
        assert "1234 bytes" in result

    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_list_not_found(self, mock_token, mock_get):
        from tools.fedramp_docs import fedramp_list_documents

        mock_get.return_value = MagicMock(status_code=404)

        result = fedramp_list_documents.invoke({"path": "nonexistent/"})
        assert "Path not found" in result

    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_list_file_instead_of_dir(self, mock_token, mock_get):
        from tools.fedramp_docs import fedramp_list_documents

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"name": "file.md", "type": "file"}),
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = fedramp_list_documents.invoke({"path": "policies/file.md"})
        assert "is a file, not a directory" in result


# ---------------------------------------------------------------------------
# fedramp_search_documents
# ---------------------------------------------------------------------------


class TestSearchDocuments:
    """Tests for fedramp_search_documents — GitHub code search."""

    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_search_results(self, mock_token, mock_get):
        from tools.fedramp_docs import fedramp_search_documents

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "total_count": 2,
                    "items": [
                        {
                            "path": "policies/AC-Policy.md",
                            "name": "AC-Policy.md",
                            "text_matches": [{"fragment": "Account management procedures"}],
                        },
                        {
                            "path": "policies/IA-Policy.md",
                            "name": "IA-Policy.md",
                            "text_matches": [],
                        },
                    ],
                }
            ),
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = fedramp_search_documents.invoke({"query": "account management"})
        assert "2 result(s)" in result
        assert "AC-Policy.md" in result
        assert "IA-Policy.md" in result
        assert "Account management procedures" in result

    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_search_no_results(self, mock_token, mock_get):
        from tools.fedramp_docs import fedramp_search_documents

        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"total_count": 0, "items": []}),
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = fedramp_search_documents.invoke({"query": "zzz_no_match_zzz"})
        assert "No results found" in result


# ---------------------------------------------------------------------------
# fedramp_propose_edit
# ---------------------------------------------------------------------------


class TestProposeEdit:
    """Tests for fedramp_propose_edit — auth-gated draft proposal."""

    def test_unauthorized_rejected(self):
        from tools.fedramp_docs import fedramp_propose_edit

        result = fedramp_propose_edit.invoke({
            "file_path": "policies/AC-Policy.md",
            "proposed_content": "New content",
            "summary": "Updated AC policy",
            "conversation_id": CONV_ID,
            "user_email": UNAUTHORIZED_EMAIL,
        })
        assert "not authorized" in result

    def test_authorized_stores_pending(self):
        from tools.fedramp_docs import fedramp_propose_edit

        result = fedramp_propose_edit.invoke({
            "file_path": "policies/AC-Policy.md",
            "proposed_content": "# Updated AC Policy",
            "summary": "Revised AC-2 controls",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "Edit proposed" in result
        assert "AC-Policy.md" in result
        assert CONV_ID in _pending_file_uploads
        assert _pending_file_uploads[CONV_ID]["content"] == "# Updated AC Policy"
        assert CONV_ID in _pending_fedramp_cards
        assert _pending_fedramp_cards[CONV_ID]["card_type"] == "fedramp_file_consent"

    def test_auth_case_insensitive(self):
        from tools.fedramp_docs import fedramp_propose_edit

        result = fedramp_propose_edit.invoke({
            "file_path": "policies/AC-Policy.md",
            "proposed_content": "content",
            "summary": "test",
            "conversation_id": CONV_ID,
            "user_email": "Devin@VirtualDojo.COM",
        })
        assert "Edit proposed" in result


# ---------------------------------------------------------------------------
# fedramp_commit_document
# ---------------------------------------------------------------------------


class TestCommitDocument:
    """Tests for fedramp_commit_document — auth-gated commit via GitHub API."""

    def test_unauthorized_rejected(self):
        from tools.fedramp_docs import fedramp_commit_document

        result = fedramp_commit_document.invoke({
            "file_path": "policies/AC-Policy.md",
            "commit_message": "Update AC",
            "conversation_id": CONV_ID,
            "user_email": UNAUTHORIZED_EMAIL,
        })
        assert "not authorized" in result

    def test_no_pending_content(self):
        from tools.fedramp_docs import fedramp_commit_document

        result = fedramp_commit_document.invoke({
            "file_path": "policies/AC-Policy.md",
            "commit_message": "Update AC",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "No pending content" in result

    @patch("tools.fedramp_docs.httpx.put")
    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_commit_success_from_pending(self, mock_token, mock_get, mock_put):
        from tools.fedramp_docs import fedramp_commit_document

        _pending_file_uploads[CONV_ID] = {
            "file_path": "policies/AC-Policy.md",
            "content": "# Updated content",
            "summary": "test",
        }

        # Mock: file already exists (GET returns SHA)
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"sha": "abc123existing"}),
        )

        # Mock: commit succeeds
        mock_put.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "commit": {"sha": "def456newcommit"},
                    "content": {"path": "policies/AC-Policy.md"},
                }
            ),
        )
        mock_put.return_value.raise_for_status = MagicMock()

        result = fedramp_commit_document.invoke({
            "file_path": "policies/AC-Policy.md",
            "commit_message": "Update AC-2 policy",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "Committed successfully" in result
        assert "def456newcommit" in result
        assert CONV_ID not in _pending_file_uploads

    @patch("tools.fedramp_docs.httpx.put")
    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_commit_new_file(self, mock_token, mock_get, mock_put):
        from tools.fedramp_docs import fedramp_commit_document

        _pending_file_uploads[CONV_ID] = {
            "file_path": "policies/new-policy.md",
            "content": "# New policy",
            "summary": "test",
        }

        mock_get.return_value = MagicMock(status_code=404)

        mock_put.return_value = MagicMock(
            status_code=201,
            json=MagicMock(
                return_value={"commit": {"sha": "newfilesha"}}
            ),
        )
        mock_put.return_value.raise_for_status = MagicMock()

        result = fedramp_commit_document.invoke({
            "file_path": "policies/new-policy.md",
            "commit_message": "Add new policy",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "Committed successfully" in result
        call_json = mock_put.call_args[1]["json"]
        assert "sha" not in call_json

    @patch("tools.fedramp_docs.httpx.put")
    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_commit_from_uploaded_file(self, mock_token, mock_get, mock_put):
        from tools.fedramp_docs import fedramp_commit_document

        _uploaded_files[CONV_ID] = {
            "content_url": "https://onedrive.example.com/file.md",
        }

        # First GET: fetch OneDrive content
        onedrive_resp = MagicMock()
        onedrive_resp.text = "# Content from OneDrive"
        onedrive_resp.raise_for_status = MagicMock()

        # Second GET: check if file exists in GitHub
        existing_resp = MagicMock()
        existing_resp.status_code = 404

        mock_get.side_effect = [onedrive_resp, existing_resp]

        mock_put.return_value = MagicMock(
            status_code=201,
            json=MagicMock(return_value={"commit": {"sha": "onedrive-sha"}}),
        )
        mock_put.return_value.raise_for_status = MagicMock()

        result = fedramp_commit_document.invoke({
            "file_path": "policies/AC-Policy.md",
            "commit_message": "Update from OneDrive",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "Committed successfully" in result
        call_json = mock_put.call_args[1]["json"]
        decoded = base64.b64decode(call_json["content"]).decode()
        assert "Content from OneDrive" in decoded


# ---------------------------------------------------------------------------
# fedramp_review_code
# ---------------------------------------------------------------------------


class TestReviewCode:
    """Tests for fedramp_review_code — NIST 800-53 code security scanner."""

    def _make_file_response(self, code_content: str):
        content_b64 = base64.b64encode(code_content.encode()).decode()
        resp = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"content": content_b64}),
        )
        resp.raise_for_status = MagicMock()
        return resp

    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_cors_wildcard_detected(self, mock_token, mock_get):
        """SC-7: CORS wildcard allow_origins=['*'] should be flagged."""
        from tools.fedramp_docs import fedramp_review_code

        code = "app.add_middleware(CORSMiddleware, allow_origins=['*'])"
        mock_get.return_value = self._make_file_response(code)

        result = fedramp_review_code.invoke({
            "repo": "virtualdojo-inc/virtualdojo",
            "file_path": "app.py",
        })
        assert "SC-7" in result
        assert "Boundary Protection" in result
        assert "CORS wildcard" in result

    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_hardcoded_credential_detected(self, mock_token, mock_get):
        """SC-12: Hardcoded credentials should be flagged."""
        from tools.fedramp_docs import fedramp_review_code

        code = 'password = "s3cret123"\nprint("hello")'
        mock_get.return_value = self._make_file_response(code)

        result = fedramp_review_code.invoke({
            "repo": "virtualdojo-inc/virtualdojo",
            "file_path": "config.py",
        })
        assert "SC-12" in result
        assert "hardcoded credential" in result.lower()

    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_bare_except_detected(self, mock_token, mock_get):
        """CM-6: Bare except clause should be flagged."""
        from tools.fedramp_docs import fedramp_review_code

        code = "try:\n    do_something()\nexcept:\n    pass"
        mock_get.return_value = self._make_file_response(code)

        result = fedramp_review_code.invoke({
            "repo": "virtualdojo-inc/virtualdojo",
            "file_path": "handler.py",
        })
        assert "CM-6" in result
        assert "Bare except" in result

    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_clean_code_no_findings(self, mock_token, mock_get):
        from tools.fedramp_docs import fedramp_review_code

        code = (
            "import os\n\n"
            "def get_config():\n"
            "    return os.environ.get('DB_HOST', 'localhost')\n"
        )
        mock_get.return_value = self._make_file_response(code)

        result = fedramp_review_code.invoke({
            "repo": "virtualdojo-inc/virtualdojo",
            "file_path": "config.py",
        })
        assert "No FedRAMP security issues found" in result
        assert "Clean report" in result

    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_file_not_found(self, mock_token, mock_get):
        from tools.fedramp_docs import fedramp_review_code

        mock_get.return_value = MagicMock(status_code=404)

        result = fedramp_review_code.invoke({
            "repo": "virtualdojo-inc/virtualdojo",
            "file_path": "nonexistent.py",
        })
        assert "File not found" in result

    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_xss_vhtml_detected(self, mock_token, mock_get):
        """SC-18: v-html directive should be flagged."""
        from tools.fedramp_docs import fedramp_review_code

        code = '<div v-html="userInput"></div>'
        mock_get.return_value = self._make_file_response(code)

        result = fedramp_review_code.invoke({
            "repo": "virtualdojo-inc/virtualdojo",
            "file_path": "template.vue",
        })
        assert "SC-18" in result
        assert "v-html" in result

    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_access_control_allow_origin_wildcard(self, mock_token, mock_get):
        """SC-7: Access-Control-Allow-Origin: * should be flagged."""
        from tools.fedramp_docs import fedramp_review_code

        code = 'response.headers["Access-Control-Allow-Origin"] = "*"'
        mock_get.return_value = self._make_file_response(code)

        result = fedramp_review_code.invoke({
            "repo": "virtualdojo-inc/virtualdojo",
            "file_path": "middleware.py",
        })
        assert "SC-7" in result
        assert "Access-Control-Allow-Origin" in result

    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_print_statement_detected(self, mock_token, mock_get):
        """CM-6: print() in production code should be flagged."""
        from tools.fedramp_docs import fedramp_review_code

        code = "def handler():\n    print(request.body)\n    return 'ok'"
        mock_get.return_value = self._make_file_response(code)

        result = fedramp_review_code.invoke({
            "repo": "virtualdojo-inc/virtualdojo",
            "file_path": "handler.py",
        })
        assert "CM-6" in result
        assert "print()" in result

    @patch("tools.fedramp_docs.httpx.get")
    @patch("tools.fedramp_docs._github_token", return_value=MOCK_TOKEN)
    def test_multiple_findings(self, mock_token, mock_get):
        from tools.fedramp_docs import fedramp_review_code

        code = (
            'password = "abc123"\n'
            "try:\n"
            "    call_api()\n"
            "except:\n"
            "    pass\n"
        )
        mock_get.return_value = self._make_file_response(code)

        result = fedramp_review_code.invoke({
            "repo": "virtualdojo-inc/virtualdojo",
            "file_path": "bad.py",
        })
        assert "SC-12" in result
        assert "CM-6" in result
        assert "Issues found:" in result


# ---------------------------------------------------------------------------
# fedramp_discard_draft
# ---------------------------------------------------------------------------


class TestDiscardDraft:
    """Tests for fedramp_discard_draft — cleanup pending state."""

    def test_unauthorized_rejected(self):
        from tools.fedramp_docs import fedramp_discard_draft

        result = fedramp_discard_draft.invoke({
            "file_path": "policies/AC-Policy.md",
            "conversation_id": CONV_ID,
            "user_email": UNAUTHORIZED_EMAIL,
        })
        assert "not authorized" in result

    def test_no_pending_draft(self):
        from tools.fedramp_docs import fedramp_discard_draft

        result = fedramp_discard_draft.invoke({
            "file_path": "policies/AC-Policy.md",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "No pending draft" in result

    def test_discard_cleans_all_state(self):
        from tools.fedramp_docs import fedramp_discard_draft

        _pending_file_uploads[CONV_ID] = {"file_path": "f.md", "content": "c"}
        _uploaded_files[CONV_ID] = {"content_url": "https://example.com"}
        _pending_fedramp_cards[CONV_ID] = {"card_type": "fedramp_file_consent"}

        result = fedramp_discard_draft.invoke({
            "file_path": "f.md",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "Discarded draft" in result
        assert "pending upload" in result
        assert "uploaded file reference" in result
        assert "pending card" in result
        assert CONV_ID not in _pending_file_uploads
        assert CONV_ID not in _uploaded_files
        assert CONV_ID not in _pending_fedramp_cards
