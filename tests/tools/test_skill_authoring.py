"""Tests for the skill-authoring write tools (auth, validation, write+verify)."""

import pytest

import skills as skills_mod
import tools.skill_authoring as sa


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("SKILLS_BUCKET_ENABLED", raising=False)
    skills_mod._catalog_cache = None
    skills_mod._cache_ts = 0.0
    yield
    skills_mod._catalog_cache = None
    skills_mod._cache_ts = 0.0


DEVIN = "devin@virtualdojo.com"


class _FakeStorage:
    """In-memory stand-in for kb.storage; patched onto the real module."""

    def __init__(self):
        self.objects = {}

    def write_text(self, path, content, content_type="text/markdown"):
        self.objects[path] = content

    def read_text(self, path):
        return self.objects.get(path)

    def exists(self, path):
        return path in self.objects

    def delete(self, path):
        del self.objects[path]

    def install(self, monkeypatch):
        import kb.storage as kbs

        monkeypatch.setattr(kbs, "write_text", self.write_text)
        monkeypatch.setattr(kbs, "read_text", self.read_text)
        monkeypatch.setattr(kbs, "exists", self.exists)
        monkeypatch.setattr(kbs, "delete", self.delete)
        return self


@pytest.fixture
def fake_storage(monkeypatch):
    return _FakeStorage().install(monkeypatch)


# ── authorization ─────────────────────────────────────────────────────────

def test_save_requires_authorized_author():
    out = sa.save_skill.invoke({
        "name": "x", "description": "d", "body": "b",
        "user_email": "intruder@example.com",
    })
    assert "not authorized" in out.lower()


def test_delete_requires_authorized_author():
    out = sa.delete_skill.invoke({"name": "x", "user_email": "intruder@example.com"})
    assert "not authorized" in out.lower()


# ── validation (no bucket write should happen) ─────────────────────────────

def test_save_rejects_bad_name():
    out = sa.save_skill.invoke({
        "name": "Bad Name/../escape", "description": "d", "body": "b", "user_email": DEVIN,
    })
    assert "Invalid skill name" in out


def test_save_rejects_empty_description():
    out = sa.save_skill.invoke({
        "name": "ok-name", "description": "", "body": "b", "user_email": DEVIN,
    })
    assert "invalid" in out.lower()


def test_valid_name_blocks_path_traversal():
    assert sa._valid_name("../../etc/passwd") is False
    assert sa._valid_name("good-skill") is True
    assert sa._valid_name("claude-thing") is False  # reserved word


# ── happy path (write + read-back verify) ──────────────────────────────────

def test_save_writes_composed_md_and_verifies(fake_storage, monkeypatch):
    monkeypatch.setenv("SKILLS_BUCKET_ENABLED", "on")
    monkeypatch.setattr(skills_mod, "load_skill_catalog", lambda force=False: [])

    out = sa.save_skill.invoke({
        "name": "deploy-helper",
        "description": "How to deploy. Use when asked to deploy.",
        "body": "## Steps\n1. do the thing",
        "user_email": DEVIN,
    })
    assert "Saved skill 'deploy-helper'" in out
    assert "support/skills/deploy-helper.md" in out

    stored = fake_storage.objects["support/skills/deploy-helper.md"]
    # Frontmatter the loader can parse, with the body preserved.
    parsed = skills_mod._parse_skill_text(stored, "x")
    assert parsed is not None
    assert parsed["name"] == "deploy-helper"
    assert "do the thing" in parsed["body"]


def test_save_warns_when_bucket_disabled(fake_storage, monkeypatch):
    monkeypatch.setattr(skills_mod, "load_skill_catalog", lambda force=False: [])
    out = sa.save_skill.invoke({
        "name": "deploy-helper", "description": "d.", "body": "b", "user_email": DEVIN,
    })
    assert "Saved skill" in out
    assert "SKILLS_BUCKET_ENABLED is" in out  # the disabled note


def test_save_reports_readback_failure(fake_storage, monkeypatch):
    monkeypatch.setenv("SKILLS_BUCKET_ENABLED", "on")
    import kb.storage as kbs

    monkeypatch.setattr(kbs, "read_text", lambda path: None)  # simulate read-back miss
    monkeypatch.setattr(skills_mod, "load_skill_catalog", lambda force=False: [])
    out = sa.save_skill.invoke({
        "name": "ghost", "description": "d.", "body": "b", "user_email": DEVIN,
    })
    assert "read-back verification failed" in out


# ── delete ─────────────────────────────────────────────────────────────────

def test_delete_missing_skill(fake_storage, monkeypatch):
    monkeypatch.setattr(skills_mod, "load_skill_catalog", lambda force=False: [])
    out = sa.delete_skill.invoke({"name": "nope", "user_email": DEVIN})
    assert "No bucket skill named 'nope'" in out


def test_delete_removes_object(fake_storage, monkeypatch):
    monkeypatch.setattr(skills_mod, "load_skill_catalog", lambda force=False: [])
    fake_storage.objects["support/skills/old.md"] = "---\nname: old\ndescription: d.\n---\nbody"
    out = sa.delete_skill.invoke({"name": "old", "user_email": DEVIN})
    assert "Deleted bucket skill 'old'" in out
    assert "support/skills/old.md" not in fake_storage.objects


# ── registry wiring ─────────────────────────────────────────────────────────

def test_tools_are_judge_and_selftune_classified():
    from judge import WRITE_TOOL_NAMES
    from selftune.evalset import WRITE_TOOLS

    for name in ("save_skill", "delete_skill"):
        assert name in WRITE_TOOL_NAMES  # fail-closed judge gates them
        assert name in WRITE_TOOLS       # self-tuning classifies them as writes
