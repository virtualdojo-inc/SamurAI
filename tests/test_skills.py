"""Tests for the Agent Skills loader (skills.py)."""

import importlib

import pytest

import skills as skills_mod


@pytest.fixture(autouse=True)
def _reset_cache():
    """Each test starts with a fresh catalog cache."""
    skills_mod._catalog_cache = None
    yield
    skills_mod._catalog_cache = None


def _write_skill(tmp_path, dirname, content):
    d = tmp_path / dirname
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(content, encoding="utf-8")
    return d


def test_real_seed_skills_load():
    """The committed seed skills parse and are discoverable."""
    catalog = skills_mod.load_skill_catalog(force=True)
    names = {s["name"] for s in catalog}
    assert "troubleshooting-cloud-run" in names
    assert "github-issue-triage" in names
    for s in catalog:
        assert s["description"]
        assert s["body"]


def test_catalog_text_lists_skills():
    text = skills_mod.skills_catalog_text()
    assert "## Available skills" in text
    assert "get_skill" in text
    assert "troubleshooting-cloud-run" in text


def test_get_skill_returns_body():
    body = skills_mod.get_skill.invoke({"name": "troubleshooting-cloud-run"})
    assert "Troubleshooting Cloud Run" in body
    assert "function response parts" in body  # known-signature section


def test_get_skill_unknown_name_is_graceful():
    out = skills_mod.get_skill.invoke({"name": "does-not-exist"})
    assert "No skill named 'does-not-exist'" in out
    assert "Available skills" in out


def test_parser_validates_against_tmp(tmp_path, monkeypatch):
    """Loader skips malformed skills and keeps valid ones."""
    monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path)

    # valid
    _write_skill(
        tmp_path,
        "good-skill",
        "---\nname: good-skill\ndescription: A valid skill for testing.\n---\nBody here.",
    )
    # missing frontmatter
    _write_skill(tmp_path, "no-frontmatter", "Just a body, no frontmatter.")
    # invalid name (uppercase)
    _write_skill(
        tmp_path,
        "bad-name",
        "---\nname: Bad_Name\ndescription: x\n---\nbody",
    )
    # empty description
    _write_skill(
        tmp_path,
        "empty-desc",
        "---\nname: empty-desc\ndescription: \n---\nbody",
    )
    # reserved word in name
    _write_skill(
        tmp_path,
        "reserved",
        "---\nname: claude-helper\ndescription: uses reserved word.\n---\nbody",
    )

    catalog = skills_mod.load_skill_catalog(force=True)
    names = {s["name"] for s in catalog}
    assert names == {"good-skill"}


def test_duplicate_names_keep_first(tmp_path, monkeypatch):
    monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path)
    _write_skill(
        tmp_path, "a", "---\nname: dup\ndescription: first.\n---\nfirst body"
    )
    _write_skill(
        tmp_path, "b", "---\nname: dup\ndescription: second.\n---\nsecond body"
    )
    catalog = skills_mod.load_skill_catalog(force=True)
    dups = [s for s in catalog if s["name"] == "dup"]
    assert len(dups) == 1


def test_missing_skills_dir_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path / "nope")
    assert skills_mod.load_skill_catalog(force=True) == []
    assert skills_mod.skills_catalog_text() == ""
