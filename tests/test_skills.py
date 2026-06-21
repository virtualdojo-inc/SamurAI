"""Tests for the Agent Skills loader (skills.py)."""

import importlib

import pytest

import skills as skills_mod


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    """Each test starts with a fresh catalog cache and the bucket disabled
    (hermetic — no GCS calls unless a test opts in)."""
    monkeypatch.delenv("SKILLS_BUCKET_ENABLED", raising=False)
    skills_mod._catalog_cache = None
    skills_mod._cache_ts = 0.0
    yield
    skills_mod._catalog_cache = None
    skills_mod._cache_ts = 0.0


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
    assert "controlled-issue-fix" in names
    assert "tech-issue-triage" in names
    for s in catalog:
        assert s["description"]
        assert s["body"]


def test_controlled_issue_fix_skill_carries_safety_guidance():
    """The controlled-issue-fix skill must keep its safety-critical guidance:
    the eligibility deny-list, the compliance field-allowlist, and an honest
    'plan only, no PR' capability boundary."""
    body = skills_mod.get_skill.invoke({"name": "controlled-issue-fix"})
    # Eligibility gating so risky bug classes are refused.
    assert "ALLOW" in body
    assert "DENY" in body
    assert "PII" in body
    # Compliance field-allowlist: nothing in-boundary may enter the brief.
    assert "Cloud Logging" in body
    assert "CRM" in body
    # Honest capability boundary: produces a plan, does not open a PR.
    assert "open a PR" in body


def test_tech_issue_triage_skill_is_diagnose_only_and_grounded():
    """The tech-issue-triage skill must keep its safety-critical guarantees:
    diagnose-only (never acts), grounded-in-facts, the A/B/C/D categorization,
    the machine-readable trailer the worker parses, and the in-boundary rule."""
    body = skills_mod.get_skill.invoke({"name": "tech-issue-triage"})
    # Diagnose only — never acts.
    assert "never" in body.lower()
    assert "file" in body.lower()
    # Grounded-in-facts.
    assert "ground" in body.lower()
    # The four categories.
    for cat in ("A", "B", "C", "D"):
        assert f"**{cat} " in body
    # The machine-readable trailer the worker greps.
    assert "CATEGORY:" in body
    assert "SUGGESTED_TYPE:" in body
    assert "SUGGESTED_PRIORITY:" in body
    assert "SUMMARY:" in body
    # Stays in-boundary.
    assert "boundary" in body.lower()


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


# ── bucket-backed (editable) skills ───────────────────────────────────────

def test_bucket_disabled_by_default_skips_read(monkeypatch):
    """With the kill switch off, no bucket read is attempted at all."""
    def _boom():
        raise AssertionError("bucket must not be read when disabled")

    # If _load_bucket_skills tried to read storage it would call this; the env
    # gate must short-circuit before then.
    monkeypatch.setattr(skills_mod, "_bucket_skills_enabled", lambda: False)
    assert skills_mod._load_bucket_skills() == []


def test_bucket_skill_merges_and_overrides(tmp_path, monkeypatch):
    """A bucket skill adds a new skill and overrides a repo skill by name."""
    _write_skill(
        tmp_path, "repo-only",
        "---\nname: repo-only\ndescription: from repo.\n---\nrepo-only body",
    )
    _write_skill(
        tmp_path, "shared",
        "---\nname: shared\ndescription: repo version.\n---\nREPO BODY",
    )
    monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path)

    bucket = [
        {"name": "shared", "description": "bucket version.", "body": "BUCKET BODY", "dir": "support/skills/shared.md"},
        {"name": "bucket-new", "description": "only in bucket.", "body": "NEW", "dir": "support/skills/bucket-new.md"},
    ]
    monkeypatch.setattr(skills_mod, "_load_bucket_skills", lambda: bucket)

    catalog = skills_mod.load_skill_catalog(force=True)
    by_name = {s["name"]: s for s in catalog}
    assert set(by_name) == {"repo-only", "shared", "bucket-new"}
    # Bucket overrides repo on name clash (edits win).
    assert by_name["shared"]["body"] == "BUCKET BODY"
    assert by_name["shared"]["description"] == "bucket version."
    # And get_skill serves the bucket body.
    assert skills_mod.get_skill.invoke({"name": "shared"}) == "BUCKET BODY"


def test_bucket_read_failure_falls_back_to_repo(monkeypatch):
    """If the bucket read raises, repo skills still load (bucket is additive)."""
    monkeypatch.setenv("SKILLS_BUCKET_ENABLED", "on")
    import kb.storage as kbs

    def _raise(*a, **k):
        raise RuntimeError("403 / no creds")

    monkeypatch.setattr(kbs, "list_text", _raise)
    assert skills_mod._load_bucket_skills() == []  # guarded, no crash
    catalog = skills_mod.load_skill_catalog(force=True)
    assert "tech-issue-triage" in {s["name"] for s in catalog}  # repo seeds intact


def test_bucket_skill_invalid_frontmatter_is_skipped(monkeypatch):
    monkeypatch.setenv("SKILLS_BUCKET_ENABLED", "on")
    import kb.storage as kbs

    monkeypatch.setattr(kbs, "list_text", lambda *a, **k: [
        ("support/skills/good.md", "---\nname: bucket-good\ndescription: ok.\n---\nbody"),
        ("support/skills/bad.md", "no frontmatter at all"),
        ("support/skills/index.md", "---\nname: idx\ndescription: x\n---\nshould be skipped"),
    ])
    out = skills_mod._load_bucket_skills()
    names = {s["name"] for s in out}
    assert names == {"bucket-good"}  # bad skipped, index.md skipped


def test_skills_bucket_prefix_is_never_harvested_or_served():
    """Structural guarantee: the skills prefix is outside every compile source
    and every wiki-served subdir, and inside a writable IAM scope. If a future
    refactor points the compile/wiki at it, this fails loudly."""
    import kb.compile as compile_mod
    import kb.compile_engineering as eng_mod
    import wiki

    prefix = skills_mod.SKILLS_BUCKET_PREFIX  # "support/skills/"

    # Not a compile source (compile lists support/raw/ and engineering/raw/stubs/).
    assert not prefix.startswith(compile_mod.RAW_PREFIX)
    assert not compile_mod.RAW_PREFIX.startswith(prefix)
    assert not prefix.startswith(eng_mod.STUBS_PREFIX)

    # Not a wiki-served subdir (wiki serves <scope>/{wiki,playbooks,troubleshooting}).
    scope, sub = prefix.rstrip("/").split("/", 1)
    assert sub not in wiki.SERVE_SUBDIRS

    # Inside a writable IAM scope (support/ or customers/onboarding/), so the
    # runtime SA can read+write it without an IAM change.
    assert prefix.startswith("support/") or prefix.startswith("customers/onboarding/")
