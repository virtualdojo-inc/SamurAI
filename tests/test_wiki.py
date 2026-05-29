"""Tests for the knowledge wiki loader (wiki.py) — now bucket/mount-backed."""

import pytest

import wiki as wiki_mod


@pytest.fixture(autouse=True)
def _reset_cache():
    wiki_mod._catalog_cache = None
    yield
    wiki_mod._catalog_cache = None


def _mk(root, scope, name, content):
    d = root / scope / "wiki"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.md").write_text(content, encoding="utf-8")


def _mount(tmp_path, monkeypatch):
    """Point the loader at a temp 'mount' and disable the repo fallback."""
    monkeypatch.setattr(wiki_mod, "KB_KNOWLEDGE_ROOT", str(tmp_path))
    monkeypatch.setattr(wiki_mod, "_REPO_KNOWLEDGE_DIR", tmp_path / "_no_repo_fallback")
    return tmp_path


def test_loads_from_mounted_scopes(tmp_path, monkeypatch):
    root = _mount(tmp_path, monkeypatch)
    # migrated-style frontmatter (title + summary)
    _mk(root, "engineering", "virtualdojo-infra",
        "---\ntitle: VirtualDojo Infrastructure\nsummary: where the bot runs.\n---\n# Infra\nCloud Run.")
    # compile-style frontmatter (title + sources + last_verified + confidence, NO summary)
    _mk(root, "support", "login-issues",
        "---\ntitle: Login Issues\nsources: [raw/x]\nlast_verified: 2026-05-29\nconfidence: high\n---\n# Login Issues\nSSO token expiry is common.")
    catalog = wiki_mod.load_knowledge_catalog(force=True)
    names = {a["name"] for a in catalog}
    assert names == {"virtualdojo-infra", "login-issues"}
    # summary derived for the compile-style article (no summary field)
    login = next(a for a in catalog if a["name"] == "login-issues")
    assert login["summary"]  # non-empty (derived from body)


def test_index_and_read_and_search(tmp_path, monkeypatch):
    root = _mount(tmp_path, monkeypatch)
    _mk(root, "engineering", "deploy-pipeline",
        "---\ntitle: Deploy Pipeline\nsummary: blue/green deploy.\n---\n# Deploy Pipeline\nHealth-gated blue/green.")
    idx = wiki_mod.knowledge_index_text()
    assert "## Knowledge base" in idx and "deploy-pipeline" in idx
    body = wiki_mod.read_knowledge.invoke({"name": "deploy-pipeline"})
    assert "blue/green" in body.lower()
    hit = wiki_mod.search_wiki.invoke({"query": "blue/green"})
    assert "deploy-pipeline" in hit


def test_read_unknown_graceful(tmp_path, monkeypatch):
    _mount(tmp_path, monkeypatch)
    _mk(tmp_path, "engineering", "a", "---\ntitle: A\nsummary: s.\n---\nbody")
    out = wiki_mod.read_knowledge.invoke({"name": "missing"})
    assert "No knowledge article named 'missing'" in out


def test_index_and_keep_files_excluded(tmp_path, monkeypatch):
    root = _mount(tmp_path, monkeypatch)
    _mk(root, "support", "real", "---\ntitle: Real\nsummary: s.\n---\nbody")
    d = root / "support" / "wiki"
    (d / "index.md").write_text("# index", encoding="utf-8")
    (d / ".keep").write_text("", encoding="utf-8")
    names = {a["name"] for a in wiki_mod.load_knowledge_catalog(force=True)}
    assert names == {"real"}


def test_repo_fallback_when_mount_empty(tmp_path, monkeypatch):
    # Empty mount, but a repo-style fallback dir with an article → used.
    monkeypatch.setattr(wiki_mod, "KB_KNOWLEDGE_ROOT", str(tmp_path / "empty_mount"))
    fallback = tmp_path / "repo_knowledge"
    fallback.mkdir()
    (fallback / "INDEX.md").write_text("# idx", encoding="utf-8")
    (fallback / "legacy.md").write_text("---\ntitle: Legacy\nsummary: s.\n---\nbody", encoding="utf-8")
    monkeypatch.setattr(wiki_mod, "_REPO_KNOWLEDGE_DIR", fallback)
    names = {a["name"] for a in wiki_mod.load_knowledge_catalog(force=True)}
    assert names == {"legacy"}  # INDEX.md excluded, fallback used
