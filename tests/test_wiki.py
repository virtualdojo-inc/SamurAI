"""Tests for the knowledge wiki loader (wiki.py) — storage-client backed."""

import pytest

import wiki as wiki_mod
from kb import storage as kb_storage


@pytest.fixture(autouse=True)
def _reset_cache():
    wiki_mod._catalog_cache = None
    wiki_mod._cache_ts = 0.0
    yield
    wiki_mod._catalog_cache = None
    wiki_mod._cache_ts = 0.0


def _fake_bucket(objs):
    """objs: {full_path: text}. Returns a list_text(prefix) impl."""
    def _list_text(prefix, suffix=".md"):
        return [(p, t) for p, t in objs.items() if p.startswith(prefix) and p.endswith(suffix)]
    return _list_text


def test_loads_from_bucket_scopes(monkeypatch):
    objs = {
        # migrated-style frontmatter (title + summary)
        "engineering/wiki/virtualdojo-infra.md":
            "---\ntitle: VirtualDojo Infrastructure\nsummary: where the bot runs.\n---\n# Infra\nCloud Run.",
        # compile-style frontmatter (no summary → derived)
        "support/wiki/login-issues.md":
            "---\ntitle: Login Issues\nsources: [raw/x]\nlast_verified: 2026-05-29\nconfidence: high\n---\n# Login Issues\nSSO token expiry is common.",
    }
    monkeypatch.setattr(kb_storage, "list_text", _fake_bucket(objs))
    catalog = wiki_mod.load_knowledge_catalog(force=True)
    names = {a["name"] for a in catalog}
    assert names == {"virtualdojo-infra", "login-issues"}
    login = next(a for a in catalog if a["name"] == "login-issues")
    assert login["summary"]  # derived from body


def test_index_read_search(monkeypatch):
    objs = {"engineering/wiki/deploy-pipeline.md":
            "---\ntitle: Deploy Pipeline\nsummary: blue/green deploy.\n---\n# Deploy Pipeline\nHealth-gated blue/green."}
    monkeypatch.setattr(kb_storage, "list_text", _fake_bucket(objs))
    idx = wiki_mod.knowledge_index_text()
    assert "## Knowledge base" in idx and "deploy-pipeline" in idx
    body = wiki_mod.read_knowledge.invoke({"name": "deploy-pipeline"})
    assert "blue/green" in body.lower()
    hit = wiki_mod.search_wiki.invoke({"query": "blue/green"})
    assert "deploy-pipeline" in hit


def test_excludes_index_and_keep(monkeypatch):
    objs = {
        "support/wiki/real.md": "---\ntitle: Real\nsummary: s.\n---\nbody",
        "support/wiki/index.md": "# index",
        "support/wiki/.keep": "",
    }
    monkeypatch.setattr(kb_storage, "list_text", _fake_bucket(objs))
    names = {a["name"] for a in wiki_mod.load_knowledge_catalog(force=True)}
    assert names == {"real"}


def test_read_unknown_graceful(monkeypatch):
    monkeypatch.setattr(kb_storage, "list_text", _fake_bucket({}))
    # empty bucket → repo fallback may apply; ensure unknown name handled
    out = wiki_mod.read_knowledge.invoke({"name": "definitely-missing-xyz"})
    assert "No knowledge article named 'definitely-missing-xyz'" in out


def test_repo_fallback_when_bucket_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(kb_storage, "list_text", _fake_bucket({}))  # bucket empty
    fallback = tmp_path / "repo_knowledge"
    fallback.mkdir()
    (fallback / "INDEX.md").write_text("# idx", encoding="utf-8")
    (fallback / "legacy.md").write_text("---\ntitle: Legacy\nsummary: s.\n---\nbody", encoding="utf-8")
    monkeypatch.setattr(wiki_mod, "_REPO_KNOWLEDGE_DIR", fallback)
    names = {a["name"] for a in wiki_mod.load_knowledge_catalog(force=True)}
    assert names == {"legacy"}


def test_scope_read_failure_isolated(monkeypatch):
    """A failure reading one scope doesn't abort the others."""
    def _list_text(prefix, suffix=".md"):
        if prefix.startswith("customers/onboarding/"):
            raise PermissionError("denied")
        if prefix.startswith("engineering/"):
            return [("engineering/wiki/a.md", "---\ntitle: A\nsummary: s.\n---\nbody")]
        return []
    monkeypatch.setattr(kb_storage, "list_text", _list_text)
    names = {a["name"] for a in wiki_mod.load_knowledge_catalog(force=True)}
    assert names == {"a"}  # engineering loaded despite onboarding failure
