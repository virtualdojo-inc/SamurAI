"""Tests for the knowledge wiki loader (wiki.py) — storage-client backed."""

import pytest

import wiki as wiki_mod
from kb import storage as kb_storage


def _reset():
    wiki_mod._catalog_cache = None
    wiki_mod._cache_ts = 0.0
    wiki_mod._lazy_names_cache = {}
    wiki_mod._lazy_body_cache = {}
    wiki_mod._refreshing = False


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    _reset()
    # Default the lazy-prefix listing to empty so tests that only fake list_text
    # never fall through to a real GCS client.
    monkeypatch.setattr(kb_storage, "list_paths", lambda prefix: [])
    yield
    _reset()


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


# ── Lazy playbook tier (support/playbooks/ is never bulk-downloaded) ─────


def _fake_lazy_bucket(monkeypatch, playbooks: dict, curated: dict | None = None):
    """playbooks: {path: text} under support/playbooks/. Tracks downloads."""
    downloads = []

    def _list_paths(prefix):
        return [p for p in playbooks if p.startswith(prefix)]

    def _read_text(path):
        downloads.append(path)
        return playbooks.get(path)

    monkeypatch.setattr(kb_storage, "list_text", _fake_bucket(curated or {}))
    monkeypatch.setattr(kb_storage, "list_paths", _list_paths)
    monkeypatch.setattr(kb_storage, "read_text", _read_text)
    return downloads


def test_playbooks_listed_by_name_not_downloaded(monkeypatch):
    playbooks = {
        f"support/playbooks/area-{i}.md": f"---\ntitle: Area {i}\n---\nbody {i}"
        for i in range(40)
    }
    downloads = _fake_lazy_bucket(monkeypatch, playbooks)
    catalog = wiki_mod.load_knowledge_catalog(force=True)
    # Playbooks are NOT in the curated catalog and nothing was downloaded.
    assert catalog == []
    assert downloads == []
    assert len(wiki_mod._lazy_names_cache) == 40


def test_index_summarizes_playbooks_in_one_line(monkeypatch):
    playbooks = {
        f"support/playbooks/area-{i}.md": f"---\ntitle: Area {i}\n---\nbody"
        for i in range(40)
    }
    curated = {"engineering/wiki/infra.md": "---\ntitle: Infra\nsummary: s.\n---\nbody"}
    _fake_lazy_bucket(monkeypatch, playbooks, curated)
    wiki_mod.load_knowledge_catalog(force=True)
    idx = wiki_mod.knowledge_index_text()
    assert "infra" in idx                      # curated article gets its own line
    assert "40 support troubleshooting playbooks" in idx
    assert "area-7" not in idx                 # playbooks are not enumerated


def test_read_knowledge_fetches_lazy_playbook_on_demand(monkeypatch):
    playbooks = {"support/playbooks/quote-errors.md":
                 "---\ntitle: Quote Errors\n---\nCheck the pricing service."}
    downloads = _fake_lazy_bucket(monkeypatch, playbooks)
    wiki_mod.load_knowledge_catalog(force=True)
    body = wiki_mod.read_knowledge.invoke({"name": "quote-errors"})
    assert "pricing service" in body
    assert downloads == ["support/playbooks/quote-errors.md"]
    # Second read is served from the body cache — no second GET.
    wiki_mod.read_knowledge.invoke({"name": "quote-errors"})
    assert len(downloads) == 1


def test_search_wiki_matches_lazy_playbook_by_name(monkeypatch):
    playbooks = {"support/playbooks/sso-login.md":
                 "---\ntitle: SSO Login\n---\nToken expiry causes 401s."}
    _fake_lazy_bucket(monkeypatch, playbooks)
    wiki_mod.load_knowledge_catalog(force=True)
    out = wiki_mod.search_wiki.invoke({"query": "sso login"})
    assert "sso-login" in out
    assert "Token expiry" in out


# ── Stale-while-revalidate cache ─────────────────────────────────────────


def test_stale_cache_is_served_not_reloaded_inline(monkeypatch):
    """A stale cache is returned immediately; the reload happens off-path."""
    calls = []

    def _tracking_refresh():
        calls.append(1)

    wiki_mod._catalog_cache = [{"name": "old", "title": "Old", "summary": "s", "body": "b"}]
    wiki_mod._cache_ts = 0.0  # far past the TTL
    monkeypatch.setattr(wiki_mod, "_refresh_in_background", _tracking_refresh)
    got = wiki_mod.load_knowledge_catalog()
    assert [a["name"] for a in got] == ["old"]  # stale served synchronously
    assert calls == [1]                          # refresh dispatched off-path


def test_fresh_cache_skips_refresh(monkeypatch):
    import time as _time

    wiki_mod._catalog_cache = [{"name": "cur", "title": "C", "summary": "s", "body": "b"}]
    wiki_mod._cache_ts = _time.time()
    monkeypatch.setattr(
        wiki_mod, "_refresh_in_background",
        lambda: pytest.fail("fresh cache must not refresh"),
    )
    assert [a["name"] for a in wiki_mod.load_knowledge_catalog()] == ["cur"]


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
