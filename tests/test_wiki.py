"""Tests for the knowledge wiki loader (wiki.py)."""

import pytest

import wiki as wiki_mod


@pytest.fixture(autouse=True)
def _reset_cache():
    wiki_mod._catalog_cache = None
    yield
    wiki_mod._catalog_cache = None


def _write_article(d, name, content):
    (d / f"{name}.md").write_text(content, encoding="utf-8")


def test_real_seed_articles_load():
    catalog = wiki_mod.load_knowledge_catalog(force=True)
    names = {a["name"] for a in catalog}
    assert "virtualdojo-infra" in names
    assert "deploy-pipeline" in names
    # INDEX.md must NOT be treated as an article.
    assert "INDEX" not in names


def test_index_text_lists_articles():
    text = wiki_mod.knowledge_index_text()
    assert "## Knowledge base" in text
    assert "read_knowledge" in text
    assert "virtualdojo-infra" in text


def test_read_knowledge_returns_body():
    out = wiki_mod.read_knowledge.invoke({"name": "deploy-pipeline"})
    assert "Deploy Pipeline" in out
    assert "blue/green" in out.lower()


def test_read_knowledge_unknown_is_graceful():
    out = wiki_mod.read_knowledge.invoke({"name": "nope"})
    assert "No knowledge article named 'nope'" in out


def test_search_wiki_finds_terms():
    out = wiki_mod.search_wiki.invoke({"query": "blue/green"})
    assert "deploy-pipeline" in out
    # also searches skills/
    out2 = wiki_mod.search_wiki.invoke({"query": "autofix"})
    assert "github-issue-triage" in out2


def test_search_wiki_empty_query():
    assert "non-empty" in wiki_mod.search_wiki.invoke({"query": "  "})


def test_malformed_article_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(wiki_mod, "KNOWLEDGE_DIR", tmp_path)
    _write_article(tmp_path, "good", "---\ntitle: Good\nsummary: A good article.\n---\nBody.")
    _write_article(tmp_path, "no-fm", "no frontmatter here")
    _write_article(tmp_path, "empty-summary", "---\ntitle: X\nsummary: \n---\nbody")
    catalog = wiki_mod.load_knowledge_catalog(force=True)
    assert {a["name"] for a in catalog} == {"good"}
