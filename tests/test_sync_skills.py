"""Tests for the in-boundary skill sync (kb/sync_skills.py)."""

import pytest

from kb import sync_skills


SKILL_A = """---
name: batch-flush-fix
description: Batch bulk writes instead of per-row flush
---
Use a Core multi-row insert; assert affected row count.
"""

SKILL_B = """---
name: clone-clin-status
description: Consolidate CLIN status writes
---
Single bulk insert path.
"""


@pytest.fixture
def fake_bucket(monkeypatch):
    """In-memory stand-in for kb.storage."""
    store: dict[str, str] = {}
    locks: dict[str, bool] = {}

    monkeypatch.setattr(sync_skills.storage, "read_text",
                        lambda p: store.get(p))
    monkeypatch.setattr(sync_skills.storage, "write_text",
                        lambda p, c, **k: store.__setitem__(p, c))
    monkeypatch.setattr(sync_skills.storage, "delete",
                        lambda p: store.pop(p, None))
    monkeypatch.setattr(sync_skills.storage, "list_paths",
                        lambda prefix: [k for k in store if k.startswith(prefix)])
    monkeypatch.setattr(sync_skills.storage, "acquire_lock",
                        lambda p, ttl_seconds=600: locks.setdefault(p, True))
    monkeypatch.setattr(sync_skills.storage, "release_lock",
                        lambda p: locks.pop(p, None))
    return store


def _set_catalog(monkeypatch, items):
    """items: list of (name, text, sha)."""
    monkeypatch.setattr(sync_skills, "_fetch_catalog_skills", lambda: items)


def test_kill_switch_off_skips(monkeypatch, fake_bucket):
    monkeypatch.delenv("SKILLS_SYNC_ENABLED", raising=False)
    _set_catalog(monkeypatch, [("batch-flush-fix", SKILL_A, "sha1")])
    result = sync_skills.run_skill_sync()
    assert result == {"skipped": True}
    assert fake_bucket == {}


def test_force_bypasses_kill_switch_and_writes(monkeypatch, fake_bucket):
    monkeypatch.delenv("SKILLS_SYNC_ENABLED", raising=False)
    _set_catalog(monkeypatch, [("batch-flush-fix", SKILL_A, "sha1")])
    result = sync_skills.run_skill_sync(force=True)
    assert result["written"] == 1
    key = "support/skills/synced/batch-flush-fix.md"
    assert key in fake_bucket
    assert "synced: true" in fake_bucket[key]
    assert "source_sha: sha1" in fake_bucket[key]


def test_provenance_keeps_skill_parseable(monkeypatch, fake_bucket):
    from skills import _parse_skill_text
    _set_catalog(monkeypatch, [("batch-flush-fix", SKILL_A, "shaX")])
    sync_skills.run_skill_sync(force=True)
    stamped = fake_bucket["support/skills/synced/batch-flush-fix.md"]
    parsed = _parse_skill_text(stamped, "test")
    assert parsed is not None
    assert parsed["name"] == "batch-flush-fix"
    assert parsed["description"] == "Batch bulk writes instead of per-row flush"


def test_unchanged_content_not_rewritten(monkeypatch, fake_bucket):
    _set_catalog(monkeypatch, [("batch-flush-fix", SKILL_A, "sha1")])
    sync_skills.run_skill_sync(force=True)
    second = sync_skills.run_skill_sync(force=True)
    assert second["written"] == 0  # idempotent


def test_prunes_synced_file_removed_from_catalog(monkeypatch, fake_bucket):
    _set_catalog(monkeypatch, [("batch-flush-fix", SKILL_A, "s1"),
                               ("clone-clin-status", SKILL_B, "s2")])
    sync_skills.run_skill_sync(force=True)
    assert len(fake_bucket) == 2
    # catalog shrinks — the removed skill's synced file is pruned
    _set_catalog(monkeypatch, [("batch-flush-fix", SKILL_A, "s1")])
    result = sync_skills.run_skill_sync(force=True)
    assert result["pruned"] == 1
    assert "support/skills/synced/clone-clin-status.md" not in fake_bucket
    assert "support/skills/synced/batch-flush-fix.md" in fake_bucket


def test_prune_never_touches_hand_authored_toplevel(monkeypatch, fake_bucket):
    # a hand-authored skill lives at the top-level prefix, NOT under synced/
    fake_bucket["support/skills/my-local-skill.md"] = SKILL_B
    _set_catalog(monkeypatch, [("batch-flush-fix", SKILL_A, "s1")])
    sync_skills.run_skill_sync(force=True)
    # synced write happened, hand-authored file untouched
    assert "support/skills/my-local-skill.md" in fake_bucket
    assert "support/skills/synced/batch-flush-fix.md" in fake_bucket


def test_lock_contention_skips(monkeypatch, fake_bucket):
    monkeypatch.setattr(sync_skills.storage, "acquire_lock",
                        lambda p, ttl_seconds=600: False)
    _set_catalog(monkeypatch, [("batch-flush-fix", SKILL_A, "s1")])
    result = sync_skills.run_skill_sync(force=True)
    assert result["skipped"] is True
    assert result["reason"] == "locked"


def test_synced_skill_loses_to_hand_authored_on_name_clash(monkeypatch):
    """skills._load_bucket_skills orders synced before top-level so the
    hand-authored local override wins in _load_catalog's last-wins dedup."""
    import skills

    items = [
        ("support/skills/synced/dup.md",
         "---\nname: dup\ndescription: synced version\n---\nsynced body"),
        ("support/skills/dup.md",
         "---\nname: dup\ndescription: local version\n---\nlocal body"),
    ]
    monkeypatch.setattr(skills, "_bucket_skills_enabled", lambda: True)
    monkeypatch.setattr(skills.storage if hasattr(skills, "storage") else skills,
                        "list_text", lambda prefix: items, raising=False)
    # skills imports storage lazily inside _load_bucket_skills; patch there:
    import kb.storage
    monkeypatch.setattr(kb.storage, "list_text", lambda prefix: items)

    loaded = skills._load_bucket_skills()
    # top-level entry must come last so it wins last-wins dedup
    assert loaded[-1]["description"] == "local version"
