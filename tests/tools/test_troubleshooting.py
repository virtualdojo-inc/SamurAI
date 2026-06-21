"""Tests for tools/troubleshooting.py — troubleshooting DB (async store)."""

import time
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_store():
    """Reset the memory singleton between tests so each test gets a clean store."""
    import memory

    memory._store = None
    memory._store_pool = None
    yield
    memory._store = None
    memory._store_pool = None


@pytest.fixture
async def _mem_store():
    """Patch the embedding function and return the initialized store.

    No DATABASE_URL in tests → get_memory_store returns the InMemoryStore
    fallback (which supports both sync and async access).
    """
    import memory

    with patch(
        "memory._create_embed_fn",
        return_value=lambda texts: [[0.1] * 768 for _ in texts],
    ):
        store = await memory.get_memory_store()
    return store


# --- Module surface ---


def test_tools_registered():
    from tools.troubleshooting import (
        TROUBLESHOOTING_TOOLS,
        save_troubleshooting_step,
        search_troubleshooting,
        delete_troubleshooting_step,
    )

    names = {t.name for t in TROUBLESHOOTING_TOOLS}
    assert names == {
        "save_troubleshooting_step",
        "search_troubleshooting",
        "delete_troubleshooting_step",
    }


def test_namespace_is_team_scoped():
    from tools.troubleshooting import TROUBLESHOOTING_NAMESPACE

    assert TROUBLESHOOTING_NAMESPACE == ("troubleshooting", "virtualdojo")


# --- Save ---


async def test_save_round_trips_all_fields(_mem_store):
    from tools.troubleshooting import save_troubleshooting_step

    result = await save_troubleshooting_step.ainvoke(
        {
            "symptom": "API key rejected on POST /activities",
            "winning_hypothesis": "activities.py imports get_current_user from app.core.deps which is JWT-only",
            "discriminating_evidence": "search_repo_code found two get_current_user defs",
            "fix_location": "app/api/v1/endpoints/activities.py:12",
            "fix_description": "change import to from app.api.deps",
            "hypotheses_ruled_out": ["token expired", "tenant mismatch"],
            "repo": "virtualdojo-inc/virtualdojo",
            "github_issue": 522,
        }
    )
    assert "Saved" in result

    items = list(_mem_store.search(("troubleshooting", "virtualdojo"), query="api key"))
    assert len(items) == 1
    v = items[0].value
    assert v["symptom"] == "API key rejected on POST /activities"
    assert v["fix_location"] == "app/api/v1/endpoints/activities.py:12"
    assert v["hypotheses_ruled_out"] == ["token expired", "tenant mismatch"]
    assert v["repo"] == "virtualdojo-inc/virtualdojo"
    assert v["github_issue"] == 522
    assert v["source"] == "manual"
    assert v["retrieval_count"] == 0
    assert isinstance(v["created_at"], float)
    assert "API key rejected" in v["content"]
    assert "app.api.deps" in v["content"] or "app/api/v1" in v["content"]


async def test_save_handles_exception_gracefully(_mem_store):
    """Save must return an error string, never raise."""
    from tools.troubleshooting import save_troubleshooting_step

    with patch(
        "tools.troubleshooting._save_step",
        side_effect=RuntimeError("vertex quota"),
    ):
        result = await save_troubleshooting_step.ainvoke(
            {
                "symptom": "s",
                "winning_hypothesis": "h",
                "discriminating_evidence": "e",
                "fix_location": "f.py:1",
                "fix_description": "d",
            }
        )
    assert "Save failed" in result
    assert "vertex quota" in result


async def test_save_minimal_fields(_mem_store):
    from tools.troubleshooting import save_troubleshooting_step

    await save_troubleshooting_step.ainvoke(
        {
            "symptom": "X",
            "winning_hypothesis": "Y",
            "discriminating_evidence": "Z",
            "fix_location": "f:1",
            "fix_description": "D",
        }
    )
    items = list(_mem_store.search(("troubleshooting", "virtualdojo"), query="X"))
    assert len(items) == 1
    v = items[0].value
    assert v["hypotheses_ruled_out"] == []
    assert v["repo"] is None
    assert v["github_issue"] is None


# --- Search ---


async def test_search_returns_formatted_matches(_mem_store):
    from tools.troubleshooting import save_troubleshooting_step, search_troubleshooting

    await save_troubleshooting_step.ainvoke(
        {
            "symptom": "API key rejected on /activities",
            "winning_hypothesis": "wrong get_current_user import",
            "discriminating_evidence": "dup imports found",
            "fix_location": "activities.py:12",
            "fix_description": "switch module",
            "github_issue": 522,
        }
    )

    out = await search_troubleshooting.ainvoke({"query": "api key"})
    assert "API key rejected" in out
    assert "wrong get_current_user import" in out
    assert "activities.py:12" in out
    assert "issue #522" in out


async def test_search_no_matches(_mem_store):
    from tools.troubleshooting import search_troubleshooting

    out = await search_troubleshooting.ainvoke({"query": "anything"})
    assert "No troubleshooting patterns matched" in out


async def test_search_handles_exception_gracefully(_mem_store):
    from tools.troubleshooting import search_troubleshooting

    with patch("memory.get_memory_store", side_effect=RuntimeError("embed broken")):
        out = await search_troubleshooting.ainvoke({"query": "x"})
    assert "Search failed" in out
    assert "embed broken" in out


# --- Delete ---


async def test_delete_removes_step(_mem_store):
    from tools.troubleshooting import (
        save_troubleshooting_step,
        delete_troubleshooting_step,
        TROUBLESHOOTING_NAMESPACE,
    )

    await save_troubleshooting_step.ainvoke(
        {
            "symptom": "to be deleted",
            "winning_hypothesis": "y",
            "discriminating_evidence": "z",
            "fix_location": "f:1",
            "fix_description": "d",
        }
    )
    items = list(_mem_store.search(TROUBLESHOOTING_NAMESPACE, query="deleted"))
    assert len(items) == 1
    step_id = items[0].key

    del_result = await delete_troubleshooting_step.ainvoke({"step_id": step_id})
    assert "Deleted" in del_result

    remaining = list(_mem_store.search(TROUBLESHOOTING_NAMESPACE, query="deleted"))
    assert len(remaining) == 0


# --- Retrieval helper used by memory.retrieve_relevant_memories ---


async def test_retrieve_bumps_retrieval_count(_mem_store):
    from tools.troubleshooting import (
        save_troubleshooting_step,
        retrieve_troubleshooting_patterns,
        TROUBLESHOOTING_NAMESPACE,
    )

    await save_troubleshooting_step.ainvoke(
        {
            "symptom": "counting test",
            "winning_hypothesis": "h",
            "discriminating_evidence": "e",
            "fix_location": "f:1",
            "fix_description": "d",
        }
    )

    await retrieve_troubleshooting_patterns("counting", limit=3)
    await retrieve_troubleshooting_patterns("counting", limit=3)
    await retrieve_troubleshooting_patterns("counting", limit=3)

    items = list(_mem_store.search(TROUBLESHOOTING_NAMESPACE, query="counting"))
    assert items[0].value["retrieval_count"] == 3


async def test_retrieve_returns_none_when_empty(_mem_store):
    from tools.troubleshooting import retrieve_troubleshooting_patterns

    assert await retrieve_troubleshooting_patterns("anything") is None


async def test_retrieve_includes_age_hint(_mem_store):
    from tools.troubleshooting import (
        save_troubleshooting_step,
        retrieve_troubleshooting_patterns,
        TROUBLESHOOTING_NAMESPACE,
    )

    await save_troubleshooting_step.ainvoke(
        {
            "symptom": "age hint test",
            "winning_hypothesis": "h",
            "discriminating_evidence": "e",
            "fix_location": "f:1",
            "fix_description": "d",
        }
    )
    items = list(_mem_store.search(TROUBLESHOOTING_NAMESPACE, query="age hint"))
    step_id = items[0].key
    val = dict(items[0].value)
    val["created_at"] = time.time() - 10 * 86400
    _mem_store.put(TROUBLESHOOTING_NAMESPACE, step_id, val)

    out = await retrieve_troubleshooting_patterns("age hint")
    assert out is not None
    assert "saved 10d ago" in out


async def test_namespace_isolation_from_core_and_team(_mem_store):
    """Troubleshooting entries must NOT appear in core or team searches."""
    from tools.troubleshooting import (
        save_troubleshooting_step,
        TROUBLESHOOTING_NAMESPACE,
    )
    from memory import CORE_NAMESPACE, TEAM_NAMESPACE

    await save_troubleshooting_step.ainvoke(
        {
            "symptom": "isolation test symptom",
            "winning_hypothesis": "h",
            "discriminating_evidence": "e",
            "fix_location": "f:1",
            "fix_description": "d",
        }
    )

    _mem_store.put(CORE_NAMESPACE, "c1", {"content": "core thing"})
    _mem_store.put(TEAM_NAMESPACE, "t1", {"content": "team thing"})

    ts_results = list(_mem_store.search(TROUBLESHOOTING_NAMESPACE, query="isolation"))
    core_results = list(_mem_store.search(CORE_NAMESPACE, query="isolation"))
    team_results = list(_mem_store.search(TEAM_NAMESPACE, query="isolation"))

    assert any(r.value.get("symptom") == "isolation test symptom" for r in ts_results)
    assert not any(r.value.get("symptom") == "isolation test symptom" for r in core_results)
    assert not any(r.value.get("symptom") == "isolation test symptom" for r in team_results)
