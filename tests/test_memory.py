"""Tests for memory.py — LangMem InMemoryStore with SQLite persistence."""

import json
import os
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset memory module singletons between tests."""
    import memory

    memory._store = None
    memory._checkpointer = None
    memory._checkpoint_conn = None
    memory._background_executor = None
    memory._core_executor = None
    memory._team_executor = None
    yield
    memory._store = None
    memory._checkpointer = None
    memory._checkpoint_conn = None
    memory._background_executor = None
    memory._core_executor = None
    memory._team_executor = None


# --- Embedding function (Vertex AI) ---


def test_embed_fn_uses_vertex_ai_not_gemini_api():
    """Memory uses Vertex AI via service-account auth, NOT the Gemini Developer
    API which requires GOOGLE_API_KEY. This is a regression guard: the bot's
    service account cannot authenticate against the Gemini Developer API, so a
    regression here would silently break memory persistence for every user."""
    from memory import _create_embed_fn

    mock_instance = MagicMock()
    mock_instance.embed_documents.return_value = [[0.2] * 768]

    with patch(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        return_value=mock_instance,
    ) as mock_cls:
        embed = _create_embed_fn()
        embed(["hello"])  # triggers lazy init

    assert mock_cls.call_count == 1
    kwargs = mock_cls.call_args.kwargs
    assert kwargs.get("model") == "text-embedding-005"
    # THE CRITICAL FLAG: without vertexai=True the class defaults to the
    # Gemini Developer API and will reject the service-account token.
    assert kwargs.get("vertexai") is True
    assert "project" in kwargs
    assert "location" in kwargs
    # Service-account path — no API key should be passed
    assert "api_key" not in kwargs
    assert "google_api_key" not in kwargs


def test_embed_fn_lazy_and_reused():
    """The underlying embeddings client should be instantiated once and reused
    for subsequent calls, to avoid auth churn."""
    from memory import _create_embed_fn

    mock_instance = MagicMock()
    mock_instance.embed_documents.return_value = [[0.2] * 768]

    with patch(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings",
        return_value=mock_instance,
    ) as mock_cls:
        embed = _create_embed_fn()
        embed(["one"])
        embed(["two"])
        embed(["three"])

    assert mock_cls.call_count == 1  # Lazy + reused
    assert mock_instance.embed_documents.call_count == 3


# --- InMemoryStore ---


def test_get_memory_store_returns_store():
    from memory import get_memory_store

    with patch("memory._create_embed_fn", return_value=lambda texts: [[0.1] * 768 for _ in texts]):
        store = get_memory_store()
    assert store is not None


def test_get_memory_store_is_singleton():
    from memory import get_memory_store

    with patch("memory._create_embed_fn", return_value=lambda texts: [[0.1] * 768 for _ in texts]):
        store1 = get_memory_store()
        store2 = get_memory_store()
    assert store1 is store2


def test_store_put_and_search():
    from memory import get_memory_store

    with patch("memory._create_embed_fn", return_value=lambda texts: [[0.1] * 768 for _ in texts]):
        store = get_memory_store()

    store.put(("memories", "user1"), "mem1", {"content": "Devin likes Python"})
    results = store.search(("memories", "user1"), query="Python")
    assert len(results) > 0
    assert results[0].value["content"] == "Devin likes Python"


def test_store_user_isolation():
    from memory import get_memory_store

    with patch("memory._create_embed_fn", return_value=lambda texts: [[0.1] * 768 for _ in texts]):
        store = get_memory_store()

    store.put(("memories", "user-a"), "m1", {"content": "secret A"})
    store.put(("memories", "user-b"), "m2", {"content": "secret B"})

    results_a = store.search(("memories", "user-a"), query="secret")
    results_b = store.search(("memories", "user-b"), query="secret")

    assert all(r.value["content"] == "secret A" for r in results_a)
    assert all(r.value["content"] == "secret B" for r in results_b)


# --- SQLite Persistence ---


def test_persist_and_load_memories(tmp_path):
    import memory

    memory.MEMORY_DB_PATH = str(tmp_path / "test_memories.sqlite")
    memory.DATA_DIR = str(tmp_path)

    with patch("memory._create_embed_fn", return_value=lambda texts: [[0.1] * 768 for _ in texts]):
        store = memory.get_memory_store()

    store.put(("memories", "user1"), "m1", {"content": "fact one"})
    store.put(("memories", "user1"), "m2", {"content": "fact two"})

    memory.persist_memories()

    # Verify SQLite has the data
    conn = sqlite3.connect(memory.MEMORY_DB_PATH)
    rows = conn.execute("SELECT * FROM memories").fetchall()
    conn.close()
    assert len(rows) == 2

    # Reset store and reload
    memory._store = None
    with patch("memory._create_embed_fn", return_value=lambda texts: [[0.1] * 768 for _ in texts]):
        store2 = memory.get_memory_store()

    results = store2.search(("memories", "user1"), query="fact")
    assert len(results) == 2


def test_persist_no_op_when_no_store():
    from memory import persist_memories

    persist_memories()  # Should not raise


# --- Checkpointer ---


@pytest.mark.asyncio
async def test_get_checkpointer_creates_sqlite_saver(tmp_path):
    import memory

    memory.CHECKPOINT_DB_PATH = str(tmp_path / "test_checkpoints.sqlite")
    memory._checkpointer = None
    memory._checkpoint_conn = None

    ckpt = await memory.get_checkpointer()
    assert ckpt is not None


@pytest.mark.asyncio
async def test_get_checkpointer_is_singleton(tmp_path):
    import memory

    memory.CHECKPOINT_DB_PATH = str(tmp_path / "test_checkpoints.sqlite")
    memory._checkpointer = None
    memory._checkpoint_conn = None

    ckpt1 = await memory.get_checkpointer()
    ckpt2 = await memory.get_checkpointer()
    assert ckpt1 is ckpt2


@pytest.mark.asyncio
async def test_get_checkpointer_falls_back_to_memory_saver():
    import memory

    memory.CHECKPOINT_DB_PATH = "/nonexistent/path/checkpoints.sqlite"
    memory._checkpointer = None
    memory._checkpoint_conn = None

    ckpt = await memory.get_checkpointer()
    from langgraph.checkpoint.memory import MemorySaver

    assert isinstance(ckpt, MemorySaver)


# --- LangMem Memory Tools ---


def test_create_memory_tools_returns_six_tools():
    from memory import create_memory_tools

    with patch("memory.get_memory_store", return_value=MagicMock()):
        tools = create_memory_tools("test-user")

    assert len(tools) == 6
    names = {t.name for t in tools}
    # User tier
    assert "manage_memory" in names
    assert "search_memory" in names
    # Core tier
    assert "manage_core_memory" in names
    assert "search_core_memory" in names
    # Team tier
    assert "manage_team_memory" in names
    assert "search_team_memory" in names


# --- Auto-Retrieval ---


@pytest.mark.asyncio
async def test_retrieve_relevant_memories_with_results():
    from memory import retrieve_relevant_memories, CORE_NAMESPACE, TEAM_NAMESPACE

    mock_store = MagicMock()
    core_result = MagicMock()
    core_result.value = {"content": "Check PRs for bugfix/issue-N branches"}
    team_result = MagicMock()
    team_result.value = {"content": "quotely-data-service uses autofix label"}
    user_result = MagicMock()
    user_result.value = {"content": "Devin prefers short responses"}

    def _search(namespace, query, limit=3):
        if namespace == CORE_NAMESPACE:
            return [core_result]
        if namespace == TEAM_NAMESPACE:
            return [team_result]
        return [user_result]

    mock_store.search.side_effect = _search

    with patch("memory.get_memory_store", return_value=mock_store):
        result = await retrieve_relevant_memories("user1", "autofix status")

    assert result is not None
    assert "Operational knowledge:" in result
    assert "bugfix/issue-N" in result
    assert "Team knowledge:" in result
    assert "autofix label" in result
    assert "Personal context:" in result
    assert "Devin prefers short responses" in result


@pytest.mark.asyncio
async def test_retrieve_relevant_memories_empty():
    from memory import retrieve_relevant_memories

    mock_store = MagicMock()
    mock_store.search.return_value = []

    with patch("memory.get_memory_store", return_value=mock_store):
        result = await retrieve_relevant_memories("user1", "anything")

    assert result is None


@pytest.mark.asyncio
async def test_retrieve_relevant_memories_emits_observability_log(capsys):
    """Every retrieval emits a single [memory] retrieved line to stdout so
    Cloud Logging can show what was injected into the system prompt."""
    from memory import retrieve_relevant_memories, CORE_NAMESPACE, TEAM_NAMESPACE

    mock_store = MagicMock()
    core_result = MagicMock()
    core_result.value = {"content": "recipe"}
    team_result = MagicMock()
    team_result.value = {"content": "team fact"}

    def _search(namespace, query, limit=3):
        if namespace == CORE_NAMESPACE:
            return [core_result]
        if namespace == TEAM_NAMESPACE:
            return [team_result, team_result]
        return []

    mock_store.search.side_effect = _search

    with (
        patch("memory.get_memory_store", return_value=mock_store),
        patch(
            "tools.troubleshooting.retrieve_troubleshooting_patterns",
            return_value="Prior troubleshooting patterns:\n- first\n- second\n- third",
        ),
    ):
        await retrieve_relevant_memories("user-abc", "api key failure on activities")

    out = capsys.readouterr().out
    assert "[memory] retrieved" in out
    assert "core=1" in out
    assert "team=2" in out
    assert "user=0" in out
    assert "troubleshooting=3" in out
    assert "user_id='user-abc'" in out
    assert "api key failure on activities" in out


@pytest.mark.asyncio
async def test_retrieve_log_fires_even_when_nothing_matches(capsys):
    """The log line must always emit — zero hits is also useful signal."""
    from memory import retrieve_relevant_memories

    mock_store = MagicMock()
    mock_store.search.return_value = []

    with (
        patch("memory.get_memory_store", return_value=mock_store),
        patch(
            "tools.troubleshooting.retrieve_troubleshooting_patterns",
            return_value=None,
        ),
    ):
        await retrieve_relevant_memories("user1", "no matches expected")

    out = capsys.readouterr().out
    assert "[memory] retrieved" in out
    assert "core=0 team=0 user=0 troubleshooting=0" in out


@pytest.mark.asyncio
async def test_retrieve_relevant_memories_partial_tiers():
    """Only tiers with results should appear in the output."""
    from memory import retrieve_relevant_memories, CORE_NAMESPACE, TEAM_NAMESPACE

    mock_store = MagicMock()
    core_result = MagicMock()
    core_result.value = {"content": "Use gcloud logging read for Cloud Run errors"}

    def _search(namespace, query, limit=3):
        if namespace == CORE_NAMESPACE:
            return [core_result]
        return []

    mock_store.search.side_effect = _search

    with patch("memory.get_memory_store", return_value=mock_store):
        result = await retrieve_relevant_memories("user1", "cloud run errors")

    assert result is not None
    assert "Operational knowledge:" in result
    assert "Team knowledge:" not in result
    assert "Personal context:" not in result


@pytest.mark.asyncio
async def test_retrieve_relevant_memories_handles_error():
    from memory import retrieve_relevant_memories

    with patch("memory.get_memory_store", side_effect=Exception("boom")):
        result = await retrieve_relevant_memories("user1", "anything")

    assert result is None


# --- Background Extractor ---


def test_get_background_extractor_returns_executor():
    from memory import get_background_extractor

    with (
        patch("memory.get_memory_store", return_value=MagicMock()),
        patch("memory._create_extractor_llm", return_value=MagicMock()),
        patch("langmem.create_memory_store_manager", return_value=MagicMock()),
        patch("langmem.ReflectionExecutor") as mock_executor_cls,
    ):
        mock_executor_cls.return_value = MagicMock()
        executor = get_background_extractor()

    assert executor is not None


def test_get_background_extractor_is_singleton():
    from memory import get_background_extractor

    with (
        patch("memory.get_memory_store", return_value=MagicMock()),
        patch("memory._create_extractor_llm", return_value=MagicMock()),
        patch("langmem.create_memory_store_manager", return_value=MagicMock()),
        patch("langmem.ReflectionExecutor") as mock_executor_cls,
    ):
        mock_executor_cls.return_value = MagicMock()
        e1 = get_background_extractor()
        e2 = get_background_extractor()

    assert e1 is e2


# --- Core Extractor ---


def test_get_core_extractor_returns_executor():
    from memory import get_core_extractor

    with (
        patch("memory.get_memory_store", return_value=MagicMock()),
        patch("memory._create_extractor_llm", return_value=MagicMock()),
        patch("langmem.create_memory_store_manager", return_value=MagicMock()),
        patch("langmem.ReflectionExecutor") as mock_executor_cls,
    ):
        mock_executor_cls.return_value = MagicMock()
        executor = get_core_extractor()

    assert executor is not None


def test_get_core_extractor_is_singleton():
    from memory import get_core_extractor

    with (
        patch("memory.get_memory_store", return_value=MagicMock()),
        patch("memory._create_extractor_llm", return_value=MagicMock()),
        patch("langmem.create_memory_store_manager", return_value=MagicMock()),
        patch("langmem.ReflectionExecutor") as mock_executor_cls,
    ):
        mock_executor_cls.return_value = MagicMock()
        e1 = get_core_extractor()
        e2 = get_core_extractor()

    assert e1 is e2


# --- Team Extractor ---


def test_get_team_extractor_returns_executor():
    from memory import get_team_extractor

    with (
        patch("memory.get_memory_store", return_value=MagicMock()),
        patch("memory._create_extractor_llm", return_value=MagicMock()),
        patch("langmem.create_memory_store_manager", return_value=MagicMock()),
        patch("langmem.ReflectionExecutor") as mock_executor_cls,
    ):
        mock_executor_cls.return_value = MagicMock()
        executor = get_team_extractor()

    assert executor is not None


def test_get_team_extractor_is_singleton():
    from memory import get_team_extractor

    with (
        patch("memory.get_memory_store", return_value=MagicMock()),
        patch("memory._create_extractor_llm", return_value=MagicMock()),
        patch("langmem.create_memory_store_manager", return_value=MagicMock()),
        patch("langmem.ReflectionExecutor") as mock_executor_cls,
    ):
        mock_executor_cls.return_value = MagicMock()
        e1 = get_team_extractor()
        e2 = get_team_extractor()

    assert e1 is e2


# --- Vertex-auth regression for the LangMem extractors (May 2026) ---


def _capture_extractor_llm(getter_name: str, env: dict | None = None):
    """Call one of the extractor singletons with everything mocked and return
    the model arg that was passed to create_memory_store_manager."""
    import memory

    chat_cls = MagicMock(return_value="fake-llm-client")
    env_patch = patch.dict(
        os.environ,
        env if env is not None else {"GCP_PROJECT_ID": "vd-test", "GCP_LOCATION": "us-east1"},
        clear=False,
    )
    with (
        env_patch,
        patch("memory.get_memory_store", return_value=MagicMock()),
        patch("langchain_google_genai.ChatGoogleGenerativeAI", chat_cls),
        patch("langmem.create_memory_store_manager") as mock_manager,
        patch("langmem.ReflectionExecutor", return_value=MagicMock()),
    ):
        mock_manager.return_value = MagicMock()
        getattr(memory, getter_name)()
        assert mock_manager.called, f"{getter_name} did not call create_memory_store_manager"
        model_arg = mock_manager.call_args.args[0]
    return chat_cls, model_arg


def test_background_extractor_uses_vertex_chat_client():
    """Must construct ChatGoogleGenerativeAI with vertexai=True, not pass a
    'google_genai:...' string (which routes through the Developer API and
    requires GOOGLE_API_KEY — Cloud Run authenticates via service account)."""
    chat_cls, model_arg = _capture_extractor_llm("get_background_extractor")
    assert model_arg == "fake-llm-client"
    chat_cls.assert_called_once()
    kwargs = chat_cls.call_args.kwargs
    assert kwargs.get("vertexai") is True
    assert kwargs.get("model") == "gemini-2.0-flash-lite"
    assert kwargs.get("project") == "vd-test"
    assert kwargs.get("location") == "us-east1"


def test_core_extractor_uses_vertex_chat_client():
    chat_cls, model_arg = _capture_extractor_llm("get_core_extractor")
    assert model_arg == "fake-llm-client"
    chat_cls.assert_called_once()
    assert chat_cls.call_args.kwargs.get("vertexai") is True


def test_team_extractor_uses_vertex_chat_client():
    chat_cls, model_arg = _capture_extractor_llm("get_team_extractor")
    assert model_arg == "fake-llm-client"
    chat_cls.assert_called_once()
    assert chat_cls.call_args.kwargs.get("vertexai") is True


def test_extractor_location_defaults_to_us_central1():
    """If GCP_LOCATION is unset, the extractor should fall back to us-central1
    (matching the bot's primary region)."""
    chat_cls, _ = _capture_extractor_llm(
        "get_background_extractor",
        env={"GCP_PROJECT_ID": "vd-test"},  # GCP_LOCATION deliberately omitted
    )
    assert chat_cls.call_args.kwargs.get("location") == "us-central1"


# --- ReflectionExecutor store= regression (2026-05-25 incident) ---


def _capture_reflection_args(getter_name: str):
    """Call an extractor and capture how ReflectionExecutor was invoked."""
    import memory

    chat_cls = MagicMock(return_value="fake-llm-client")
    fake_store = MagicMock(name="fake-store")

    with (
        patch.dict(
            os.environ,
            {"GCP_PROJECT_ID": "vd-test", "GCP_LOCATION": "us-central1"},
            clear=False,
        ),
        patch("memory.get_memory_store", return_value=fake_store),
        patch("langchain_google_genai.ChatGoogleGenerativeAI", chat_cls),
        patch("langmem.create_memory_store_manager") as mock_manager,
        patch("langmem.ReflectionExecutor") as mock_executor_cls,
    ):
        mock_manager.return_value = MagicMock(name="fake-manager")
        mock_executor_cls.return_value = MagicMock()
        getattr(memory, getter_name)()
        assert mock_executor_cls.called, (
            f"{getter_name} did not call ReflectionExecutor"
        )
        return mock_executor_cls.call_args, fake_store


def test_background_extractor_passes_store_to_reflection_executor():
    """Regression: LangMem's ReflectionExecutor needs `store=` even when the
    underlying manager already has one. Without it every conversation logs
    "ReflectionExecutor could not resolve store to persist memories to"
    and no user memories are saved."""
    call, fake_store = _capture_reflection_args("get_background_extractor")
    assert call.kwargs.get("store") is fake_store, (
        f"ReflectionExecutor was called without store=. Args: {call}"
    )


def test_core_extractor_passes_store_to_reflection_executor():
    call, fake_store = _capture_reflection_args("get_core_extractor")
    assert call.kwargs.get("store") is fake_store


def test_team_extractor_passes_store_to_reflection_executor():
    call, fake_store = _capture_reflection_args("get_team_extractor")
    assert call.kwargs.get("store") is fake_store
