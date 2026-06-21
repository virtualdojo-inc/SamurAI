"""Integration test: the Postgres LangGraph checkpointer.

Runs only when TEST_DATABASE_URL points at a real Postgres. Verifies
get_checkpointer() returns a working AsyncPostgresSaver (durable, shared)
instead of the per-instance /tmp SQLite saver.
"""
import os

import pytest

PG_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not PG_URL, reason="set TEST_DATABASE_URL to a Postgres URL to run"
)


async def test_postgres_checkpointer_round_trip(monkeypatch):
    import memory

    monkeypatch.setenv("DATABASE_URL", PG_URL)
    # reset singletons so get_checkpointer rebuilds against Postgres
    memory._checkpointer = None
    memory._checkpoint_pool = None
    memory._checkpoint_conn = None

    cp = await memory.get_checkpointer()
    # It must be the Postgres saver, not the SQLite/in-memory fallback.
    assert type(cp).__name__ == "AsyncPostgresSaver"

    from langgraph.checkpoint.base import empty_checkpoint

    cfg = {"configurable": {"thread_id": "pytest-thread", "checkpoint_ns": ""}}
    await cp.aput(cfg, empty_checkpoint(), {}, {})
    got = await cp.aget_tuple(cfg)
    assert got is not None

    if memory._checkpoint_pool is not None:
        await memory._checkpoint_pool.close()
    memory._checkpointer = None
    memory._checkpoint_pool = None
