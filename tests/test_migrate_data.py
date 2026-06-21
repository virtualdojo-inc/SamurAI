"""Integration test for migrate_data.py — old /data SQLite -> Postgres.

Runs only when TEST_DATABASE_URL points at a real pgvector Postgres. Builds sample
old-format SQLite stores, runs the migration, and asserts the data lands in Postgres.
"""
import json
import os
import sqlite3
import uuid
from unittest.mock import patch

import pytest

PG_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not PG_URL, reason="set TEST_DATABASE_URL to a pgvector Postgres URL to run"
)


def _fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * 768
        v[hash(t) % 768] = 1.0
        out.append(v)
    return out


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    import memory
    import task_store
    from db import session as dbsession

    monkeypatch.setenv("DATABASE_URL", PG_URL)
    for mod, attr in (
        (memory, "_store"), (memory, "_store_pool"),
        (task_store, "_task_store"),
        (dbsession, "_engine"), (dbsession, "_sessionmaker"),
    ):
        setattr(mod, attr, None)
    yield
    for mod, attr in (
        (memory, "_store"), (memory, "_store_pool"),
        (task_store, "_task_store"),
        (dbsession, "_engine"), (dbsession, "_sessionmaker"),
    ):
        setattr(mod, attr, None)


def _make_old_sqlite(d, tid, conv, email, memkey):
    tdb = os.path.join(d, "tasks.sqlite")
    c = sqlite3.connect(tdb)
    c.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, user_id TEXT, user_name TEXT, "
        "user_email TEXT, user_timezone TEXT, conversation_id TEXT, task_type TEXT, "
        "prompt TEXT, cron_expression TEXT, run_at TEXT, status TEXT, created_at REAL, "
        "last_run_at REAL, next_run_at TEXT, run_count INTEGER, error_count INTEGER, "
        "last_error TEXT, max_failures INTEGER, locked_until REAL)"
    )
    c.execute(
        "INSERT INTO tasks (id,user_id,user_name,user_email,user_timezone,conversation_id,"
        "task_type,prompt,status,created_at,run_count,error_count,max_failures,locked_until) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (tid, "u", "N", "e@x.com", "UTC", "c", "recurring", "migrated task", "active",
         123.0, 0, 0, 3, 0.0),
    )
    c.execute("CREATE TABLE conversation_refs (conversation_id TEXT PRIMARY KEY, user_id TEXT, ref_json TEXT, updated_at REAL)")
    c.execute("INSERT INTO conversation_refs VALUES (?,?,?,?)", (conv, "u", '{"v":1}', 123.0))
    c.execute("CREATE TABLE team_roster (email TEXT PRIMARY KEY, teams_id TEXT, display_name TEXT, service_url TEXT, tenant_id TEXT, updated_at REAL)")
    c.execute("INSERT INTO team_roster VALUES (?,?,?,?,?,?)", (email, "tid", "Mig", "https://x", "tn", 123.0))
    c.commit(); c.close()

    mdb = os.path.join(d, "langmem_memories.sqlite")
    c = sqlite3.connect(mdb)
    c.execute("CREATE TABLE memories (namespace TEXT, key TEXT, value_json TEXT, created_at TEXT, updated_at TEXT, PRIMARY KEY (namespace,key))")
    c.execute(
        "INSERT INTO memories VALUES (?,?,?,?,?)",
        (json.dumps(["core"]), memkey, json.dumps({"content": "migrated memory about blue-green deploys"}), "", ""),
    )
    c.commit(); c.close()
    return tdb, mdb


async def test_migrate_round_trips_tasks_and_memories(tmp_path, monkeypatch):
    import migrate_data
    import memory
    from task_store import TaskStore

    tid = f"mig{uuid.uuid4().hex[:5]}"
    conv = f"migconv-{uuid.uuid4().hex[:6]}"
    email = f"{uuid.uuid4().hex[:6]}@mig.com"
    memkey = f"migmem-{uuid.uuid4().hex[:6]}"
    tdb, mdb = _make_old_sqlite(str(tmp_path), tid, conv, email, memkey)
    monkeypatch.setattr(migrate_data, "TASKS_SQLITE", tdb)
    monkeypatch.setattr(migrate_data, "MEM_SQLITE", mdb)

    with patch("memory._create_embed_fn", return_value=_fake_embed):
        result = await migrate_data.run()

    assert result["tasks"] >= 1
    assert result["conversation_refs"] >= 1
    assert result["team_roster"] >= 1
    assert result["memories"] >= 1

    # tasks/refs/roster landed in Postgres
    store = TaskStore(PG_URL)
    await store.initialize()
    assert (await store.get_task(tid))["prompt"] == "migrated task"
    assert await store.get_conversation_ref(conv) == '{"v":1}'
    assert (await store.get_team_member(email))["display_name"] == "Mig"

    # memory landed in pgvector
    with patch("memory._create_embed_fn", return_value=_fake_embed):
        mem_store = await memory.get_memory_store()
    hits = await mem_store.asearch(("core",), query="blue-green deploys", limit=5)
    assert any(h.key == memkey for h in hits)
