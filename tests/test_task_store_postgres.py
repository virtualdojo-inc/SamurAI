"""Integration test: TaskStore against a real Postgres.

Runs only when TEST_DATABASE_URL is set (a pgvector Postgres). Verifies the
dual-dialect SQLAlchemy TaskStore functions on Postgres, not just SQLite. Uses
unique keys so it is safely re-runnable against a shared DB.
"""
import os
import uuid

import pytest

from task_store import TaskStore

PG_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not PG_URL, reason="set TEST_DATABASE_URL to a Postgres URL to run"
)


@pytest.fixture
async def store():
    s = TaskStore(PG_URL)
    await s.initialize()
    return s


async def test_task_crud_on_postgres(store):
    uid = f"user-{uuid.uuid4().hex[:8]}"
    task = await store.create_task(
        user_id=uid, user_name="PG", user_email="pg@test.com", user_timezone="UTC",
        conversation_id="c1", task_type="recurring", prompt="pg task",
        cron_expression="0 9 * * *",
    )
    assert len(task["id"]) == 8
    assert task["status"] == "active"
    assert task["created_at"] > 0

    fetched = await store.get_task(task["id"])
    assert fetched["prompt"] == "pg task"

    mine = await store.list_tasks(user_id=uid)
    assert len(mine) == 1 and mine[0]["id"] == task["id"]

    assert await store.update_task(task["id"], status="paused") is True
    assert (await store.get_task(task["id"]))["status"] == "paused"

    # try_lock is the atomic UPDATE ... WHERE locked_until < now path
    assert await store.try_lock(task["id"]) is True
    assert await store.try_lock(task["id"]) is False  # already locked
    await store.unlock(task["id"])

    assert await store.delete_task(task["id"]) is True
    assert await store.get_task(task["id"]) is None


async def test_record_run_auto_pause_on_postgres(store):
    uid = f"user-{uuid.uuid4().hex[:8]}"
    task = await store.create_task(
        user_id=uid, user_name="PG", user_email="pg@test.com", user_timezone="UTC",
        conversation_id="c2", task_type="recurring", prompt="fragile",
    )
    await store.record_run(task["id"], success=False, error_message="e")
    await store.record_run(task["id"], success=False, error_message="e")
    updated = await store.record_run(task["id"], success=False, error_message="e")
    assert updated["status"] == "failed" and updated["error_count"] == 3
    await store.delete_task(task["id"])


async def test_conversation_ref_and_roster_on_postgres(store):
    cid = f"conv-{uuid.uuid4().hex[:8]}"
    await store.save_conversation_ref(cid, "u1", '{"v": 1}')
    await store.save_conversation_ref(cid, "u2", '{"v": 2}')  # upsert
    assert await store.get_conversation_ref(cid) == '{"v": 2}'

    email = f"{uuid.uuid4().hex[:8]}@test.com"
    await store.save_team_member(email=email.upper(), teams_id="t1", display_name="Z", service_url="https://x")
    m = await store.get_team_member(email)  # case-insensitive
    assert m["teams_id"] == "t1" and m["email"] == email.lower()
    assert any(x["email"] == email.lower() for x in await store.list_team_members())
