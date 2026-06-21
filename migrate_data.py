"""One-time data migration: the old /data SQLite stores -> Postgres.

Must run INSIDE the VPC (a Cloud Run Job, or the candidate revision) where both
the /data gcsfuse mount and the private-IP Cloud SQL instance are reachable —
a laptop can't reach the private DB. Requires DATABASE_URL (the Postgres target).

Idempotent:
  - tasks / conversation_refs / team_roster: upsert by primary key (session.merge).
  - memories: upsert by (namespace, key) via store.aput (re-embeds once into pgvector).

Run: ``python migrate_data.py`` (with DATABASE_URL set).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migrate_data")

DATA_DIR = os.environ.get("SAMURAI_DATA_DIR", "/data")
TASKS_SQLITE = os.path.join(DATA_DIR, "tasks.sqlite")
MEM_SQLITE = os.path.join(DATA_DIR, "langmem_memories.sqlite")


def _require_postgres() -> None:
    url = os.environ.get("DATABASE_URL", "")
    if "postgresql" not in url:
        raise RuntimeError(
            "DATABASE_URL must point at Postgres to run the migration "
            f"(got {url[:30]!r}...). Refusing to run."
        )


async def migrate_tasks(path: str | None = None) -> dict:
    """Copy tasks / conversation_refs / team_roster from old SQLite into Postgres."""
    path = path or TASKS_SQLITE
    if not os.path.exists(path):
        return {"tasks": 0, "conversation_refs": 0, "team_roster": 0, "note": "no sqlite file"}

    from db import session as dbsession
    from db.models import Base, ConversationRef, TASK_STORE_TABLES, Task, TeamRoster

    dbsession.init_engine()  # DATABASE_URL (Postgres)
    engine = dbsession.init_engine()
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda c: Base.metadata.create_all(c, tables=TASK_STORE_TABLES, checkfirst=True)
        )

    old = sqlite3.connect(path)
    old.row_factory = sqlite3.Row
    counts = {"tasks": 0, "conversation_refs": 0, "team_roster": 0}
    maker = dbsession.get_sessionmaker()

    def _rows(table: str) -> list[dict]:
        try:
            return [dict(r) for r in old.execute(f"SELECT * FROM {table}")]
        except sqlite3.OperationalError:
            return []  # table absent in old DB

    async with maker() as s:
        for r in _rows("tasks"):
            await s.merge(Task(**{k: r[k] for k in r if k in Task.__table__.columns}))
            counts["tasks"] += 1
        for r in _rows("conversation_refs"):
            await s.merge(ConversationRef(**{k: r[k] for k in r if k in ConversationRef.__table__.columns}))
            counts["conversation_refs"] += 1
        for r in _rows("team_roster"):
            await s.merge(TeamRoster(**{k: r[k] for k in r if k in TeamRoster.__table__.columns}))
            counts["team_roster"] += 1
        await s.commit()

    old.close()
    logger.info("[migrate] tasks=%(tasks)d refs=%(conversation_refs)d roster=%(team_roster)d", counts)
    return counts


async def migrate_memories(path: str | None = None) -> dict:
    """Copy LangMem memories from old SQLite into the Postgres pgvector store.

    Each aput re-embeds the value once (Vertex) and stores it in pgvector — a
    one-time cost that removes the per-boot re-embedding the InMemoryStore did.
    """
    path = path or MEM_SQLITE
    if not os.path.exists(path):
        return {"memories": 0, "note": "no sqlite file"}

    from memory import get_memory_store

    store = await get_memory_store()  # AsyncPostgresStore (DATABASE_URL set)
    old = sqlite3.connect(path)
    n = 0
    try:
        cur = old.execute("SELECT namespace, key, value_json FROM memories")
    except sqlite3.OperationalError:
        old.close()
        return {"memories": 0, "note": "no memories table"}

    for ns_json, key, value_json in cur:
        try:
            ns = tuple(json.loads(ns_json))
            value = json.loads(value_json)
            await store.aput(ns, key, value)
            n += 1
            if n % 250 == 0:
                logger.info("[migrate] memories migrated so far: %d", n)
        except Exception as e:  # one bad row must not abort the run
            logger.warning("[migrate] skipped a memory (%s): %s", key, e)
    old.close()
    logger.info("[migrate] memories=%d", n)
    return {"memories": n}


async def run() -> dict:
    _require_postgres()
    tasks = await migrate_tasks()
    mems = await migrate_memories()
    result = {**tasks, **mems}
    logger.info("[migrate] DONE: %s", result)
    print(f"[migrate] DONE: {result}", flush=True)
    return result


if __name__ == "__main__":
    asyncio.run(run())
