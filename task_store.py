"""Persistence for background tasks, conversation references, and team roster.

Migrated from raw aiosqlite to async SQLAlchemy so the SAME code runs on:
  - SQLite (tests/local — when no DATABASE_URL is set), and
  - the in-boundary Cloud SQL Postgres instance `samurai-db` (prod — DATABASE_URL).

The public API (TaskStore + get_task_store + the dict-returning methods) is
unchanged, so scheduler.py / tools/background_tasks.py / app.py are untouched.
``TaskStore(db_path)`` still accepts a SQLite file path (used by the test suite);
``get_task_store()`` uses DATABASE_URL in prod, falling back to the SQLite file.
"""

import logging
import os
import time
import uuid

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.models import (
    Base,
    ConversationRef,
    TASK_STORE_TABLES,
    Task,
    TeamRoster,
)

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("SAMURAI_DATA_DIR", "/data")
TASK_DB_PATH = os.path.join(DATA_DIR, "tasks.sqlite")

_task_store = None


def _row_to_dict(obj) -> dict:
    """Map an ORM row to the plain dict shape the old aiosqlite store returned."""
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}


class TaskStore:
    """Background tasks, conversation references, and team roster (async SQLAlchemy)."""

    def __init__(self, db_path: str):
        # Accept either a SQLAlchemy URL (e.g. postgresql+asyncpg://…) or a plain
        # SQLite file path (the historical constructor arg, used by tests).
        self.db_path = db_path
        self._url = db_path if "://" in db_path else f"sqlite+aiosqlite:///{db_path}"
        self._engine = None
        self._sessionmaker: async_sessionmaker | None = None

    async def initialize(self) -> None:
        """Create the engine + the task-store tables. Idempotent."""
        if self._engine is None:
            self._engine = create_async_engine(self._url, future=True)
            self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)
        async with self._engine.begin() as conn:
            await conn.run_sync(
                lambda c: Base.metadata.create_all(c, tables=TASK_STORE_TABLES, checkfirst=True)
            )
        logger.info("Task store initialized: %s", self.db_path)

    # ── Task CRUD ──────────────────────────────────────────────────────

    async def create_task(
        self,
        user_id: str,
        user_name: str,
        user_email: str,
        user_timezone: str,
        conversation_id: str,
        task_type: str,
        prompt: str,
        cron_expression: str | None = None,
        run_at: str | None = None,
        max_failures: int = 3,
    ) -> dict:
        task_id = str(uuid.uuid4())[:8]
        obj = Task(
            id=task_id,
            user_id=user_id,
            user_name=user_name,
            user_email=user_email,
            user_timezone=user_timezone,
            conversation_id=conversation_id,
            task_type=task_type,
            prompt=prompt,
            cron_expression=cron_expression,
            run_at=run_at,
            status="active",
            created_at=time.time(),
            run_count=0,
            error_count=0,
            max_failures=max_failures,
            locked_until=0.0,
        )
        async with self._sessionmaker() as session:
            session.add(obj)
            await session.commit()
        logger.info("Created task %s: %s", task_id, prompt[:60])
        return _row_to_dict(obj)

    async def get_task(self, task_id: str) -> dict | None:
        async with self._sessionmaker() as session:
            obj = await session.get(Task, task_id)
            return _row_to_dict(obj) if obj else None

    async def list_tasks(
        self, user_id: str | None = None, status: str | None = None
    ) -> list[dict]:
        stmt = select(Task)
        if user_id:
            stmt = stmt.where(Task.user_id == user_id)
        if status:
            stmt = stmt.where(Task.status == status)
        stmt = stmt.order_by(Task.created_at.desc())
        async with self._sessionmaker() as session:
            result = await session.execute(stmt)
            return [_row_to_dict(o) for o in result.scalars().all()]

    async def update_task(self, task_id: str, **fields) -> bool:
        if not fields:
            return False
        async with self._sessionmaker() as session:
            result = await session.execute(
                update(Task).where(Task.id == task_id).values(**fields)
            )
            await session.commit()
            return result.rowcount > 0

    async def delete_task(self, task_id: str) -> bool:
        async with self._sessionmaker() as session:
            result = await session.execute(delete(Task).where(Task.id == task_id))
            await session.commit()
            return result.rowcount > 0

    async def try_lock(self, task_id: str, lock_duration: float = 300) -> bool:
        """Atomically lock a task for execution (UPDATE ... WHERE locked_until < now)."""
        now = time.time()
        async with self._sessionmaker() as session:
            result = await session.execute(
                update(Task)
                .where(Task.id == task_id, Task.locked_until < now)
                .values(locked_until=now + lock_duration)
            )
            await session.commit()
            return result.rowcount > 0

    async def unlock(self, task_id: str) -> None:
        async with self._sessionmaker() as session:
            await session.execute(
                update(Task).where(Task.id == task_id).values(locked_until=0)
            )
            await session.commit()

    async def record_run(
        self, task_id: str, success: bool, error_message: str | None = None
    ) -> dict | None:
        """After execution: update run_count, handle errors, auto-pause if needed."""
        task = await self.get_task(task_id)
        if not task:
            return None

        now = time.time()
        updates: dict = {"last_run_at": now, "run_count": task["run_count"] + 1}

        if success:
            updates["error_count"] = 0
            updates["last_error"] = None
            if task["task_type"] == "one_shot":
                updates["status"] = "completed"
        else:
            new_error_count = task["error_count"] + 1
            updates["error_count"] = new_error_count
            updates["last_error"] = (error_message or "Unknown error")[:500]
            if new_error_count >= task["max_failures"]:
                updates["status"] = "failed"
                logger.warning(
                    "Task %s auto-paused after %d consecutive failures",
                    task_id,
                    new_error_count,
                )

        await self.update_task(task_id, **updates)
        task.update(updates)
        return task

    # ── Conversation References ────────────────────────────────────────

    async def save_conversation_ref(
        self, conversation_id: str, user_id: str, ref_json: str
    ) -> None:
        async with self._sessionmaker() as session:
            await session.merge(
                ConversationRef(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    ref_json=ref_json,
                    updated_at=time.time(),
                )
            )
            await session.commit()

    async def get_conversation_ref(self, conversation_id: str) -> str | None:
        async with self._sessionmaker() as session:
            result = await session.execute(
                select(ConversationRef.ref_json).where(
                    ConversationRef.conversation_id == conversation_id
                )
            )
            return result.scalar_one_or_none()

    # ── Team Roster ────────────────────────────────────────────────────

    async def save_team_member(
        self,
        email: str,
        teams_id: str,
        display_name: str,
        service_url: str,
        tenant_id: str = "",
    ) -> None:
        async with self._sessionmaker() as session:
            await session.merge(
                TeamRoster(
                    email=email.lower(),
                    teams_id=teams_id,
                    display_name=display_name,
                    service_url=service_url,
                    tenant_id=tenant_id,
                    updated_at=time.time(),
                )
            )
            await session.commit()

    async def get_team_member(self, email: str) -> dict | None:
        async with self._sessionmaker() as session:
            obj = await session.get(TeamRoster, email.lower())
            return _row_to_dict(obj) if obj else None

    async def list_team_members(self) -> list[dict]:
        async with self._sessionmaker() as session:
            result = await session.execute(
                select(TeamRoster).order_by(TeamRoster.display_name)
            )
            return [_row_to_dict(o) for o in result.scalars().all()]


async def get_task_store() -> TaskStore:
    """Get or create the singleton TaskStore.

    Uses DATABASE_URL (Postgres) in prod; falls back to the SQLite file on /data
    when DATABASE_URL is unset (tests/local).
    """
    global _task_store
    if _task_store is None:
        url = os.environ.get("DATABASE_URL")
        if url:
            _task_store = TaskStore(url)
        else:
            os.makedirs(DATA_DIR, exist_ok=True)
            _task_store = TaskStore(TASK_DB_PATH)
        await _task_store.initialize()
    return _task_store
