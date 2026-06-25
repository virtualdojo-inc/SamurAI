"""Async SQLAlchemy engine + session factory for SamurAI's Postgres backbone.

Mirrors the CMO service's ``db/session.py``. SamurAI has no settings module (it
uses ``os.environ`` directly, like ``task_store.py``), so the URL is read from
``DATABASE_URL`` — injected from the ``samurai-database-url`` secret (the
in-boundary Cloud SQL instance ``samurai-db``).
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Wire the GCP secret 'samurai-database-url' "
            "into the Cloud Run service (the in-boundary Cloud SQL instance samurai-db)."
        )
    return url


def init_engine(database_url: str | None = None) -> AsyncEngine:
    """Create (or return cached) async engine bound to the configured database URL."""
    global _engine, _sessionmaker
    if _engine is not None:
        return _engine
    # NullPool: this engine backs only low-frequency callers (the /admin
    # endpoint, the one-shot data migration, the code sandbox's CodeRun
    # writes). A persistent pool here held connections open idle for hours
    # (SQLAlchemy's default pool never recycles idle connections), eating into
    # samurai-db's 50-connection budget for no benefit. NullPool opens a
    # connection per use and closes it on release — zero idle connections.
    _engine = create_async_engine(
        database_url or _database_url(), poolclass=NullPool, future=True
    )
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        init_engine()
    assert _sessionmaker is not None
    return _sessionmaker


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    maker = get_sessionmaker()
    async with maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
