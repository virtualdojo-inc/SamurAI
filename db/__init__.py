"""SamurAI Postgres backbone (async SQLAlchemy + Alembic).

The in-boundary Cloud SQL instance ``samurai-db`` (private IP) replaces the slow
SQLite-on-GCS-FUSE store. Mirrors the CMO service's ``db/`` package. ``DATABASE_URL``
comes from the ``samurai-database-url`` secret on the Cloud Run service.
"""

from db.session import dispose_engine, get_session, get_sessionmaker, init_engine

__all__ = ["init_engine", "get_session", "get_sessionmaker", "dispose_engine"]
