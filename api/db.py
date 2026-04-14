"""
api/db.py — SQLAlchemy session factory for the FastAPI service.

Re-uses the same DATABASE_URL as main.py but creates its own engine
(connection pool is not shared across processes).
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings

_engine = create_engine(
    settings.database_url,
    pool_size=3,
    max_overflow=5,
    pool_pre_ping=True,
)

_SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


def get_session() -> Session:
    """FastAPI dependency — yields a DB session, closes it after the request."""
    with _SessionLocal() as session:
        yield session
