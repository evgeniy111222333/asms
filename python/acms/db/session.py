"""Database engine and session factory management."""

import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from .models import Base

logger = logging.getLogger(__name__)

_engine = None
_session_factory = None


def get_engine(db_url: str = "postgresql://acms:acms@localhost:5432/acms",
               pool_size: int = 10, max_overflow: int = 20):
    """Get or create the database engine.

    Args:
        db_url: PostgreSQL connection string.
        pool_size: Connection pool size.
        max_overflow: Maximum overflow connections.

    Returns:
        SQLAlchemy Engine instance.
    """
    global _engine, _session_factory
    if _engine is None:
        _engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        Base.metadata.create_all(_engine)
        _session_factory = sessionmaker(bind=_engine)
    return _engine


def get_session(db_url: str = "postgresql://acms:acms@localhost:5432/acms",
                pool_size: int = 10, max_overflow: int = 20):
    """Get a new database session.

    Args:
        db_url: PostgreSQL connection string.
        pool_size: Connection pool size.
        max_overflow: Maximum overflow connections.

    Returns:
        SQLAlchemy Session instance.
    """
    get_engine(db_url=db_url, pool_size=pool_size, max_overflow=max_overflow)
    return _session_factory()


def init_db(db_url: str = "postgresql://acms:acms@localhost:5432/acms",
            pool_size: int = 10, max_overflow: int = 20) -> sessionmaker:
    """Initialize database and create all tables.

    Args:
        db_url: PostgreSQL connection string.
        pool_size: Connection pool size.
        max_overflow: Maximum overflow connections.

    Returns:
        SessionMaker factory.
    """
    engine = create_engine(
        db_url,
        poolclass=QueuePool,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        pool_recycle=3600,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def close_db():
    """Dispose of the database engine and reset globals."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
        _engine = None
        _session_factory = None


__all__ = [
    "_engine",
    "_session_factory",
    "get_engine",
    "get_session",
    "init_db",
    "close_db",
]
