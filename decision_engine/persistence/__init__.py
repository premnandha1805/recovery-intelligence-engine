"""
decision_engine.persistence
===========================
PostgreSQL persistence layer for the Recovery Decision Engine.
Uses psycopg v3 directly with raw SQL migrations and no ORM.
"""

from typing import Any, Optional
import pathlib


def run_migrations(
    database_url: Optional[str] = None,
    migrations_dir: Optional[pathlib.Path | str] = None,
) -> list[str]:
    """Execute forward-only migrations using psycopg v3."""
    from decision_engine.persistence.migrate import run_migrations as _run_migrations

    return _run_migrations(database_url=database_url, migrations_dir=migrations_dir)


__all__ = ["run_migrations"]
