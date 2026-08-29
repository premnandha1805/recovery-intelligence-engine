"""
decision_engine.persistence
===========================
PostgreSQL persistence layer for the Recovery Decision Engine.
Uses psycopg v3 directly with raw SQL migrations and no ORM.
"""

from __future__ import annotations

import pathlib
from typing import Any, Optional

from decision_engine.persistence.repository import DecisionRepository
from decision_engine.persistence.in_memory import InMemoryDecisionRepository
from decision_engine.persistence.postgres import PostgresDecisionRepository


def run_migrations(
    database_url: Optional[str] = None,
    migrations_dir: Optional[pathlib.Path | str] = None,
) -> list[str]:
    """Execute forward-only migrations using psycopg v3."""
    from decision_engine.persistence.migrate import run_migrations as _run_migrations

    return _run_migrations(database_url=database_url, migrations_dir=migrations_dir)


__all__ = [
    "DecisionRepository",
    "InMemoryDecisionRepository",
    "PostgresDecisionRepository",
    "run_migrations",
]
