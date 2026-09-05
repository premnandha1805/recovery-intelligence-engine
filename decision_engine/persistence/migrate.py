"""
decision_engine/persistence/migrate.py
======================================
Lightweight, forward-only PostgreSQL migration runner for the Recovery Decision Engine.
Direct raw SQL execution tracked via schema_migrations (id, applied_at).
Uses psycopg v3 directly with transactional execution per migration.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
from typing import Optional

from dotenv import load_dotenv
import psycopg

# Regex to match numbered SQL migration files: e.g. 001_initial_decision_tables.sql
MIGRATION_FILE_PATTERN = re.compile(r"^\d+.*\.sql$")

# Default migrations directory relative to this script
MIGRATIONS_DIR = pathlib.Path(__file__).resolve().parent / "migrations"


class MigrationError(RuntimeError):
    """Raised when a database migration fails during execution."""


def sanitize_database_url(url: str) -> str:
    """Mask credentials in database URL to prevent secret leakage in logs/errors."""
    if not url:
        return ""
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)


def get_migration_files(migrations_dir: pathlib.Path | str | None = None) -> list[pathlib.Path]:
    """
    Discover only numbered .sql files from the migrations directory,
    sorted numerically/alphanumerically.
    """
    target_dir = pathlib.Path(migrations_dir) if migrations_dir is not None else MIGRATIONS_DIR
    if not target_dir.exists() or not target_dir.is_dir():
        raise FileNotFoundError(f"Migrations directory not found: {target_dir}")

    files = [
        f for f in target_dir.iterdir()
        if f.is_file() and MIGRATION_FILE_PATTERN.match(f.name)
    ]
    return sorted(files, key=lambda f: f.name)


def run_migrations(
    database_url: Optional[str] = None,
    migrations_dir: Optional[pathlib.Path | str] = None,
) -> list[str]:
    """
    Execute any unapplied forward-only SQL migrations in order, idempotently.

    Parameters
    ----------
    database_url : str, optional
        PostgreSQL connection string. If None, resolves from DATABASE_URL or
        POSTGRES_URL environment variables.
    migrations_dir : Path or str, optional
        Path to migrations folder. Defaults to decision_engine/persistence/migrations.

    Returns
    -------
    list[str]
        List of migration filenames applied in this execution run.
    """
    load_dotenv()
    url = database_url or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")

    if not url:
        raise ValueError(
            "Database URL not provided. Please set DATABASE_URL or pass database_url to run_migrations()."
        )

    safe_url = sanitize_database_url(url)
    migration_files = get_migration_files(migrations_dir)

    try:
        conn = psycopg.connect(url)
    except Exception as exc:
        raise ConnectionError(
            f"Failed to connect to PostgreSQL at {safe_url}: {exc}"
        ) from None

    applied_in_run: list[str] = []

    try:
        with conn:
            # 1. Bootstrap schema_migrations tracking table
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        id VARCHAR(255) PRIMARY KEY,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                cur.execute("SELECT id FROM schema_migrations;")
                applied_ids = {row[0] for row in cur.fetchall()}

            # 2. Iterate through ordered migration files and execute unapplied
            for mig_file in migration_files:
                mig_id = mig_file.name
                if mig_id in applied_ids:
                    continue

                sql_content = mig_file.read_text(encoding="utf-8")

                # Execute individual migration within an isolated atomic transaction
                try:
                    with conn.transaction():
                        with conn.cursor() as cur:
                            cur.execute(sql_content)
                            cur.execute(
                                "INSERT INTO schema_migrations (id, applied_at) VALUES (%s, CURRENT_TIMESTAMP);",
                                (mig_id,),
                            )
                    applied_in_run.append(mig_id)
                except Exception as exc:
                    raise MigrationError(
                        f"Migration '{mig_id}' failed: {exc}. Rolling back."
                    ) from exc

    finally:
        conn.close()

    return applied_in_run


def main() -> None:
    """CLI entrypoint: python -m decision_engine.persistence.migrate"""
    parser = argparse.ArgumentParser(
        description="Run forward-only PostgreSQL migrations for Recovery Decision Engine."
    )
    parser.add_argument(
        "--database-url",
        "-d",
        type=str,
        default=None,
        help="PostgreSQL connection string (defaults to DATABASE_URL / POSTGRES_URL env var)",
    )
    parser.add_argument(
        "--migrations-dir",
        "-m",
        type=str,
        default=None,
        help="Custom path to migrations directory",
    )

    args = parser.parse_args()

    try:
        applied = run_migrations(
            database_url=args.database_url,
            migrations_dir=args.migrations_dir,
        )
        if applied:
            print(f"Successfully applied {len(applied)} migration(s):")
            for m in applied:
                print(f"  - {m}")
        else:
            print("No new migrations to apply. Database schema is up to date.")
    except Exception as exc:
        print(f"Migration error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
