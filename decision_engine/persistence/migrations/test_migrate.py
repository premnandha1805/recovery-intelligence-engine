"""
decision_engine/persistence/migrations/test_migrate.py
======================================================
Re-exports test suite and fixtures from decision_engine.persistence.test_migrate.
"""

from decision_engine.persistence.test_migrate import (  # noqa: F401
    clean_test_db,
    test_migration_file_discovery,
    test_sanitize_database_url,
    test_missing_database_url_raises_error,
    test_migrations_fresh_database_and_idempotency,
    test_schema_parity_and_forbidden_column_names,
    test_postgresql_types_and_indexes,
    test_data_roundtrip,
)
