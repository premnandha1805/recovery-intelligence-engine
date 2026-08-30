"""
decision_engine/persistence/test_migrate.py
===========================================
Test suite for PostgreSQL forward-only migration runner and schema definitions.

Integration tests execute against a real PostgreSQL test database configured via
the TEST_DATABASE_URL environment variable.

If TEST_DATABASE_URL is not configured, integration tests are explicitly skipped
to clearly communicate that an integration database was not provided, preventing
false-positive claims of offline database integration.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import pytest
import psycopg

from decision_engine.persistence.migrate import (
    get_migration_files,
    run_migrations,
    sanitize_database_url,
)

from dotenv import load_dotenv

load_dotenv(".env.test")
load_dotenv()

TEST_DB_URL = os.getenv("TEST_DATABASE_URL")
MIGRATIONS_DIR = pathlib.Path(__file__).resolve().parent / "migrations"


# ---------------------------------------------------------------------------
# Unit Tests (Offline / No Database Required)
# ---------------------------------------------------------------------------

def test_migration_file_discovery() -> None:
    """Ensure only numbered .sql files are discovered and sorted in numerical order."""
    files = get_migration_files(MIGRATIONS_DIR)
    filenames = [f.name for f in files]
    assert "001_initial_decision_tables.sql" in filenames
    assert "002_add_indexes.sql" in filenames
    assert filenames == sorted(filenames)
    # Ensure non-SQL or unnumbered files (README.md, test files) are strictly excluded
    for name in filenames:
        assert name.endswith(".sql")
        assert name[0].isdigit()


def test_sanitize_database_url() -> None:
    """Ensure credentials in connection URLs are masked."""
    raw = "postgresql://myuser:secretpassword123@localhost:5432/mydb"
    sanitized = sanitize_database_url(raw)
    assert "secretpassword123" not in sanitized
    assert "myuser:***@localhost:5432/mydb" in sanitized


def test_missing_database_url_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure running migrations with no database URL raises a clear ValueError."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    with pytest.raises(ValueError, match="Database URL not provided"):
        run_migrations(database_url=None)


# ---------------------------------------------------------------------------
# Integration Tests (Requires TEST_DATABASE_URL)
# ---------------------------------------------------------------------------

integration_mark = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="TEST_DATABASE_URL is not set; skipping PostgreSQL persistence integration tests.",
)


@pytest.fixture
def clean_test_db():
    """Fixture ensuring a clean database state before test execution."""
    if not TEST_DB_URL:
        pytest.skip("TEST_DATABASE_URL is not set.")

    # Drop existing tables cascade to guarantee a clean slate
    with psycopg.connect(TEST_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS decision_audit CASCADE;")
            cur.execute("DROP TABLE IF EXISTS decision_audit_events CASCADE;")
            cur.execute("DROP TABLE IF EXISTS schema_migrations CASCADE;")
        conn.commit()

    yield TEST_DB_URL

    # Cleanup afterwards
    try:
        with psycopg.connect(TEST_DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS decision_audit CASCADE;")
                cur.execute("DROP TABLE IF EXISTS decision_audit_events CASCADE;")
                cur.execute("DROP TABLE IF EXISTS schema_migrations CASCADE;")
            conn.commit()
    except Exception:
        pass


@integration_mark
def test_migrations_fresh_database_and_idempotency(clean_test_db: str) -> None:
    """
    Test A & B:
    Run 1 on fresh DB -> all migrations apply, schema_migrations has 2 records.
    Run 2 immediately -> 0 migrations apply, no errors, schema unchanged.
    """
    db_url = clean_test_db

    # First run against fresh empty database
    applied_run1 = run_migrations(database_url=db_url)
    assert applied_run1 == ["001_initial_decision_tables.sql", "002_add_indexes.sql"]

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, applied_at FROM schema_migrations ORDER BY id ASC;")
            records = cur.fetchall()
            assert len(records) == 2
            assert records[0][0] == "001_initial_decision_tables.sql"
            assert records[1][0] == "002_add_indexes.sql"
            assert isinstance(records[0][1], datetime.datetime)

    # Second run immediately after
    applied_run2 = run_migrations(database_url=db_url)
    assert applied_run2 == []

    # Verify migration records remain unchanged
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM schema_migrations ORDER BY id ASC;")
            records_after = cur.fetchall()
            assert len(records_after) == 2


@integration_mark
def test_schema_parity_and_forbidden_column_names(clean_test_db: str) -> None:
    """
    Test C: Verify authoritative schema parity against Day 7 audit.py.
    Specifically check:
      - decision_audit: llm_proposed_decision, error, evaluated_at
      - decision_audit_events: llm_proposed_decision, evaluated_at
    Assert forbidden/drifted names do NOT exist:
      - decision_audit_events.llm_decision
      - decision_audit.timestamp
      - decision_audit.error_status
    """
    db_url = clean_test_db
    run_migrations(database_url=db_url)

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            # Query columns for decision_audit
            cur.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'decision_audit';
                """
            )
            audit_cols = {row[0]: row[1] for row in cur.fetchall()}

            # Query columns for decision_audit_events
            cur.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'decision_audit_events';
                """
            )
            events_cols = {row[0]: row[1] for row in cur.fetchall()}

    # Required Day 7 exact field names in decision_audit
    assert "llm_proposed_decision" in audit_cols
    assert "error" in audit_cols
    assert "evaluated_at" in audit_cols
    assert "raw_arm_probabilities" in audit_cols
    assert "raw_arm_net_values" in audit_cols
    assert "llm_confidence" in audit_cols
    assert "expected_incremental_value" in audit_cols
    assert "payment_id" in audit_cols
    assert "decision_id" in audit_cols
    assert "request_id" in audit_cols
    assert "llm_reasoning" in audit_cols
    assert "llm_risk_level" in audit_cols
    assert "guardrail_verdict" in audit_cols
    assert "guardrail_reason" in audit_cols
    assert "final_action" in audit_cols
    assert "decision_source" in audit_cols
    assert "state_fingerprint" in audit_cols

    # Required Day 7 exact field names in decision_audit_events
    assert "decision_id" in events_cols
    assert "payment_id" in events_cols
    assert "request_id" in events_cols
    assert "evaluated_at" in events_cols
    assert "decision_source" in events_cols
    assert "final_action" in events_cols
    assert "model_decision" in events_cols
    assert "llm_proposed_decision" in events_cols
    assert "guardrail_overridden" in events_cols
    assert "guardrail_reason" in events_cols
    assert "state_fingerprint" in events_cols

    # Explicit assertions that incorrect / outdated field names DO NOT exist
    assert "llm_decision" not in events_cols, "decision_audit_events must NOT have llm_decision"
    assert "timestamp" not in audit_cols, "decision_audit must NOT have timestamp"
    assert "error_status" not in audit_cols, "decision_audit must NOT have error_status"


@integration_mark
def test_postgresql_types_and_indexes(clean_test_db: str) -> None:
    """
    Test D & E: Verify PostgreSQL native types and required indexes.
    - JSONB: raw_arm_probabilities, raw_arm_net_values
    - DOUBLE PRECISION: llm_confidence, expected_incremental_value
    - TIMESTAMPTZ: evaluated_at
    - Indexes on payment_id, request_id, evaluated_at, (payment_id, evaluated_at DESC)
    """
    db_url = clean_test_db
    run_migrations(database_url=db_url)

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            # Query column data types
            cur.execute(
                """
                SELECT column_name, data_type, udt_name
                FROM information_schema.columns
                WHERE table_name = 'decision_audit';
                """
            )
            audit_types = {row[0]: (row[1].lower(), row[2].lower()) for row in cur.fetchall()}

            cur.execute(
                """
                SELECT column_name, data_type, udt_name
                FROM information_schema.columns
                WHERE table_name = 'decision_audit_events';
                """
            )
            events_types = {row[0]: (row[1].lower(), row[2].lower()) for row in cur.fetchall()}

            # Query indexes
            cur.execute(
                """
                SELECT tablename, indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public' AND tablename IN ('decision_audit', 'decision_audit_events');
                """
            )
            indexes = cur.fetchall()

    # Verify native Postgres types
    assert audit_types["raw_arm_probabilities"][1] == "jsonb"
    assert audit_types["raw_arm_net_values"][1] == "jsonb"
    assert "double precision" in audit_types["llm_confidence"][0] or audit_types["llm_confidence"][1] == "float8"
    assert "double precision" in audit_types["expected_incremental_value"][0] or audit_types["expected_incremental_value"][1] == "float8"
    assert "timestamp with time zone" in audit_types["evaluated_at"][0] or audit_types["evaluated_at"][1] == "timestamptz"
    assert "timestamp with time zone" in events_types["evaluated_at"][0] or events_types["evaluated_at"][1] == "timestamptz"

    # Verify indexes
    index_names = {row[1] for row in indexes}
    index_defs = {row[1]: row[2] for row in indexes}

    assert "idx_decision_audit_request_id" in index_names
    assert "idx_decision_audit_evaluated_at" in index_names
    assert "idx_decision_audit_events_payment_id" in index_names
    assert "idx_decision_audit_events_request_id" in index_names
    assert "idx_decision_audit_events_evaluated_at" in index_names
    assert "idx_decision_audit_events_payment_evaluated_desc" in index_names

    # Check composite index definition
    comp_def = index_defs["idx_decision_audit_events_payment_evaluated_desc"].lower()
    assert "payment_id" in comp_def
    assert "evaluated_at desc" in comp_def


@integration_mark
def test_data_roundtrip(clean_test_db: str) -> None:
    """
    Test F: Insert and read representative JSONB, DOUBLE PRECISION, and TIMESTAMPTZ data.
    """
    db_url = clean_test_db
    run_migrations(database_url=db_url)

    now = datetime.datetime.now(datetime.timezone.utc)
    raw_probs = {"WAIT": 0.05, "RETRY": 0.85, "ESCALATE": 0.10}
    raw_net = {"WAIT": 0.0, "RETRY": 150.75, "ESCALATE": 90.20}

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            # Insert into decision_audit
            cur.execute(
                """
                INSERT INTO decision_audit (
                    payment_id, decision_id, request_id,
                    raw_arm_probabilities, raw_arm_net_values,
                    llm_proposed_decision, llm_confidence, llm_reasoning,
                    llm_risk_level, expected_incremental_value,
                    guardrail_verdict, guardrail_reason,
                    final_action, decision_source, error,
                    evaluated_at, state_fingerprint
                ) VALUES (
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s, %s
                );
                """,
                (
                    "pay_test_001",
                    "dec_test_001",
                    "req_test_001",
                    json.dumps(raw_probs),
                    json.dumps(raw_net),
                    "RETRY",
                    0.92,
                    "High recovery net value",
                    "low",
                    150.75,
                    "passed",
                    "all checks passed",
                    "RETRY",
                    "llm",
                    None,
                    now,
                    "fingerprint_abc_123",
                ),
            )

            # Insert into decision_audit_events
            cur.execute(
                """
                INSERT INTO decision_audit_events (
                    decision_id, payment_id, request_id,
                    evaluated_at, decision_source, final_action,
                    model_decision, llm_proposed_decision,
                    guardrail_overridden, guardrail_reason, state_fingerprint
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s
                );
                """,
                (
                    "event_dec_001",
                    "pay_test_001",
                    "req_test_001",
                    now,
                    "llm",
                    "RETRY",
                    "RETRY",
                    "RETRY",
                    False,
                    "",
                    "fingerprint_abc_123",
                ),
            )
        conn.commit()

        # Query back and verify data integrity
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM decision_audit WHERE payment_id = %s;", ("pay_test_001",))
            row = cur.fetchone()
            assert row is not None
            # Verify JSONB parsing
            assert row[3] == raw_probs
            assert row[4] == raw_net
            assert row[5] == "RETRY"  # llm_proposed_decision
            assert abs(row[6] - 0.92) < 1e-6  # llm_confidence
            assert abs(row[9] - 150.75) < 1e-6  # expected_incremental_value
            assert row[14] is None  # error
            assert isinstance(row[15], datetime.datetime)  # evaluated_at

            cur.execute("SELECT * FROM decision_audit_events WHERE decision_id = %s;", ("event_dec_001",))
            ev_row = cur.fetchone()
            assert ev_row is not None
            assert ev_row[7] == "RETRY"  # llm_proposed_decision
            assert ev_row[8] is False  # guardrail_overridden
