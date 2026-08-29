"""
decision_engine/persistence/test_repository.py
=============================================
Test suite for DecisionRepository abstraction:
- OFFLINE unit tests for InMemoryDecisionRepository (zero database dependency)
- Protocol conformity checks
- POSTGRESQL integration tests for PostgresDecisionRepository (when TEST_DATABASE_URL is set)
"""

from __future__ import annotations

import asyncio
import datetime
import os
import sys
import pytest
import psycopg

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from decision_engine.persistence.in_memory import InMemoryDecisionRepository
from decision_engine.persistence.migrate import run_migrations
from decision_engine.persistence.postgres import PostgresDecisionRepository
from decision_engine.persistence.repository import DecisionRepository

TEST_DB_URL = os.getenv("TEST_DATABASE_URL")
CONTROLLED_PAYMENT_ID = "test_repo_pay_001"
CONTROLLED_PAYMENT_ID_2 = "test_repo_pay_002"


# ---------------------------------------------------------------------------
# Offline Unit Tests (InMemoryDecisionRepository)
# ---------------------------------------------------------------------------

def test_protocol_conformity_in_memory() -> None:
    """Verify InMemoryDecisionRepository implements the DecisionRepository Protocol."""
    repo = InMemoryDecisionRepository()
    assert isinstance(repo, DecisionRepository)


@pytest.mark.asyncio
async def test_in_memory_save_and_get() -> None:
    """Verify basic save and retrieval for InMemoryDecisionRepository."""
    repo = InMemoryDecisionRepository()
    await repo.save_current_decision(
        payment_id="pay_inmem_01",
        decision_id="dec_01",
        final_action="RETRY",
        decision_source="llm",
        llm_proposed_decision="RETRY",
        llm_confidence=0.88,
        expected_incremental_value=120.0,
        evaluated_at=datetime.datetime(2026, 8, 30, 10, 0, tzinfo=datetime.timezone.utc),
    )

    record = await repo.get_current_decision("pay_inmem_01")
    assert record is not None
    assert record["payment_id"] == "pay_inmem_01"
    assert record["final_action"] == "RETRY"
    assert record["llm_confidence"] == 0.88
    assert record["expected_incremental_value"] == 120.0


@pytest.mark.asyncio
async def test_in_memory_upsert() -> None:
    """Verify that save_current_decision overwrites existing decision for the same payment_id."""
    repo = InMemoryDecisionRepository()
    await repo.save_current_decision(
        payment_id="pay_inmem_upsert",
        decision_id="dec_initial",
        final_action="WAIT",
        llm_confidence=0.5,
    )

    initial = await repo.get_current_decision("pay_inmem_upsert")
    assert initial is not None
    assert initial["final_action"] == "WAIT"

    # Overwrite
    await repo.save_current_decision(
        payment_id="pay_inmem_upsert",
        decision_id="dec_updated",
        final_action="ESCALATE",
        llm_confidence=0.95,
    )

    updated = await repo.get_current_decision("pay_inmem_upsert")
    assert updated is not None
    assert updated["decision_id"] == "dec_updated"
    assert updated["final_action"] == "ESCALATE"
    assert updated["llm_confidence"] == 0.95


@pytest.mark.asyncio
async def test_in_memory_missing_lookup() -> None:
    """Verify missing payment returns None for decision and empty list for events."""
    repo = InMemoryDecisionRepository()
    assert await repo.get_current_decision("nonexistent_payment") is None
    assert await repo.get_events("nonexistent_payment") == []


@pytest.mark.asyncio
async def test_in_memory_event_append() -> None:
    """Verify event append never overwrites and preserves event details."""
    repo = InMemoryDecisionRepository()
    await repo.append_decision_event(
        payment_id="pay_ev_01",
        decision_id="ev_01",
        final_action="RETRY",
        evaluated_at=datetime.datetime(2026, 8, 30, 12, 0, tzinfo=datetime.timezone.utc),
    )

    events = await repo.get_events("pay_ev_01")
    assert len(events) == 1
    assert events[0]["decision_id"] == "ev_01"
    assert events[0]["final_action"] == "RETRY"


@pytest.mark.asyncio
async def test_in_memory_multiple_events_chronological() -> None:
    """Verify multiple events for same payment_id are returned in chronological order."""
    repo = InMemoryDecisionRepository()
    t1 = datetime.datetime(2026, 8, 30, 10, 0, tzinfo=datetime.timezone.utc)
    t2 = datetime.datetime(2026, 8, 30, 11, 0, tzinfo=datetime.timezone.utc)
    t3 = datetime.datetime(2026, 8, 30, 12, 0, tzinfo=datetime.timezone.utc)

    # Append out of order
    await repo.append_decision_event(payment_id="pay_multi", decision_id="ev_3", evaluated_at=t3, final_action="STOP")
    await repo.append_decision_event(payment_id="pay_multi", decision_id="ev_1", evaluated_at=t1, final_action="WAIT")
    await repo.append_decision_event(payment_id="pay_multi", decision_id="ev_2", evaluated_at=t2, final_action="RETRY")

    events = await repo.get_events("pay_multi")
    assert len(events) == 3
    assert [e["decision_id"] for e in events] == ["ev_1", "ev_2", "ev_3"]


@pytest.mark.asyncio
async def test_in_memory_clear() -> None:
    """Verify clear() empties all stored decisions and events."""
    repo = InMemoryDecisionRepository()
    await repo.save_current_decision(payment_id="p1", final_action="WAIT")
    await repo.append_decision_event(payment_id="p1", decision_id="e1", final_action="WAIT")

    repo.clear()
    assert await repo.get_current_decision("p1") is None
    assert await repo.get_events("p1") == []


@pytest.mark.asyncio
async def test_in_memory_validation_requires_payment_id() -> None:
    """Verify methods fail fast if payment_id is missing."""
    repo = InMemoryDecisionRepository()
    with pytest.raises(ValueError, match="payment_id is required"):
        await repo.save_current_decision(final_action="WAIT")

    with pytest.raises(ValueError, match="payment_id is required"):
        await repo.append_decision_event(final_action="WAIT")


# ---------------------------------------------------------------------------
# PostgreSQL Integration Tests (Requires TEST_DATABASE_URL)
# ---------------------------------------------------------------------------

integration_mark = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="TEST_DATABASE_URL is not set; skipping PostgreSQL repository integration tests.",
)


@pytest.fixture
def clean_postgres_repo():
    """Ensure Day 8A migrations are applied and clean test rows before/after test."""
    if not TEST_DB_URL:
        pytest.skip("TEST_DATABASE_URL is not set.")

    # Ensure schema migrations are up to date
    run_migrations(database_url=TEST_DB_URL)

    test_ids = (CONTROLLED_PAYMENT_ID, CONTROLLED_PAYMENT_ID_2)

    def _cleanup() -> None:
        with psycopg.connect(TEST_DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM decision_audit WHERE payment_id = ANY(%s);",
                    (list(test_ids),),
                )
                cur.execute(
                    "DELETE FROM decision_audit_events WHERE payment_id = ANY(%s);",
                    (list(test_ids),),
                )
            conn.commit()

    _cleanup()
    repo = PostgresDecisionRepository(database_url=TEST_DB_URL)
    yield repo
    _cleanup()


@integration_mark
def test_protocol_conformity_postgres() -> None:
    """Verify PostgresDecisionRepository implements DecisionRepository Protocol."""
    repo = PostgresDecisionRepository(database_url=TEST_DB_URL or "postgresql://dummy@localhost:5432/db")
    assert isinstance(repo, DecisionRepository)


@integration_mark
@pytest.mark.asyncio
async def test_postgres_current_decision_insert_and_get(clean_postgres_repo: PostgresDecisionRepository) -> None:
    """Verify insert and retrieval of current decision from PostgreSQL."""
    repo = clean_postgres_repo
    now = datetime.datetime.now(datetime.timezone.utc)

    await repo.save_current_decision(
        payment_id=CONTROLLED_PAYMENT_ID,
        decision_id="dec_pg_01",
        request_id="req_pg_01",
        raw_arm_probabilities={"WAIT": 0.1, "RETRY": 0.9},
        raw_arm_net_values={"WAIT": 0.0, "RETRY": 150.0},
        llm_proposed_decision="RETRY",
        llm_confidence=0.91,
        llm_reasoning="Strong net value",
        llm_risk_level="low",
        expected_incremental_value=150.0,
        guardrail_verdict="passed",
        guardrail_reason="All rules cleared",
        final_action="RETRY",
        decision_source="llm",
        error=None,
        evaluated_at=now,
        state_fingerprint="fp_test_123",
    )

    record = await repo.get_current_decision(CONTROLLED_PAYMENT_ID)
    assert record is not None
    assert record["payment_id"] == CONTROLLED_PAYMENT_ID
    assert record["decision_id"] == "dec_pg_01"
    assert record["request_id"] == "req_pg_01"
    assert record["final_action"] == "RETRY"
    assert record["llm_proposed_decision"] == "RETRY"
    assert record["error"] is None
    assert isinstance(record["evaluated_at"], datetime.datetime)


@integration_mark
@pytest.mark.asyncio
async def test_postgres_current_decision_upsert(clean_postgres_repo: PostgresDecisionRepository) -> None:
    """Verify ON CONFLICT UPSERT updates existing record without creating duplicates."""
    repo = clean_postgres_repo
    t1 = datetime.datetime(2026, 8, 30, 12, 0, tzinfo=datetime.timezone.utc)
    t2 = datetime.datetime(2026, 8, 30, 12, 5, tzinfo=datetime.timezone.utc)

    # First insert
    await repo.save_current_decision(
        payment_id=CONTROLLED_PAYMENT_ID,
        decision_id="dec_first",
        final_action="WAIT",
        decision_source="model",
        llm_confidence=0.6,
        evaluated_at=t1,
    )

    first = await repo.get_current_decision(CONTROLLED_PAYMENT_ID)
    assert first is not None
    assert first["final_action"] == "WAIT"

    # Second UPSERT for same payment_id
    await repo.save_current_decision(
        payment_id=CONTROLLED_PAYMENT_ID,
        decision_id="dec_second",
        final_action="RETRY",
        decision_source="llm",
        llm_confidence=0.95,
        evaluated_at=t2,
    )

    second = await repo.get_current_decision(CONTROLLED_PAYMENT_ID)
    assert second is not None
    assert second["decision_id"] == "dec_second"
    assert second["final_action"] == "RETRY"
    assert abs(second["llm_confidence"] - 0.95) < 1e-6


@integration_mark
@pytest.mark.asyncio
async def test_postgres_events_insert_and_chronological_retrieval(
    clean_postgres_repo: PostgresDecisionRepository,
) -> None:
    """Verify appending multiple events for same payment_id and ordered retrieval."""
    repo = clean_postgres_repo
    t1 = datetime.datetime(2026, 8, 30, 10, 0, tzinfo=datetime.timezone.utc)
    t2 = datetime.datetime(2026, 8, 30, 11, 0, tzinfo=datetime.timezone.utc)

    await repo.append_decision_event(
        payment_id=CONTROLLED_PAYMENT_ID,
        decision_id="ev_pg_2",
        evaluated_at=t2,
        final_action="RETRY",
        llm_proposed_decision="RETRY",
        guardrail_overridden=False,
    )
    await repo.append_decision_event(
        payment_id=CONTROLLED_PAYMENT_ID,
        decision_id="ev_pg_1",
        evaluated_at=t1,
        final_action="WAIT",
        llm_proposed_decision="WAIT",
        guardrail_overridden=False,
    )

    events = await repo.get_events(CONTROLLED_PAYMENT_ID)
    assert len(events) == 2
    assert events[0]["decision_id"] == "ev_pg_1"
    assert events[1]["decision_id"] == "ev_pg_2"


@integration_mark
@pytest.mark.asyncio
async def test_postgres_jsonb_and_types_roundtrip(clean_postgres_repo: PostgresDecisionRepository) -> None:
    """Verify JSONB, DOUBLE PRECISION, and TIMESTAMPTZ round-trip cleanly to Python native types."""
    repo = clean_postgres_repo
    now = datetime.datetime.now(datetime.timezone.utc)
    probas = {"WAIT": 0.05, "RETRY": 0.85, "ESCALATE": 0.10}
    net_vals = {"WAIT": 0.0, "RETRY": 150.25, "ESCALATE": 85.50}

    await repo.save_current_decision(
        payment_id=CONTROLLED_PAYMENT_ID,
        decision_id="dec_roundtrip",
        raw_arm_probabilities=probas,
        raw_arm_net_values=net_vals,
        llm_confidence=0.85,
        expected_incremental_value=150.25,
        final_action="RETRY",
        decision_source="llm",
        evaluated_at=now,
    )

    rec = await repo.get_current_decision(CONTROLLED_PAYMENT_ID)
    assert rec is not None
    # JSONB returned as normal Python dicts
    assert rec["raw_arm_probabilities"] == probas
    assert rec["raw_arm_net_values"] == net_vals
    # DOUBLE PRECISION returned as Python float
    assert isinstance(rec["llm_confidence"], float)
    assert abs(rec["llm_confidence"] - 0.85) < 1e-6
    assert isinstance(rec["expected_incremental_value"], float)
    assert abs(rec["expected_incremental_value"] - 150.25) < 1e-6
    # TIMESTAMPTZ returned as timezone-aware datetime
    assert isinstance(rec["evaluated_at"], datetime.datetime)
    assert rec["evaluated_at"].tzinfo is not None


@integration_mark
@pytest.mark.asyncio
async def test_postgres_schema_exact_field_names(clean_postgres_repo: PostgresDecisionRepository) -> None:
    """
    Verify exact field names match Day 8A schema:
    - decision_audit: llm_proposed_decision, error, evaluated_at
    - decision_audit_events: llm_proposed_decision, evaluated_at
    - Forbidden names do NOT exist
    """
    repo = clean_postgres_repo
    now = datetime.datetime.now(datetime.timezone.utc)

    await repo.save_current_decision(
        payment_id=CONTROLLED_PAYMENT_ID,
        decision_id="dec_schema_test",
        llm_proposed_decision="RETRY",
        error=None,
        evaluated_at=now,
        final_action="RETRY",
        decision_source="llm",
    )
    await repo.append_decision_event(
        payment_id=CONTROLLED_PAYMENT_ID,
        decision_id="ev_schema_test",
        llm_proposed_decision="RETRY",
        evaluated_at=now,
        final_action="RETRY",
    )

    rec = await repo.get_current_decision(CONTROLLED_PAYMENT_ID)
    assert rec is not None
    assert "llm_proposed_decision" in rec
    assert "error" in rec
    assert "evaluated_at" in rec
    assert "llm_decision" not in rec
    assert "error_status" not in rec
    assert "timestamp" not in rec

    events = await repo.get_events(CONTROLLED_PAYMENT_ID)
    assert len(events) == 1
    ev = events[0]
    assert "llm_proposed_decision" in ev
    assert "evaluated_at" in ev
    assert "llm_decision" not in ev


@integration_mark
@pytest.mark.asyncio
async def test_postgres_missing_payment(clean_postgres_repo: PostgresDecisionRepository) -> None:
    """Verify missing payment_id returns None for decision and [] for events in Postgres."""
    repo = clean_postgres_repo
    assert await repo.get_current_decision("pay_missing_pg_999") is None
    assert await repo.get_events("pay_missing_pg_999") == []
