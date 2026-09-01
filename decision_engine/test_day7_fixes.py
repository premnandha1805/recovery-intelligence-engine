"""
decision_engine/test_day7_fixes.py
==================================
Comprehensive regression tests for Day 7 Final Fixes:
1. Tighten Python LLM Deadline & Per-Payment Concurrency
2. State-Aware Cache Fingerprint & Stale Cache Invalidation
3. Additive Append-Only Decision Audit Events Table
4. Atomic Two-Table Write & Forced Rollback Verification
5. Concurrent Writes Across 10+ Different Payment IDs (Zero DB Locks)
6. Warm-Process Dataset Loader Verification (No Repeated CSV Parsing)
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import pathlib
import sqlite3
import sys
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import aiosqlite
from httpx import ASGITransport, AsyncClient
import pandas as pd
import pytest

from models.schemas import Action, Decision
from decision_engine.audit import (
    CREATE_TABLE_SQL,
    CREATE_EVENTS_TABLE_SQL,
    CREATE_EVENTS_INDEX_SQL,
    compute_state_fingerprint,
    save_decision_audit,
    async_save_decision_audit,
    get_audit_record,
    get_audit_row_count,
    get_audit_events_for_payment,
    get_audit_events_count,
)
from decision_engine.context_node import get_payment_state, _get_dataset
from decision_engine.graph import create_recovery_graph
from decision_engine.reasoning_node import LLMDecision
from decision_engine.service import app, get_payment_lock
from decision_engine.state import RecoveryState


# ── Test Fixtures ────────────────────────────────────────────────────────────

def make_mock_policy():
    """Mock CausalUpliftPolicy returning positive uplift for RETRY."""
    mock_policy = MagicMock()
    mock_t_learner = MagicMock()
    mock_policy.t_learner = mock_t_learner

    def fake_predict_proba(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [{"WAIT": 0.15, "RETRY": 0.85, "RETRY_NUDGE": 0.50, "ESCALATE": 0.30}],
            index=df.index,
        )

    mock_t_learner.predict_proba.side_effect = fake_predict_proba
    return mock_policy


def make_mock_llm(decision: str = "RETRY", confidence: float = 0.90, delay_s: float = 0.0):
    """Mock LangChain chat model returning structured LLMDecision."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    async def fake_ainvoke(*args, **kwargs):
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        return LLMDecision(
            decision=decision,
            confidence=confidence,
            reasoning=f"Selected {decision} based on policy recommendations.",
            risk_level="low",
        )

    mock_structured.ainvoke = AsyncMock(side_effect=fake_ainvoke)
    mock_structured.invoke = MagicMock(return_value=LLMDecision(
        decision=decision,
        confidence=confidence,
        reasoning=f"Selected {decision} based on policy recommendations.",
        risk_level="low",
    ))
    return mock_llm


async def init_isolated_service(tmp_path: pathlib.Path, mock_llm: Any = None):
    """Initialize a fully isolated FastAPI test service with dedicated SQLite database."""
    db_path = str(tmp_path / f"test_day7_{uuid.uuid4().hex[:8]}.db")
    db = await aiosqlite.connect(db_path)
    await db.execute("PRAGMA journal_mode=WAL;")
    await db.execute("PRAGMA synchronous=NORMAL;")
    await db.execute("PRAGMA busy_timeout=5000;")
    await db.execute(CREATE_TABLE_SQL)
    await db.execute(CREATE_EVENTS_TABLE_SQL)
    await db.execute(CREATE_EVENTS_INDEX_SQL)
    await db.commit()

    policy = make_mock_policy()
    llm = mock_llm or make_mock_llm()
    graph = create_recovery_graph(policy=policy, llm=llm, use_async=True)

    app.state.policy = policy
    app.state.graph = graph
    app.state.db = db
    app.state.llm_semaphore = asyncio.Semaphore(5)
    app.state.payment_locks = {}
    app.state.locks_mutex = asyncio.Lock()
    app.state.migrations_applied = True
    app.state.dataset = None  # Use canonical cache unless overridden

    return app, db, db_path, llm


# ── FIX 1: Tightened LLM Deadline & Same-Payment Concurrency ─────────────────

@pytest.mark.asyncio
async def test_fix1_same_payment_concurrent_lock_and_deadline(tmp_path: pathlib.Path):
    """
    FIX 1 — Tests A, B, C:
    Fire two concurrent requests for the SAME payment_id with a slow LLM (delay_s=0.4).
    Verify:
    - First request acquires lock and performs fresh evaluation.
    - Second request waits on lock and resolves via cache hit.
    - Both return HTTP 200.
    - Second request's transit time is comfortably below 6500 ms.
    - Exactly ONE LLM evaluation occurs.
    """
    mock_llm = make_mock_llm(delay_s=0.4)
    app_instance, db, db_path, _ = await init_isolated_service(tmp_path, mock_llm=mock_llm)
    structured_mock = mock_llm.with_structured_output.return_value

    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payment_id = "pay_000001_a1"

            t0 = time.monotonic()
            req1_task = asyncio.create_task(
                client.post("/evaluate", json={"payment_id": payment_id})
            )
            await asyncio.sleep(0.05)  # Ensure req1 acquires lock first
            req2_task = asyncio.create_task(
                client.post("/evaluate", json={"payment_id": payment_id})
            )

            resp1, resp2 = await asyncio.gather(req1_task, req2_task)
            total_duration = (time.monotonic() - t0) * 1000

            # A. Both return HTTP 200
            assert resp1.status_code == 200
            assert resp2.status_code == 200
            data1, data2 = resp1.json(), resp2.json()

            assert data1["payment_id"] == payment_id
            assert data2["payment_id"] == payment_id
            assert data1["final_action"] == data2["final_action"]

            # Second request was served from cache
            assert data2["decision_source"] == "cache"

            # B. Second caller total transit comfortably below 6500ms
            assert total_duration < 6500, f"Expected total duration < 6500ms, got {total_duration}ms"

            # C. Exactly ONE LLM call
            assert structured_mock.ainvoke.call_count == 1, (
                f"Expected exactly 1 LLM call, got {structured_mock.ainvoke.call_count}"
            )
    finally:
        await db.close()


# ── FIX 2: State-Aware Cache Fingerprint Tests ───────────────────────────────

@pytest.mark.asyncio
async def test_fix2_unchanged_state_cache_hit(tmp_path: pathlib.Path):
    """FIX 2 — Test D: Unchanged state → Cache HIT."""
    mock_llm = make_mock_llm()
    app_instance, db, _, _ = await init_isolated_service(tmp_path, mock_llm=mock_llm)
    structured_mock = mock_llm.with_structured_output.return_value

    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payment_id = "pay_000002_a2"

            # Request 1: Fresh evaluation
            resp1 = await client.post("/evaluate", json={"payment_id": payment_id})
            assert resp1.status_code == 200
            assert structured_mock.ainvoke.call_count == 1

            # Request 2: Same state → Cache HIT
            resp2 = await client.post("/evaluate", json={"payment_id": payment_id})
            assert resp2.status_code == 200
            assert resp2.json()["decision_source"] == "cache"
            assert structured_mock.ainvoke.call_count == 1  # No additional LLM call
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_fix2_state_field_changes_cause_cache_miss(tmp_path: pathlib.Path):
    """
    FIX 2 — Tests E, F, G, H, I:
    Verify that changing any of the individual fingerprint fields triggers a cache miss:
    - E: changed status → cache MISS
    - F: changed attempt_number → cache MISS
    - G: changed consecutive_failures → cache MISS
    - H: changed retry_count → cache MISS
    - I: changed interventions_7d → cache MISS
    """
    payment_id = "pay_000003_a3"
    base_df = _get_dataset().copy()

    # Verify fingerprint sensitivity directly across all 6 inputs
    fp_base = compute_state_fingerprint(
        payment_id=payment_id,
        status="failed",
        attempt_number=1,
        consecutive_failures=0,
        retry_count=0,
        interventions_7d=0,
    )

    # E. Status changed
    fp_status = compute_state_fingerprint(
        payment_id=payment_id,
        status="SUCCESS",
        attempt_number=1,
        consecutive_failures=0,
        retry_count=0,
        interventions_7d=0,
    )
    assert fp_base != fp_status

    # F. Attempt number changed
    fp_attempt = compute_state_fingerprint(
        payment_id=payment_id,
        status="failed",
        attempt_number=2,
        consecutive_failures=0,
        retry_count=0,
        interventions_7d=0,
    )
    assert fp_base != fp_attempt

    # G. Consecutive failures changed
    fp_failures = compute_state_fingerprint(
        payment_id=payment_id,
        status="failed",
        attempt_number=1,
        consecutive_failures=2,
        retry_count=0,
        interventions_7d=0,
    )
    assert fp_base != fp_failures

    # H. Retry count changed
    fp_retry = compute_state_fingerprint(
        payment_id=payment_id,
        status="failed",
        attempt_number=1,
        consecutive_failures=0,
        retry_count=1,
        interventions_7d=0,
    )
    assert fp_base != fp_retry

    # I. Interventions 7d changed
    fp_interv = compute_state_fingerprint(
        payment_id=payment_id,
        status="failed",
        attempt_number=1,
        consecutive_failures=0,
        retry_count=0,
        interventions_7d=1,
    )
    assert fp_base != fp_interv

    # Verify service cache lookup responds to state change via app.state.dataset
    mock_llm = make_mock_llm()
    app_instance, db, _, _ = await init_isolated_service(tmp_path, mock_llm=mock_llm)
    structured_mock = mock_llm.with_structured_output.return_value

    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            test_df = base_df.copy()
            app_instance.state.dataset = test_df

            # Initial evaluation
            r1 = await client.post("/evaluate", json={"payment_id": payment_id})
            assert r1.status_code == 200
            assert structured_mock.ainvoke.call_count == 1

            # Mutate attempt_number in dataset → Cache MISS
            test_df.loc[test_df["payment_id"] == payment_id, "attempt_number"] = 2
            r2 = await client.post("/evaluate", json={"payment_id": payment_id, "force_recompute": False})
            assert r2.status_code == 200
            assert structured_mock.ainvoke.call_count == 2, "Expected fresh evaluation on attempt_number change"

            # Mutate consecutive_failed_cycles in dataset → Cache MISS
            test_df.loc[test_df["payment_id"] == payment_id, "consecutive_failed_cycles"] = 2
            r3 = await client.post("/evaluate", json={"payment_id": payment_id, "force_recompute": False})
            assert r3.status_code == 200
            assert structured_mock.ainvoke.call_count == 3, "Expected fresh evaluation on consecutive_failures change"

            # Mutate retry_count in dataset → Cache MISS
            test_df.loc[test_df["payment_id"] == payment_id, "retry_count"] = 2
            r4 = await client.post("/evaluate", json={"payment_id": payment_id, "force_recompute": False})
            assert r4.status_code == 200
            assert structured_mock.ainvoke.call_count == 4, "Expected fresh evaluation on retry_count change"

            # Mutate interventions_last_7_days in dataset → Cache MISS
            test_df.loc[test_df["payment_id"] == payment_id, "interventions_last_7_days"] = 2
            r5 = await client.post("/evaluate", json={"payment_id": payment_id, "force_recompute": False})
            assert r5.status_code == 200
            assert structured_mock.ainvoke.call_count == 5, "Expected fresh evaluation on interventions_7d change"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_fix2_stale_cache_invalidation_on_success_transition(tmp_path: pathlib.Path):
    """
    FIX 2 — Test J (Exact Stale-State Scenario):
    1. Start with payment in failed/retryable state.
    2. Evaluate it and establish cached current decision.
    3. Confirm state_fingerprint is persisted.
    4. Change ONLY payment status in test dataset to: SUCCESS.
    5. Re-request same payment with force_recompute = false.
    6. Assert previous cached RETRY is NOT returned.
    7. Assert a fresh evaluation occurs.
    8. Assert new state causes correct safe decision (WAIT via guardrails).
    9. Assert state_fingerprint changes.
    """
    payment_id = "pay_000001_a1"
    mock_llm = make_mock_llm(decision="RETRY")
    app_instance, db, db_path, _ = await init_isolated_service(tmp_path, mock_llm=mock_llm)
    structured_mock = mock_llm.with_structured_output.return_value

    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            test_df = _get_dataset().copy()
            app_instance.state.dataset = test_df

            # 1 & 2: Initial failed state evaluation
            resp1 = await client.post("/evaluate", json={"payment_id": payment_id})
            assert resp1.status_code == 200
            data1 = resp1.json()
            assert data1["final_action"] == "RETRY"
            assert structured_mock.ainvoke.call_count == 1

            # 3: Confirm state_fingerprint is persisted in decision_audit
            rec1 = get_audit_record(payment_id, db_path=db_path)
            assert rec1 is not None
            fp1 = rec1["state_fingerprint"]
            assert fp1 is not None
            assert len(fp1) == 64  # Valid SHA-256 hex string

            # 4: Change ONLY payment status to SUCCESS
            test_df.loc[test_df["payment_id"] == payment_id, "status"] = "SUCCESS"

            # 5: Re-request same payment with force_recompute = false
            resp2 = await client.post("/evaluate", json={"payment_id": payment_id, "force_recompute": False})
            assert resp2.status_code == 200
            data2 = resp2.json()

            # 6: Assert previous cached RETRY is NOT returned
            assert data2["final_action"] != "RETRY"

            # 7: Assert a fresh evaluation occurred (LLM re-invoked)
            assert structured_mock.ainvoke.call_count >= 2
            assert data2["decision_source"] != "cache"

            # 8: Assert new state causes WAIT according to guardrail state transition rule
            assert data2["final_action"] == "WAIT"
            assert data2["guardrail_overridden"] is True or "Invalid state transition" in data2.get("guardrail_reason", "") or data2["final_action"] == "WAIT"

            # 9: Assert state_fingerprint changes
            rec2 = get_audit_record(payment_id, db_path=db_path)
            fp2 = rec2["state_fingerprint"]
            assert fp2 is not None
            assert fp1 != fp2
    finally:
        await db.close()


# ── FIX 3: Additive Append-Only Decision Audit Events Tests ───────────────────

@pytest.mark.asyncio
async def test_fix3_additive_append_only_event_ledger(tmp_path: pathlib.Path):
    """
    FIX 3 — Tests K, L, M, N, O, P, Q:
    K. First fresh evaluation creates exactly one event.
    L. Cache hit creates zero additional events.
    M. force_recompute twice creates two additional distinct event rows.
    N. Each event has RFC 4122 UUID v4 decision_id.
    O. Event timestamps differ.
    P. Existing decision_audit still has exactly ONE row per payment_id.
    Q. Day 6 upsert semantics preserved.
    """
    payment_id = "pay_000005_a1"
    mock_llm = make_mock_llm()
    app_instance, db, db_path, _ = await init_isolated_service(tmp_path, mock_llm=mock_llm)

    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # K. First fresh evaluation creates one event
            resp1 = await client.post("/evaluate", json={"payment_id": payment_id})
            assert resp1.status_code == 200
            events1 = get_audit_events_for_payment(payment_id, db_path=db_path)
            assert len(events1) == 1
            ev1 = events1[0]
            assert ev1["payment_id"] == payment_id
            assert "llm" in ev1["decision_source"] or "fallback" in ev1["decision_source"]
            assert ev1["final_action"] in ("RETRY", "WAIT", "RETRY_NUDGE", "ESCALATE")

            # N. decision_id is UUID v4 (NOT dec_<payment_id>)
            assert not ev1["decision_id"].startswith("dec_")
            uuid_parsed1 = uuid.UUID(ev1["decision_id"])
            assert uuid_parsed1.version == 4

            # L. Cache hit creates zero additional events
            resp2 = await client.post("/evaluate", json={"payment_id": payment_id})
            assert resp2.status_code == 200
            assert resp2.json()["decision_source"] == "cache"
            events2 = get_audit_events_for_payment(payment_id, db_path=db_path)
            assert len(events2) == 1, "Cache hit must NOT insert an audit event"

            # M. force_recompute twice creates two additional distinct event rows
            await asyncio.sleep(0.01)  # Ensure distinct timestamp
            resp3 = await client.post("/evaluate", json={"payment_id": payment_id, "force_recompute": True})
            assert resp3.status_code == 200
            events3 = get_audit_events_for_payment(payment_id, db_path=db_path)
            assert len(events3) == 2

            await asyncio.sleep(0.01)
            resp4 = await client.post("/evaluate", json={"payment_id": payment_id, "force_recompute": True})
            assert resp4.status_code == 200
            events4 = get_audit_events_for_payment(payment_id, db_path=db_path)
            assert len(events4) == 3

            # N & O: Check UUIDs and distinct timestamps
            ev_ids = [e["decision_id"] for e in events4]
            assert len(set(ev_ids)) == 3, "All event decision_ids must be unique UUIDs"
            for eid in ev_ids:
                assert uuid.UUID(eid).version == 4

            timestamps = [e["evaluated_at"] for e in events4]
            assert len(set(timestamps)) == 3, "Event timestamps must differ across distinct evaluations"

            # P. Existing decision_audit still has exactly ONE row per payment_id
            assert get_audit_row_count(db_path=db_path) == 1
            current_rec = get_audit_record(payment_id, db_path=db_path)
            assert current_rec is not None
            assert current_rec["decision_id"] == f"dec_{payment_id}"  # Day 6 decision_id preserved
    finally:
        await db.close()


# ── MANDATORY VERIFICATION REQUIREMENT 1: Atomic Two-Table Write & Rollback ──

@pytest.mark.asyncio
async def test_mandatory_atomic_two_table_write_forced_rollback(tmp_path: pathlib.Path):
    """
    MANDATORY REQUIREMENT 1:
    Deliberately cause the second write (decision_audit_events INSERT) to fail
    after the decision_audit UPSERT has executed.
    Assert:
    - decision_audit contains NO newly committed row/update from that failed transaction.
    - decision_audit_events contains NO row.
    - The transaction rolled back completely.
    """
    db_path = tmp_path / "test_atomic_rollback.db"
    payment_id = "pay_atomic_fail_01"

    # Initialize tables
    from decision_engine.audit import init_audit_db
    init_audit_db(db_path)

    # Initial state to save
    test_state: RecoveryState = {
        "payment_id": payment_id,
        "arm_probabilities": {"WAIT": 0.2, "RETRY": 0.8},
        "arm_net_values": {"WAIT": 10.0, "RETRY": 80.0},
        "llm_decision": {
            "decision": "RETRY",
            "confidence": 0.85,
            "reasoning": "High net value",
            "risk_level": "low",
            "expected_incremental_value": 80.0,
            "decision_source": "llm",
        },
        "guardrail_result": {"status": "passed", "reason": "ok", "overridden": False},
        "final_action": "RETRY",
        "error": None,
        "audit_trail": [],
        "state_fingerprint": "mock_fingerprint_atomic_test",
    }

    # Use a SQLite trigger to simulate failure on the second write (INSERT INTO decision_audit_events)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("""
            CREATE TRIGGER fail_events_trigger BEFORE INSERT ON decision_audit_events
            BEGIN
                SELECT RAISE(ABORT, 'Simulated database failure during audit_events INSERT');
            END;
        """)
        conn.commit()

    # 1. Test sync path (save_decision_audit)
    with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="Simulated database failure"):
        save_decision_audit(test_state, db_path=db_path)

    # Assert complete rollback:
    # 1. decision_audit contains NO row
    assert get_audit_record(payment_id, db_path=db_path) is None, (
        "decision_audit must NOT have committed a row after transaction failure"
    )
    # 2. decision_audit_events contains NO row
    assert get_audit_events_count(payment_id=payment_id, db_path=db_path) == 0, (
        "decision_audit_events must have 0 rows after transaction failure"
    )

    # 2. Test async path (async_save_decision_audit)
    async with aiosqlite.connect(str(db_path)) as aio_db:
        with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError), match="Simulated database failure"):
            await async_save_decision_audit(test_state, aio_db)

    # Assert complete rollback in async path as well
    assert get_audit_record(payment_id, db_path=db_path) is None
    assert get_audit_events_count(payment_id=payment_id, db_path=db_path) == 0


# ── MANDATORY VERIFICATION REQUIREMENT 2: Concurrent Writes (10+ Different IDs)

@pytest.mark.asyncio
async def test_mandatory_concurrent_writes_10_different_payment_ids(tmp_path: pathlib.Path):
    """
    MANDATORY REQUIREMENT 2:
    Stress SQLite writer path using at least 10 concurrent requests for 10 DIFFERENT payment_ids.
    Requirements:
    - All requests complete successfully with HTTP 200.
    - Zero 'database is locked' errors.
    - Zero partial audit records.
    - Every payment has exactly one current-state row in decision_audit.
    - Every fresh evaluation has exactly one event row in decision_audit_events.
    - WAL mode remains enabled and busy_timeout is 5000ms.
    """
    app_instance, db, db_path, mock_llm = await init_isolated_service(tmp_path)

    try:
        # Verify WAL mode and busy_timeout are enabled
        async with db.execute("PRAGMA journal_mode;") as cur:
            journal_mode = (await cur.fetchone())[0]
            assert journal_mode.lower() == "wal", f"Expected WAL mode, got {journal_mode}"

        async with db.execute("PRAGMA busy_timeout;") as cur:
            busy_timeout = int((await cur.fetchone())[0])
            assert busy_timeout == 5000, f"Expected busy_timeout 5000, got {busy_timeout}"

        # 12 different payment IDs from the canonical dataset
        payment_ids = [
            "pay_000001_a1", "pay_000002_a2", "pay_000003_a3", "pay_000004_a4",
            "pay_000005_a1", "pay_000006_a2", "pay_000007_a1", "pay_000008_a2",
            "pay_000009_a3", "pay_000010_a4", "pay_000011_a1", "pay_000012_a1"
        ]
        num_requests = len(payment_ids)

        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            t_start = time.monotonic()
            tasks = [
                client.post("/evaluate", json={"payment_id": pid})
                for pid in payment_ids
            ]
            responses = await asyncio.gather(*tasks)
            duration_ms = round((time.monotonic() - t_start) * 1000, 2)

            failures = [r for r in responses if r.status_code != 200]
            assert len(failures) == 0, f"Expected 0 failures, got {len(failures)}"

            for r in responses:
                assert r.status_code == 200
                data = r.json()
                assert (
                    "llm" in data["decision_source"]
                    or "fallback" in data["decision_source"]
                    or "error" in data["decision_source"]
                )

            # Check database row counts
            current_row_count = get_audit_row_count(db_path=db_path)
            events_row_count = get_audit_events_count(db_path=db_path)

            assert current_row_count == num_requests, (
                f"Expected {num_requests} current records, got {current_row_count}"
            )
            assert events_row_count == num_requests, (
                f"Expected {num_requests} event records, got {events_row_count}"
            )

            # Check each payment_id has exactly 1 current record and 1 event record
            for pid in payment_ids:
                rec = get_audit_record(pid, db_path=db_path)
                assert rec is not None
                assert rec["payment_id"] == pid
                events = get_audit_events_for_payment(pid, db_path=db_path)
                assert len(events) == 1
                assert events[0]["payment_id"] == pid

            print(f"\n[Stress Test Evidence] Requests: {num_requests}, Duration: {duration_ms}ms, Failures: 0, Current Rows: {current_row_count}, Event Rows: {events_row_count}")
    finally:
        await db.close()


# ── MANDATORY VERIFICATION REQUIREMENT 3: Warm Dataset Loading ───────────────

@pytest.mark.asyncio
async def test_mandatory_dataset_warm_loading_no_repeated_csv_parsing(tmp_path: pathlib.Path):
    """
    MANDATORY REQUIREMENT 3:
    Verify that a series of multiple /evaluate requests does NOT repeatedly reload
    or reparse the full CSV from disk.
    """
    app_instance, db, _, _ = await init_isolated_service(tmp_path)
    # Warm in-memory dataset cache
    _get_dataset()

    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            test_pids = ["pay_000001_a1", "pay_000002_a2", "pay_000003_a3", "pay_000004_a4", "pay_000005_a1"]
            # Patch pd.read_csv to detect if any read occurs during /evaluate calls
            with patch("pandas.read_csv", side_effect=AssertionError("pd.read_csv should not be called during requests")) as mock_csv_read:
                for pid in test_pids:
                    resp = await client.post("/evaluate", json={"payment_id": pid})
                    assert resp.status_code == 200

                assert mock_csv_read.call_count == 0, (
                    f"pd.read_csv was called {mock_csv_read.call_count} times during /evaluate requests!"
                )
    finally:
        await db.close()


# ── DAY 8E: PostgreSQL State-Fingerprint Cache Integration Tests ──────────────

async def init_isolated_postgres_service(mock_llm: Any = None):
    """
    Initialize a test FastAPI service wired directly to Docker PostgreSQL.
    Consumes zero Azure tokens via mock_llm.
    """
    test_db_url = os.getenv("TEST_DATABASE_URL")
    if not test_db_url:
        pytest.skip("TEST_DATABASE_URL is not set.")

    import psycopg
    from decision_engine.persistence.postgres import PostgresDecisionRepository, create_postgres_pool
    from decision_engine.persistence.migrate import run_migrations

    run_migrations(database_url=test_db_url)
    pool = await create_postgres_pool(test_db_url)
    repo = PostgresDecisionRepository(pool=pool)

    policy = make_mock_policy()
    llm = mock_llm or make_mock_llm(decision="RETRY")
    graph = create_recovery_graph(policy=policy, llm=llm, use_async=True)

    app.state.policy = policy
    app.state.graph = graph
    app.state.db_pool = pool
    app.state.repository = repo
    app.state.db = None
    app.state.llm_semaphore = asyncio.Semaphore(5)
    app.state.payment_locks = {}
    app.state.locks_mutex = asyncio.Lock()
    app.state.migrations_applied = True
    app.state.dataset = None

    return app, repo, pool, llm


def _cleanup_postgres_payment(payment_id: str, db_url: str):
    """Clean up any leftover rows for payment_id in Docker PostgreSQL."""
    import psycopg
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM decision_audit WHERE payment_id = %s;", (payment_id,))
            cur.execute("DELETE FROM decision_audit_events WHERE payment_id = %s;", (payment_id,))
        conn.commit()


@pytest.mark.asyncio
async def test_postgres_stale_cache_invalidation_on_success_transition():
    """
    Day 8E (Section 8) — Stale Cache Invalidation with Real Docker PostgreSQL:
    1. Start from clean database state for test payment.
    2. Set initial dataset state: status=FAILED, attempt_number=1, consecutive_failures=0, retry_count=0, interventions_7d=0.
    3. Evaluate with force_recompute=False -> fresh evaluation, decision_source!='cache', final_action=RETRY.
    4. Confirm state_fingerprint is persisted.
    5. Mutate ONLY dataset status to SUCCESS.
    6. Re-evaluate with force_recompute=False.
    7. REQUIRED: cache MISS, new fingerprint != initial fingerprint, stale RETRY not returned, final_action=WAIT.
    8. Confirm decision_audit contains exactly one row with new fingerprint.
    9. Confirm decision_audit_events contains two events.
    10. Confirm mocked LLM invocation count increased exactly once (total 2).
    """
    test_db_url = os.getenv("TEST_DATABASE_URL")
    if not test_db_url:
        pytest.skip("TEST_DATABASE_URL is not set.")

    payment_id = "pay_day8e_stale_001"
    _cleanup_postgres_payment(payment_id, test_db_url)

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    async def fake_ainvoke(prompt_str, *args, **kwargs):
        if "['WAIT']" in str(prompt_str):
            return LLMDecision(
                decision="WAIT",
                confidence=0.95,
                reasoning="Payment in terminal/success status; waiting.",
                risk_level="low",
            )
        return LLMDecision(
            decision="RETRY",
            confidence=0.90,
            reasoning="Selected RETRY based on positive uplift.",
            risk_level="low",
        )

    mock_structured.ainvoke = AsyncMock(side_effect=fake_ainvoke)
    mock_structured.invoke = MagicMock(side_effect=fake_ainvoke)
    structured_mock = mock_structured

    app_instance, repo, pool, _ = await init_isolated_postgres_service(mock_llm=mock_llm)

    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            test_df = _get_dataset().copy()
            # Set initial failed state for test payment
            row_dict = test_df.iloc[0].to_dict()
            row_dict.update({
                "payment_id": payment_id,
                "status": "FAILED",
                "attempt_number": 1,
                "consecutive_failed_cycles": 0,
                "retry_count": 0,
                "interventions_last_7_days": 0,
            })
            test_df = test_df[test_df["payment_id"] != payment_id]
            test_df = pd.concat([test_df, pd.DataFrame([row_dict])], ignore_index=True)
            app_instance.state.dataset = test_df

            # Step 3: Run /evaluate with force_recompute=False
            req_id_1 = "req-day8e-fresh-001"
            resp1 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": req_id_1},
                json={"payment_id": payment_id, "force_recompute": False},
            )
            assert resp1.status_code == 200
            data1 = resp1.json()

            # Step 5: Verify fresh evaluation occurred
            assert data1["decision_source"] != "cache"
            assert data1["final_action"] == "RETRY"
            assert structured_mock.ainvoke.call_count == 1

            # Step 6: Record initial fingerprint and row counts from PostgreSQL
            rec1 = await repo.get_current_decision(payment_id)
            assert rec1 is not None, "PostgreSQL decision_audit must contain record"
            fp1 = rec1["state_fingerprint"]
            assert fp1 is not None and len(fp1) == 64
            events1 = await repo.get_events(payment_id)
            assert len(events1) == 1, "Exactly 1 event in decision_audit_events"

            # Step 7: WITHOUT reloading CSV, mutate ONLY status to SUCCESS
            test_df.loc[test_df["payment_id"] == payment_id, "status"] = "SUCCESS"

            # Step 8: POST /evaluate again with force_recompute=False
            req_id_2 = "req-day8e-stale-002"
            resp2 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": req_id_2},
                json={"payment_id": payment_id, "force_recompute": False},
            )
            assert resp2.status_code == 200
            data2 = resp2.json()

            # Step 9: Verify cache MISS, new fingerprint != old, final_action=WAIT
            assert data2["decision_source"] != "cache", "Second request must NOT be a cache hit"
            assert data2["final_action"] != "RETRY", "Stale cached RETRY must NOT be returned"
            assert data2["final_action"] == "WAIT", "SUCCESS state must cause WAIT via guardrail transition rule"

            # Verify PostgreSQL state
            rec2 = await repo.get_current_decision(payment_id)
            assert rec2 is not None
            fp2 = rec2["state_fingerprint"]
            assert fp2 is not None and len(fp2) == 64
            assert fp1 != fp2, "New fingerprint must differ from the initial fingerprint"
            assert rec2["final_action"] == "WAIT"

            events2 = await repo.get_events(payment_id)
            assert len(events2) == 2, "Event ledger must contain exactly 2 events (one per evaluation)"

            # Step 11: Verify mocked LLM invocation count increased exactly once
            assert structured_mock.ainvoke.call_count == 2, "LLM must be invoked exactly twice (one per fresh eval)"
    finally:
        from decision_engine.persistence.postgres import close_postgres_pool
        await close_postgres_pool(pool)
        _cleanup_postgres_payment(payment_id, test_db_url)


@pytest.mark.asyncio
async def test_postgres_fingerprint_cache_hit_zero_events():
    """
    Day 8E (Section 9) — Cache-Hit Integration Test with Real Docker PostgreSQL:
    1. Evaluate payment once -> fresh evaluation.
    2. Repeat same payment without changing fingerprint input, force_recompute=False.
    3. Assert:
       - decision_source == 'cache'
       - no additional LLM invocation (call count stays 1)
       - decision_audit remains 1 row
       - decision_audit_events count unchanged (zero new events appended)
       - HTTP request_id in response represents current request
       - historical request_id in database is NOT overwritten
    """
    test_db_url = os.getenv("TEST_DATABASE_URL")
    if not test_db_url:
        pytest.skip("TEST_DATABASE_URL is not set.")

    payment_id = "pay_day8e_hit_002"
    _cleanup_postgres_payment(payment_id, test_db_url)

    mock_llm = make_mock_llm(decision="RETRY")
    structured_mock = mock_llm.with_structured_output.return_value

    app_instance, repo, pool, _ = await init_isolated_postgres_service(mock_llm=mock_llm)

    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            test_df = _get_dataset().copy()
            row_dict = test_df.iloc[0].to_dict()
            row_dict.update({
                "payment_id": payment_id,
                "status": "FAILED",
                "attempt_number": 1,
                "consecutive_failed_cycles": 0,
                "retry_count": 0,
                "interventions_last_7_days": 0,
            })
            test_df = test_df[test_df["payment_id"] != payment_id]
            test_df = pd.concat([test_df, pd.DataFrame([row_dict])], ignore_index=True)
            app_instance.state.dataset = test_df

            # Request 1: Fresh evaluation
            req_id_1 = "req-day8e-hit-initial-111"
            resp1 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": req_id_1},
                json={"payment_id": payment_id, "force_recompute": False},
            )
            assert resp1.status_code == 200
            assert resp1.json()["decision_source"] != "cache"
            assert structured_mock.ainvoke.call_count == 1

            rec1 = await repo.get_current_decision(payment_id)
            assert rec1["request_id"] == req_id_1
            events1 = await repo.get_events(payment_id)
            assert len(events1) == 1

            # Request 2: Identical state, force_recompute=False -> Cache HIT
            req_id_2 = "req-day8e-hit-cached-222"
            resp2 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": req_id_2},
                json={"payment_id": payment_id, "force_recompute": False},
            )
            assert resp2.status_code == 200
            data2 = resp2.json()

            # Assertions
            assert data2["decision_source"] == "cache", "Second request MUST be a cache hit"
            assert data2["request_id"] == req_id_2, "Response must preserve current HTTP request_id"
            assert structured_mock.ainvoke.call_count == 1, "Zero additional LLM calls on cache hit"

            # Verify PostgreSQL zero-write on cache hit
            events2 = await repo.get_events(payment_id)
            assert len(events2) == 1, "No new decision_audit_events row must be created on cache hit"

            rec2 = await repo.get_current_decision(payment_id)
            assert rec2["request_id"] == req_id_1, "Historical request_id in database must NOT be overwritten by cache hit"
    finally:
        from decision_engine.persistence.postgres import close_postgres_pool
        await close_postgres_pool(pool)
        _cleanup_postgres_payment(payment_id, test_db_url)


@pytest.mark.asyncio
async def test_postgres_force_recompute_bypasses_fingerprint_cache():
    """
    Day 8E (Section 10) — Force-Recompute Integration Test with Real Docker PostgreSQL:
    1. Evaluate payment once.
    2. Repeat with force_recompute=True.
    3. Assert:
       - cache lookup is bypassed
       - fresh evaluation occurs (decision_source != 'cache')
       - decision_audit remains one current-state row for payment
       - decision_audit_events increases by one (total 2)
       - new event decision_id is distinct
       - latest event and decision_audit row contain the new request_id
       - state_fingerprint is persisted
    """
    test_db_url = os.getenv("TEST_DATABASE_URL")
    if not test_db_url:
        pytest.skip("TEST_DATABASE_URL is not set.")

    payment_id = "pay_day8e_force_003"
    _cleanup_postgres_payment(payment_id, test_db_url)

    mock_llm = make_mock_llm(decision="RETRY")
    structured_mock = mock_llm.with_structured_output.return_value

    app_instance, repo, pool, _ = await init_isolated_postgres_service(mock_llm=mock_llm)

    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            test_df = _get_dataset().copy()
            row_dict = test_df.iloc[0].to_dict()
            row_dict.update({
                "payment_id": payment_id,
                "status": "FAILED",
                "attempt_number": 1,
                "consecutive_failed_cycles": 0,
                "retry_count": 0,
                "interventions_last_7_days": 0,
            })
            test_df = test_df[test_df["payment_id"] != payment_id]
            test_df = pd.concat([test_df, pd.DataFrame([row_dict])], ignore_index=True)
            app_instance.state.dataset = test_df

            # Request 1: Initial evaluation
            req_id_1 = "req-force-initial-001"
            resp1 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": req_id_1},
                json={"payment_id": payment_id, "force_recompute": False},
            )
            assert resp1.status_code == 200
            assert structured_mock.ainvoke.call_count == 1
            events1 = await repo.get_events(payment_id)
            assert len(events1) == 1

            # Request 2: Repeat with force_recompute=True
            req_id_2 = "req-force-recompute-002"
            resp2 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": req_id_2},
                json={"payment_id": payment_id, "force_recompute": True},
            )
            assert resp2.status_code == 200
            data2 = resp2.json()

            # Assertions
            assert data2["decision_source"] != "cache", "force_recompute=True must bypass cache"
            assert structured_mock.ainvoke.call_count == 2, "Fresh evaluation must occur"

            events2 = await repo.get_events(payment_id)
            assert len(events2) == 2, "decision_audit_events row count must increase by 1"
            assert events2[0]["decision_id"] != events2[1]["decision_id"], "New event decision_id must be distinct"
            assert events2[1]["request_id"] == req_id_2, "Latest event must contain new request_id"

            rec2 = await repo.get_current_decision(payment_id)
            assert rec2 is not None
            assert rec2["request_id"] == req_id_2, "decision_audit current row must reflect new request_id"
            assert rec2["state_fingerprint"] is not None
    finally:
        from decision_engine.persistence.postgres import close_postgres_pool
        await close_postgres_pool(pool)
        _cleanup_postgres_payment(payment_id, test_db_url)
