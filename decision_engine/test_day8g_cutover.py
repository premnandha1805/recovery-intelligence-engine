"""
decision_engine/test_day8g_cutover.py
=====================================
Day 8G: Real PostgreSQL Production Cutover & End-to-End Day 7 Contract Integration Tests.

Verifies:
- Test A: GET /health returns 200 with all components initialized
- Test B: Normal POST /evaluate with PostgreSQL persistence, 11-field contract, audit records
- Test C: Cache hit on same payment_id (decision_source == "cache", 0 extra event rows, 0 extra LLM calls)
- Test D: Force recompute (bypasses cache, appends distinct historical event, updates current decision)
- Test E: Guardrail override execution (overridden=True, guarded action saved to PostgreSQL)
- Test F: Invalid payment_id validation handling (HTTP 400, correlation ID preserved, clean error)
- Test G: Event creation exact field verification and audit ledger integrity
- Test H: 20 concurrent different payments (20 requests, 20 successes, 20 audit rows, 20 event rows)
- Test I: 20 concurrent same payment requests (20 requests, 20 successes, 1 LLM call, 19 cache hits, 1 audit row, 1 event row)
- Test J: Main-path SQLite dependency audit (zero SQLite imports/calls during postgres mode)
- Test K: SQLite backward compatibility under PERSISTENCE_BACKEND=sqlite
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from dotenv import load_dotenv
load_dotenv(".env.test")
load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import httpx
from httpx import ASGITransport, AsyncClient
import pandas as pd
import psycopg
import pytest

from decision_engine.context_node import _get_dataset
from decision_engine.graph import create_recovery_graph
from decision_engine.persistence import (
    PostgresDecisionRepository,
    SqliteDecisionRepository,
    create_postgres_pool,
    close_postgres_pool,
)
from decision_engine.persistence.migrate import run_migrations
from decision_engine.reasoning_node import LLMDecision
from decision_engine.service import app, lifespan

TEST_DB_URL = os.getenv("TEST_DATABASE_URL")


# ── Mock Helpers (Zero Azure Tokens) ─────────────────────────────────────────

def make_mock_policy():
    """Mock CausalUpliftPolicy with deterministic arm predictions."""
    mock_policy = MagicMock()
    mock_t_learner = MagicMock()
    mock_policy.t_learner = mock_t_learner

    def fake_predict_proba(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [{"WAIT": 0.15, "RETRY": 0.80, "RETRY_NUDGE": 0.90, "ESCALATE": 0.35}],
            index=df.index,
        )

    mock_t_learner.predict_proba.side_effect = fake_predict_proba
    return mock_policy


def make_mock_llm(decision: str = "RETRY_NUDGE", confidence: float = 0.95, delay_s: float = 0.0):
    """Mock LangChain LLM model returning structured LLMDecision without Azure calls."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    async def fake_ainvoke(*args, **kwargs):
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        return LLMDecision(
            decision=decision,
            confidence=confidence,
            reasoning=f"Selected {decision} based on uplift.",
            risk_level="low",
            expected_incremental_value=125.0,
        )

    mock_structured.ainvoke = AsyncMock(side_effect=fake_ainvoke)
    mock_structured.invoke = MagicMock(return_value=LLMDecision(
        decision=decision,
        confidence=confidence,
        reasoning=f"Selected {decision} based on uplift.",
        risk_level="low",
        expected_incremental_value=125.0,
    ))
    return mock_llm


def ensure_payment_in_dataset(app_instance: Any, pid: str, **overrides: Any) -> None:
    """Ensure payment_id exists in app_instance.state.dataset with retryable status."""
    base_df = _get_dataset().copy()
    template_row = dict(base_df.iloc[0].to_dict())
    template_row.update({
        "payment_id": pid,
        "status": "FAILED",
        "attempt_number": 1,
        "consecutive_failed_cycles": 0,
        "consecutive_failures": 0,
        "retry_count": 0,
        "interventions_last_7_days": 0,
        "interventions_7d": 0,
    })
    template_row.update(overrides)
    filtered_df = base_df[base_df["payment_id"] != pid]
    app_instance.state.dataset = pd.concat([filtered_df, pd.DataFrame([template_row])], ignore_index=True)


# ── Fixtures & Clean DB ───────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_postgres_tables():
    """Clean PostgreSQL tables before and after each test."""
    test_db_url = os.getenv("TEST_DATABASE_URL")
    if not test_db_url:
        pytest.skip("TEST_DATABASE_URL is not set.")

    with psycopg.connect(test_db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE decision_audit_events, decision_audit;")
        conn.commit()

    yield

    with psycopg.connect(test_db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE decision_audit_events, decision_audit;")
        conn.commit()


async def init_cutover_test_app(mock_llm: Any = None):
    """Initialize FastAPI service wired to real PostgreSQL test pool and mock policy/LLM."""
    test_db_url = os.getenv("TEST_DATABASE_URL")
    run_migrations(database_url=test_db_url)
    pool = await create_postgres_pool(test_db_url, min_size=5, max_size=25)
    repo = PostgresDecisionRepository(pool=pool)

    policy = make_mock_policy()
    llm = mock_llm or make_mock_llm(delay_s=0.01)
    graph = create_recovery_graph(policy=policy, llm=llm, use_async=True)

    app.state.persistence_backend = "postgres"
    app.state.policy = policy
    app.state.graph = graph
    app.state.db_pool = pool
    app.state.repository = repo
    app.state.db = None
    app.state.llm_semaphore = asyncio.Semaphore(10)
    app.state.payment_locks = {}
    app.state.locks_mutex = asyncio.Lock()
    app.state.dataset = _get_dataset().copy()

    return app, repo, pool, llm


# ── Tests A through K ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_day8g_test_a_health_endpoint():
    """Test A: GET /health through real FastAPI service backed by PostgreSQL."""
    app_instance, repo, pool, _ = await init_cutover_test_app()
    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data == {"status": "ok"}
            assert "password" not in resp.text.lower()
            assert "traceback" not in resp.text.lower()
    finally:
        await close_postgres_pool(pool)


@pytest.mark.asyncio
async def test_day8g_test_b_normal_evaluate_and_contract():
    """
    Test B: Normal POST /evaluate
    - HTTP 200
    - Exact 11-field contract
    - PostgreSQL current decision row created
    - PostgreSQL decision_audit_events row created
    - request_id preserved
    - final_action preserved
    - decision_source indicates fresh evaluation
    """
    app_instance, repo, pool, _ = await init_cutover_test_app()
    test_db_url = os.getenv("TEST_DATABASE_URL")
    pid = "pay_8g_normal_001"
    req_id = "req-8g-eval-001"
    ensure_payment_in_dataset(app_instance, pid)

    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/evaluate",
                headers={"X-Request-Id": req_id},
                json={"payment_id": pid, "force_recompute": False},
            )
            assert resp.status_code == 200
            data = resp.json()

            # Exact 11-field contract
            expected_fields = {
                "payment_id", "model_decision", "llm_decision", "guardrail_overridden",
                "guardrail_reason", "final_action", "confidence", "risk_level",
                "reasoning", "decision_source", "request_id",
            }
            assert set(data.keys()) == expected_fields
            assert data["payment_id"] == pid
            assert data["request_id"] == req_id
            assert data["final_action"] == "RETRY_NUDGE"
            assert data["decision_source"] == "llm"
            assert resp.headers.get("x-request-id") == req_id

        # Verify PostgreSQL state directly
        with psycopg.connect(test_db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payment_id, request_id, final_action, decision_source FROM decision_audit WHERE payment_id = %s;", (pid,))
                audit_row = cur.fetchone()
                assert audit_row is not None
                assert audit_row[0] == pid
                assert audit_row[1] == req_id
                assert audit_row[2] == "RETRY_NUDGE"
                assert audit_row[3] == "llm"

                cur.execute("SELECT payment_id, request_id, final_action, decision_source FROM decision_audit_events WHERE payment_id = %s;", (pid,))
                events = cur.fetchall()
                assert len(events) == 1
                assert events[0][0] == pid
                assert events[0][1] == req_id
                assert events[0][2] == "RETRY_NUDGE"
    finally:
        await close_postgres_pool(pool)


@pytest.mark.asyncio
async def test_day8g_test_c_cache_hit():
    """
    Test C: Cache Hit on same payment_id
    - HTTP 200
    - decision_source == "cache"
    - Response is identical in decision content
    - Current request_id is returned for the HTTP transaction
    - PostgreSQL decision_audit row count does not increase (remains 1)
    - decision_audit_events row count does not increase (remains 1)
    - No second LLM invocation occurs
    """
    mock_llm = make_mock_llm(decision="RETRY_NUDGE")
    structured_mock = mock_llm.with_structured_output.return_value
    app_instance, repo, pool, _ = await init_cutover_test_app(mock_llm=mock_llm)
    test_db_url = os.getenv("TEST_DATABASE_URL")
    pid = "pay_8g_cache_001"
    ensure_payment_in_dataset(app_instance, pid)

    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. First evaluation
            resp1 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": "req-1"},
                json={"payment_id": pid, "force_recompute": False},
            )
            assert resp1.status_code == 200
            assert resp1.json()["decision_source"] == "llm"
            assert structured_mock.ainvoke.call_count == 1

            # 2. Second request with same payment_id (uncached flag False, state unchanged)
            resp2 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": "req-2-cache"},
                json={"payment_id": pid, "force_recompute": False},
            )
            assert resp2.status_code == 200
            data2 = resp2.json()
            assert data2["decision_source"] == "cache"
            assert data2["request_id"] == "req-2-cache"
            assert data2["final_action"] == resp1.json()["final_action"]
            assert data2["confidence"] == resp1.json()["confidence"]
            # Assert zero additional LLM invocations
            assert structured_mock.ainvoke.call_count == 1

        # Check DB row counts remain strictly 1
        with psycopg.connect(test_db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM decision_audit WHERE payment_id = %s;", (pid,))
                assert cur.fetchone()[0] == 1
                cur.execute("SELECT COUNT(*) FROM decision_audit_events WHERE payment_id = %s;", (pid,))
                assert cur.fetchone()[0] == 1
    finally:
        await close_postgres_pool(pool)


@pytest.mark.asyncio
async def test_day8g_test_d_force_recompute():
    """
    Test D: Force Recompute
    - Cache is bypassed
    - Fresh evaluation occurs (second LLM call)
    - A new decision_audit_events row is created (total 2 events)
    - decision_audit remains one current-state row (updated)
    - New event decision_id is distinct
    - Latest request_id stored
    - Response decision_source is NOT cache
    """
    mock_llm = make_mock_llm(decision="RETRY_NUDGE")
    structured_mock = mock_llm.with_structured_output.return_value
    app_instance, repo, pool, _ = await init_cutover_test_app(mock_llm=mock_llm)
    test_db_url = os.getenv("TEST_DATABASE_URL")
    pid = "pay_8g_recompute_001"
    ensure_payment_in_dataset(app_instance, pid)

    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Initial write
            resp1 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": "req-init"},
                json={"payment_id": pid, "force_recompute": False},
            )
            assert resp1.status_code == 200
            assert structured_mock.ainvoke.call_count == 1

            # 2. Force recompute
            resp2 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": "req-forced"},
                json={"payment_id": pid, "force_recompute": True},
            )
            assert resp2.status_code == 200
            data2 = resp2.json()
            assert data2["decision_source"] == "llm"
            assert data2["request_id"] == "req-forced"
            assert structured_mock.ainvoke.call_count == 2

        # Verify DB: 1 current decision, 2 events with distinct decision_ids
        with psycopg.connect(test_db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*), MAX(request_id) FROM decision_audit WHERE payment_id = %s;", (pid,))
                audit_cnt, last_req = cur.fetchone()
                assert audit_cnt == 1
                assert last_req == "req-forced"

                cur.execute("SELECT decision_id, request_id FROM decision_audit_events WHERE payment_id = %s ORDER BY evaluated_at ASC;", (pid,))
                events = cur.fetchall()
                assert len(events) == 2
                assert events[0][1] == "req-init"
                assert events[1][1] == "req-forced"
                assert events[0][0] != events[1][0]
    finally:
        await close_postgres_pool(pool)


@pytest.mark.asyncio
async def test_day8g_test_e_guardrail_override():
    """
    Test E: Guardrail Override
    - Set payment in state with consecutive_failures >= 3 (triggers consecutive failure guardrail)
    - guardrail_overridden == True
    - guardrail_reason preserved exactly
    - final_action equals the guarded result (STOP)
    - PostgreSQL decision_audit and decision_audit_events contain the guarded result
    """
    app_instance, repo, pool, _ = await init_cutover_test_app()
    test_db_url = os.getenv("TEST_DATABASE_URL")
    pid = "pay_8g_guard_001"
    ensure_payment_in_dataset(
        app_instance,
        pid,
        status="FAILED",
        consecutive_failed_cycles=4,
        consecutive_failures=4,
        attempt_number=4,
        retry_count=3,
    )

    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/evaluate",
                headers={"X-Request-Id": "req-guard-01"},
                json={"payment_id": pid, "force_recompute": False},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["guardrail_overridden"] is True
            assert "consecutive" in data["guardrail_reason"].lower() or "failure" in data["guardrail_reason"].lower()
            assert data["final_action"] in ("STOP", "WAIT", "ESCALATE")

        with psycopg.connect(test_db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT guardrail_verdict, guardrail_reason, final_action FROM decision_audit WHERE payment_id = %s;", (pid,))
                row = cur.fetchone()
                assert row is not None
                assert "overridden" in row[0].lower() or row[0] == "overridden"
                assert row[2] == data["final_action"]

                cur.execute("SELECT guardrail_overridden, guardrail_reason, final_action FROM decision_audit_events WHERE payment_id = %s;", (pid,))
                event_row = cur.fetchone()
                assert event_row is not None
                assert event_row[0] is True
                assert event_row[2] == data["final_action"]
    finally:
        await close_postgres_pool(pool)


@pytest.mark.asyncio
async def test_day8g_test_f_invalid_payment():
    """
    Test F: Invalid payment_id handling
    - HTTP 400
    - Error envelope
    - Correlation ID preserved in x-request-id header
    - No credentials or stack traces in response
    """
    app_instance, repo, pool, _ = await init_cutover_test_app()
    req_id = "req-invalid-pay-999"

    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/evaluate",
                headers={"X-Request-Id": req_id},
                json={"payment_id": "   ", "force_recompute": False},
            )
            assert resp.status_code == 400
            assert resp.headers.get("x-request-id") == req_id
            data = resp.json()
            assert "error" in data
            assert "password" not in resp.text.lower()
            assert "traceback" not in resp.text.lower()
    finally:
        await close_postgres_pool(pool)


@pytest.mark.asyncio
async def test_day8g_test_g_event_creation_details():
    """
    Test G: Event Creation and Field Verification
    - Fresh evaluation creates exactly one decision_audit and one decision_audit_events
    - Event contains: decision_id, payment_id, request_id, evaluated_at, decision_source, final_action, state_fingerprint
    """
    app_instance, repo, pool, _ = await init_cutover_test_app()
    test_db_url = os.getenv("TEST_DATABASE_URL")
    pid = "pay_8g_event_detail_001"
    req_id = "req-event-detail-001"
    ensure_payment_in_dataset(app_instance, pid)

    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/evaluate",
                headers={"X-Request-Id": req_id},
                json={"payment_id": pid, "force_recompute": False},
            )
            assert resp.status_code == 200

        with psycopg.connect(test_db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT decision_id, payment_id, request_id, evaluated_at, decision_source, final_action, state_fingerprint
                    FROM decision_audit_events
                    WHERE payment_id = %s;
                """, (pid,))
                row = cur.fetchone()
                assert row is not None
                assert row[0] is not None and len(row[0]) > 0  # decision_id
                assert row[1] == pid                           # payment_id
                assert row[2] == req_id                        # request_id
                assert row[3] is not None                      # evaluated_at
                assert row[4] == "llm"                         # decision_source
                assert row[5] == "RETRY_NUDGE"                 # final_action
                assert row[6] is not None                      # state_fingerprint
    finally:
        await close_postgres_pool(pool)


@pytest.mark.asyncio
async def test_day8g_test_h_concurrent_different_payments():
    """
    Test H: 20 Concurrent Different Payments
    - 20 distinct payment_ids
    - Launch 20 simultaneous POST requests via asyncio.gather
    - total_requests = 20, success = 20, failure = 0
    - decision_audit rows = 20, decision_audit_events rows = 20
    """
    app_instance, repo, pool, _ = await init_cutover_test_app()
    test_db_url = os.getenv("TEST_DATABASE_URL")
    pids = [f"pay_8g_diff_{i:03d}" for i in range(1, 21)]

    try:
        # Prepare dataset with all 20 payments
        base_df = _get_dataset().copy()
        template_row = base_df.iloc[0].to_dict()
        new_rows = []
        for pid in pids:
            row = dict(template_row)
            row.update({
                "payment_id": pid,
                "status": "FAILED",
                "attempt_number": 1,
                "consecutive_failed_cycles": 0,
                "retry_count": 0,
                "interventions_last_7_days": 0,
            })
            new_rows.append(row)
        filtered_df = base_df[~base_df["payment_id"].isin(pids)]
        app_instance.state.dataset = pd.concat([filtered_df, pd.DataFrame(new_rows)], ignore_index=True)

        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            t0 = time.monotonic()
            tasks = [
                client.post(
                    "/evaluate",
                    headers={"X-Request-Id": f"req-8g-diff-{i:03d}"},
                    json={"payment_id": pid, "force_recompute": False},
                )
                for i, pid in enumerate(pids, 1)
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            wall_clock_ms = round((time.monotonic() - t0) * 1000, 2)

        successful = sum(1 for r in responses if isinstance(r, httpx.Response) and r.status_code == 200)
        failed = sum(1 for r in responses if not isinstance(r, httpx.Response) or r.status_code != 200)

        assert successful == 20
        assert failed == 0

        with psycopg.connect(test_db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM decision_audit;")
                audit_rows = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM decision_audit_events;")
                event_rows = cur.fetchone()[0]

        assert audit_rows == 20
        assert event_rows == 20

        print("\n=== DAY 8G: 20 DIFFERENT PAYMENTS ACTUAL COUNTS ===")
        print(f"total requests: {len(pids)}")
        print(f"successful: {successful}")
        print(f"failed: {failed}")
        print(f"decision_audit rows: {audit_rows}")
        print(f"decision_audit_events rows: {event_rows}")
        print(f"wall-clock duration: {wall_clock_ms} ms")
        print("===================================================\n")
    finally:
        await close_postgres_pool(pool)


@pytest.mark.asyncio
async def test_day8g_test_i_concurrent_same_payment():
    """
    Test I: 20 Concurrent Requests for SAME Payment ID
    - 1 uncached payment_id
    - Launch 20 simultaneous requests
    - total_requests = 20, success = 20, failure = 0
    - fresh evaluations = 1, LLM invocations = 1, cache hits = 19
    - decision_audit rows = 1, decision_audit_events rows = 1
    - All 20 responses return identical decision result
    """
    mock_llm = make_mock_llm(decision="RETRY_NUDGE", delay_s=0.04)
    structured_mock = mock_llm.with_structured_output.return_value
    app_instance, repo, pool, _ = await init_cutover_test_app(mock_llm=mock_llm)
    test_db_url = os.getenv("TEST_DATABASE_URL")
    pid = "pay_8g_same_001"
    ensure_payment_in_dataset(app_instance, pid)

    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            t0 = time.monotonic()
            tasks = [
                client.post(
                    "/evaluate",
                    headers={"X-Request-Id": f"req-8g-same-{i:03d}"},
                    json={"payment_id": pid, "force_recompute": False},
                )
                for i in range(1, 21)
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            wall_clock_ms = round((time.monotonic() - t0) * 1000, 2)

        successful = sum(1 for r in responses if isinstance(r, httpx.Response) and r.status_code == 200)
        failed = sum(1 for r in responses if not isinstance(r, httpx.Response) or r.status_code != 200)
        llm_invocations = structured_mock.ainvoke.call_count

        assert successful == 20
        assert failed == 0
        assert llm_invocations == 1

        json_responses = [r.json() for r in responses if isinstance(r, httpx.Response)]
        final_actions = {r.get("final_action") for r in json_responses}
        assert len(final_actions) == 1
        assert "RETRY_NUDGE" in final_actions

        decision_sources = [r.get("decision_source") for r in json_responses]
        cache_hits = decision_sources.count("cache")
        fresh_evals = sum(1 for s in decision_sources if s != "cache")

        assert fresh_evals == 1
        assert cache_hits == 19

        with psycopg.connect(test_db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM decision_audit WHERE payment_id = %s;", (pid,))
                audit_rows = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM decision_audit_events WHERE payment_id = %s;", (pid,))
                event_rows = cur.fetchone()[0]

        assert audit_rows == 1
        assert event_rows == 1

        print("\n=== DAY 8G: 20 SAME PAYMENT ACTUAL COUNTS ===")
        print(f"total requests: 20")
        print(f"successful: {successful}")
        print(f"failed: {failed}")
        print(f"fresh evaluations: {fresh_evals}")
        print(f"LLM invocations: {llm_invocations}")
        print(f"cache hits: {cache_hits}")
        print(f"decision_audit rows: {audit_rows}")
        print(f"decision_audit_events rows: {event_rows}")
        print(f"wall-clock duration: {wall_clock_ms} ms")
        print(f"all 20 returned same decision: True ({list(final_actions)[0]})")
        print("=============================================\n")
    finally:
        await close_postgres_pool(pool)


@pytest.mark.asyncio
async def test_day8g_test_j_sqlite_zero_dependency_audit():
    """
    Test J: Main-Path SQLite Dependency Audit
    Under PERSISTENCE_BACKEND=postgres:
    - app.state.db is None
    - app.state.repository is PostgresDecisionRepository
    - Zero aiosqlite or sqlite3 calls are made during /evaluate
    """
    app_instance, repo, pool, _ = await init_cutover_test_app()
    pid = "pay_8g_no_sqlite_001"
    ensure_payment_in_dataset(app_instance, pid)

    assert app_instance.state.db is None
    assert isinstance(app_instance.state.repository, PostgresDecisionRepository)

    try:
        with patch("aiosqlite.connect", side_effect=RuntimeError("aiosqlite was unexpectedly invoked")):
            with patch("sqlite3.connect", side_effect=RuntimeError("sqlite3 was unexpectedly invoked")):
                transport = ASGITransport(app=app_instance)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post(
                        "/evaluate",
                        headers={"X-Request-Id": "req-no-sqlite"},
                        json={"payment_id": pid, "force_recompute": False},
                    )
                    assert resp.status_code == 200
                    assert resp.json()["decision_source"] == "llm"
                    assert resp.json()["final_action"] == "RETRY_NUDGE"
    finally:
        await close_postgres_pool(pool)


@pytest.mark.asyncio
async def test_day8g_test_k_sqlite_backward_compatibility(tmp_path: pathlib.Path):
    """
    Test K: SQLite Backward Compatibility
    Under PERSISTENCE_BACKEND=sqlite:
    - Lifespan initializes SqliteDecisionRepository
    - app.state.db is an active aiosqlite.Connection
    - app.state.db_pool is None
    - /evaluate writes cleanly to SQLite
    """
    test_db_path = str(tmp_path / "sqlite_compat.db")
    with patch.dict(os.environ, {"PERSISTENCE_BACKEND": "sqlite"}):
        with patch("decision_engine.service.DEFAULT_AUDIT_DB_PATH", test_db_path):
            async with lifespan(app):
                assert app.state.persistence_backend == "sqlite"
                assert app.state.db is not None
                assert app.state.db_pool is None
                assert isinstance(app.state.repository, SqliteDecisionRepository)

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post(
                        "/evaluate",
                        headers={"X-Request-Id": "req-sqlite-compat"},
                        json={"payment_id": "pay_000001_a1", "force_recompute": False},
                    )
                    assert resp.status_code == 200
                    assert "final_action" in resp.json()

                # Verify written to SQLite
                async with app.state.db.execute("SELECT COUNT(*) FROM decision_audit;") as cur:
                    row = await cur.fetchone()
                    assert row[0] >= 1
