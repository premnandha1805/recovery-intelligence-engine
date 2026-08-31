"""
decision_engine/test_day8i_failure_recovery.py
==============================================
Day 8I: PostgreSQL Failure and Recovery Testing.

Verifies:
1. Scenario 1: PostgreSQL unavailable at startup -> FastAPI fails fast with clear error,
   no SQLite fallback, no retry loop, respects DB_CONNECT_TIMEOUT_MS.
2. Scenario 2: PostgreSQL connection failure during an in-flight request:
   - Controlled error envelope (HTTP 500)
   - Preserves request_id in response headers and structured logs
   - No raw traceback or credentials leaked
   - No false persistence reported
   - Zero SQLite fallback
   - No client-side retry
"""

from __future__ import annotations

import asyncio
import logging
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
    create_postgres_pool,
    close_postgres_pool,
)
from decision_engine.persistence.migrate import run_migrations
from decision_engine.reasoning_node import LLMDecision
from decision_engine.service import app, lifespan


TEST_DB_URL = os.getenv("TEST_DATABASE_URL")


# ── Mock Helpers (Zero Azure Tokens) ─────────────────────────────────────────

def make_mock_policy():
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


def make_mock_llm():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    async def fake_ainvoke(*args, **kwargs):
        return LLMDecision(
            decision="RETRY_NUDGE",
            confidence=0.95,
            reasoning="Selected RETRY_NUDGE based on uplift.",
            risk_level="low",
            expected_incremental_value=125.0,
        )

    mock_structured.ainvoke = AsyncMock(side_effect=fake_ainvoke)
    mock_structured.invoke = MagicMock(return_value=LLMDecision(
        decision="RETRY_NUDGE",
        confidence=0.95,
        reasoning="Selected RETRY_NUDGE based on uplift.",
        risk_level="low",
        expected_incremental_value=125.0,
    ))
    return mock_llm


def ensure_test_payment(app_instance: Any, pid: str) -> None:
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
    filtered_df = base_df[base_df["payment_id"] != pid]
    app_instance.state.dataset = pd.concat([filtered_df, pd.DataFrame([template_row])], ignore_index=True)


async def init_failure_test_app():
    test_db_url = os.getenv("TEST_DATABASE_URL")
    run_migrations(database_url=test_db_url)
    pool = await create_postgres_pool(test_db_url, min_size=2, max_size=5)
    repo = PostgresDecisionRepository(pool=pool)

    policy = make_mock_policy()
    llm = make_mock_llm()
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

    return app, repo, pool


# ── Scenario 1: Startup Failure When PostgreSQL is Unavailable ──────────────

@pytest.mark.asyncio
async def test_scenario1_startup_fails_fast_when_postgres_unavailable():
    """
    Scenario 1:
    - PostgreSQL is configured as PERSISTENCE_BACKEND=postgres.
    - PostgreSQL is unavailable (unreachable host/port with short timeout).
    - FastAPI lifespan fails fast by raising RuntimeError.
    - Error message clearly identifies PostgreSQL pool initialization failure.
    - NO automatic fallback to SQLite occurs.
    - Does not hang indefinitely; respects connect timeout.
    """
    unreachable_url = "postgresql://recovery_test:invalid_pass@127.0.0.1:59999/recovery_test"

    t0 = time.monotonic()
    with patch.dict(os.environ, {
        "PERSISTENCE_BACKEND": "postgres",
        "DATABASE_URL": unreachable_url,
        "DB_CONNECT_TIMEOUT_MS": "1500",
    }):
        with pytest.raises(RuntimeError) as exc_info:
            async with lifespan(app):
                pass
    elapsed = time.monotonic() - t0

    err_str = str(exc_info.value)
    # 1. Error identifies PostgreSQL pool initialization failure
    assert "PostgreSQL pool initialization failed" in err_str or "Service refusing to start" in err_str
    # 2. Bounded by connect timeout
    assert elapsed < 5.0, f"Startup took {elapsed:.2f}s, expected fast failure (<5s)"
    # 3. No SQLite fallback occurred
    assert app.state.db is None
    assert getattr(app.state, "persistence_backend", "") == "postgres"

    print("\n=== DAY 8I SCENARIO 1: STARTUP FAIL-FAST RESULT ===")
    print(f"observed error: {err_str[:120]}...")
    print(f"startup duration before failure: {elapsed:.2f}s")
    print(f"sqlite fallback occurred: False")
    print("===================================================\n")


# ── Scenario 2: In-Flight Request Failure (Connection Lost During Execution) ─

@pytest.mark.asyncio
async def test_scenario2_mid_request_failure_during_save(caplog: pytest.LogCaptureFixture):
    """
    Scenario 2A: PostgreSQL connection fails during write (save_decision_with_event).
    - Request fails cleanly with HTTP 500
    - Controlled error envelope: {"error": {"code": "INTERNAL_ERROR", "message": "Internal server error"}}
    - Request ID preserved in response header x-request-id
    - Server failure log contains the same request_id
    - No raw traceback or secrets exposed in HTTP response
    - No false persisted success reported
    - Zero rows written to PostgreSQL
    - Zero SQLite fallback
    """
    app_instance, repo, pool = await init_failure_test_app()
    test_db_url = os.getenv("TEST_DATABASE_URL")
    pid = "pay_8i_fail_write_001"
    req_id = "req-8i-inflight-write-fail-001"
    ensure_test_payment(app_instance, pid)

    # Clean DB before test
    with psycopg.connect(test_db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM decision_audit_events WHERE payment_id = %s;", (pid,))
            cur.execute("DELETE FROM decision_audit WHERE payment_id = %s;", (pid,))
        conn.commit()

    try:
        # Simulate connection loss during repository write
        with patch.object(
            repo,
            "save_decision_with_event",
            side_effect=psycopg.OperationalError("server closed the connection unexpectedly"),
        ):
            with caplog.at_level(logging.ERROR):
                transport = ASGITransport(app=app_instance, raise_app_exceptions=False)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post(
                        "/evaluate",
                        headers={"X-Request-Id": req_id},
                        json={"payment_id": pid, "force_recompute": False},
                    )

            # 1. HTTP 500 status code
            assert resp.status_code == 500

            # 2. Controlled error envelope
            data = resp.json()
            assert data == {"error": {"code": "INTERNAL_ERROR", "message": "Internal server error"}}

            # 3. Request ID in HTTP response header
            assert resp.headers.get("x-request-id") == req_id

            # 4. Same request_id in server-side logs
            log_records = [r for r in caplog.records if req_id in r.message]
            assert len(log_records) >= 1, f"Expected request_id {req_id} in log records"

            # 5. No credentials, DATABASE_URL, or raw tracebacks in response
            text = resp.text.lower()
            assert "password" not in text
            assert "postgresql://" not in text
            assert "traceback" not in text
            assert "operationalerror" not in text

            # 6. No SQLite fallback
            assert app_instance.state.db is None

        # 7. Persistence truthfulness: verify zero rows in PostgreSQL
        with psycopg.connect(test_db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM decision_audit WHERE payment_id = %s;", (pid,))
                assert cur.fetchone()[0] == 0
                cur.execute("SELECT COUNT(*) FROM decision_audit_events WHERE payment_id = %s;", (pid,))
                assert cur.fetchone()[0] == 0

        print("\n=== DAY 8I SCENARIO 2A: WRITE FAILURE RESULT ===")
        print(f"HTTP status: {resp.status_code}")
        print(f"response body: {data}")
        print(f"x-request-id header: {resp.headers.get('x-request-id')}")
        print(f"logged with request_id: True")
        print(f"persisted rows in PostgreSQL: 0")
        print(f"sqlite fallback: None (app.state.db is None)")
        print("================================================\n")
    finally:
        await close_postgres_pool(pool)


@pytest.mark.asyncio
async def test_scenario2_mid_request_failure_during_cache_read(caplog: pytest.LogCaptureFixture):
    """
    Scenario 2B: PostgreSQL connection fails during cache read (get_current_decision).
    - Request fails cleanly with HTTP 500
    - Controlled error envelope returned
    - Request ID preserved in header and logs
    - No false cached response returned
    - No SQLite fallback
    """
    app_instance, repo, pool = await init_failure_test_app()
    pid = "pay_8i_fail_read_001"
    req_id = "req-8i-inflight-read-fail-002"
    ensure_test_payment(app_instance, pid)

    try:
        with patch.object(
            repo,
            "get_current_decision",
            side_effect=psycopg.OperationalError("connection reset by peer"),
        ):
            with caplog.at_level(logging.ERROR):
                transport = ASGITransport(app=app_instance, raise_app_exceptions=False)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post(
                        "/evaluate",
                        headers={"X-Request-Id": req_id},
                        json={"payment_id": pid, "force_recompute": False},
                    )

            assert resp.status_code == 500
            assert resp.headers.get("x-request-id") == req_id
            assert resp.json() == {"error": {"code": "INTERNAL_ERROR", "message": "Internal server error"}}
            assert app_instance.state.db is None

            log_records = [r for r in caplog.records if req_id in r.message]
            assert len(log_records) >= 1
    finally:
        await close_postgres_pool(pool)


@pytest.mark.asyncio
async def test_scenario2_no_client_or_server_retry_loop():
    """
    Scenario 2C: Verify DB failure does not cause an infinite retry loop or excessive execution time.
    """
    app_instance, repo, pool = await init_failure_test_app()
    pid = "pay_8i_retry_check_001"
    req_id = "req-8i-retry-check-003"
    ensure_test_payment(app_instance, pid)

    call_count = 0

    async def fake_failing_save(**kwargs):
        nonlocal call_count
        call_count += 1
        raise psycopg.OperationalError("database connection disconnected")

    try:
        with patch.object(repo, "save_decision_with_event", side_effect=fake_failing_save):
            t0 = time.monotonic()
            transport = ASGITransport(app=app_instance, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/evaluate",
                    headers={"X-Request-Id": req_id},
                    json={"payment_id": pid, "force_recompute": False},
                )
            elapsed = time.monotonic() - t0

            # Single execution, zero retry loop
            assert call_count == 1
            assert resp.status_code == 500
            assert elapsed < 2.0, f"Request took {elapsed:.2f}s, suspected retry loop"
    finally:
        await close_postgres_pool(pool)
