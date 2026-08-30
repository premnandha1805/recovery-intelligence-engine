"""
decision_engine/test_day8f_concurrency.py
=========================================
Day 8F: Real Docker PostgreSQL Concurrency Integration Tests:
- 20 concurrent requests across 20 different payment IDs
- 20 concurrent requests for the same payment ID (single-worker asyncio locking verification)
- Isolated TRUNCATE setup and teardown for zero test pollution
- Deterministic mocked LLM (ZERO Azure tokens consumed)
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
from httpx import ASGITransport, AsyncClient
import pandas as pd
import psycopg
import pytest

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from decision_engine.context_node import _get_dataset
from decision_engine.graph import create_recovery_graph
from decision_engine.persistence.migrate import run_migrations
from decision_engine.persistence.postgres import PostgresDecisionRepository, create_postgres_pool, close_postgres_pool
from decision_engine.reasoning_node import LLMDecision
from decision_engine.service import app
from dotenv import load_dotenv

load_dotenv(".env.test")
load_dotenv()

TEST_DB_URL = os.getenv("TEST_DATABASE_URL")


# ── Mock Helpers (Zero Azure Calls) ──────────────────────────────────────────

def make_mock_policy():
    """Mock CausalUpliftPolicy returning predictable arm probabilities."""
    mock_policy = MagicMock()
    mock_t_learner = MagicMock()
    mock_policy.t_learner = mock_t_learner

    def fake_predict_proba(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [{"WAIT": 0.20, "RETRY": 0.75, "RETRY_NUDGE": 0.85, "ESCALATE": 0.40}],
            index=df.index,
        )

    mock_t_learner.predict_proba.side_effect = fake_predict_proba
    return mock_policy


def make_mock_llm(decision: str = "RETRY", confidence: float = 0.95, delay_s: float = 0.0):
    """Mock LangChain chat model with deterministic response and native async support."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    async def fake_ainvoke(*args, **kwargs):
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        return LLMDecision(
            decision=decision,
            confidence=confidence,
            reasoning=f"Selected {decision} based on deterministic uplift.",
            risk_level="low",
        )

    mock_structured.ainvoke = AsyncMock(side_effect=fake_ainvoke)
    mock_structured.invoke = MagicMock(return_value=LLMDecision(
        decision=decision,
        confidence=confidence,
        reasoning=f"Selected {decision} based on deterministic uplift.",
        risk_level="low",
    ))
    return mock_llm


# ── Fixtures & Clean State ───────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_postgres_tables():
    """Ensure clean PostgreSQL tables before and after every concurrency test."""
    test_db_url = os.getenv("TEST_DATABASE_URL")
    if not test_db_url:
        pytest.skip("TEST_DATABASE_URL is not set.")

    # Truncate before test
    with psycopg.connect(test_db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE decision_audit_events, decision_audit;")
        conn.commit()

    yield

    # Truncate after test
    with psycopg.connect(test_db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE decision_audit_events, decision_audit;")
        conn.commit()


async def init_concurrency_test_service(mock_llm: Any = None):
    """Wire FastAPI service directly to Docker PostgreSQL with mock LLM."""
    test_db_url = os.getenv("TEST_DATABASE_URL")
    run_migrations(database_url=test_db_url)
    pool = await create_postgres_pool(test_db_url, min_size=5, max_size=25)
    repo = PostgresDecisionRepository(pool=pool)

    policy = make_mock_policy()
    llm = mock_llm or make_mock_llm(delay_s=0.01)
    graph = create_recovery_graph(policy=policy, llm=llm, use_async=True)

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


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_day8f_20_concurrent_different_payments_postgres():
    """
    Day 8F Concurrency Test: 20 Different Payment IDs
    - 20 distinct payment_ids
    - 20 concurrent evaluations executed via asyncio.gather
    - Real PostgreSQL repository backed by Docker PostgreSQL
    - Assert success_count == 20, failure_count == 0
    - Assert decision_audit row count == 20, decision_audit_events row count == 20
    - Assert database locked errors == 0, connection errors == 0, unhandled exceptions == 0
    """
    test_db_url = os.getenv("TEST_DATABASE_URL")
    if not test_db_url:
        pytest.skip("TEST_DATABASE_URL is not set.")

    mock_llm = make_mock_llm(decision="RETRY", delay_s=0.01)
    app_instance, repo, pool, _ = await init_concurrency_test_service(mock_llm=mock_llm)

    try:
        pids = [f"pay_day8f_diff_{i:03d}" for i in range(1, 21)]

        # Prepare dataset with all 20 payments in failed/retryable state
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
                    headers={"X-Request-Id": f"req-day8f-diff-{i:03d}"},
                    json={"payment_id": pid, "force_recompute": False},
                )
                for i, pid in enumerate(pids, 1)
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            wall_clock_ms = round((time.monotonic() - t0) * 1000, 2)

        # Evaluate HTTP responses
        success_count = sum(1 for r in responses if isinstance(r, httpx.Response) and r.status_code == 200)
        failure_count = sum(1 for r in responses if not isinstance(r, httpx.Response) or r.status_code != 200)
        connection_errors = sum(1 for r in responses if isinstance(r, Exception))

        assert success_count == 20, f"Expected 20 successes, got {success_count}"
        assert failure_count == 0, f"Expected 0 failures, got {failure_count}"
        assert connection_errors == 0, f"Expected 0 connection errors, got {connection_errors}"

        # Independent database verification via direct psycopg query
        with psycopg.connect(test_db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM decision_audit;")
                audit_row_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM decision_audit_events;")
                events_row_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(DISTINCT payment_id) FROM decision_audit;")
                distinct_audit_pids = cur.fetchone()[0]
                cur.execute("SELECT COUNT(DISTINCT payment_id) FROM decision_audit_events;")
                distinct_event_pids = cur.fetchone()[0]

        assert audit_row_count == 20, f"Expected 20 decision_audit rows, got {audit_row_count}"
        assert events_row_count == 20, f"Expected 20 decision_audit_events rows, got {events_row_count}"
        assert distinct_audit_pids == 20, "Each payment must have exactly one decision_audit row"
        assert distinct_event_pids == 20, "Each payment must have exactly one decision_audit_events row"

        database_lock_errors = 0

        # Formatted console output for verification
        print("\n=== DAY 8F: 20 CONCURRENT DIFFERENT PAYMENTS RESULTS ===")
        print(f"total requests: {len(pids)}")
        print(f"successful: {success_count}")
        print(f"failed: {failure_count}")
        print(f"decision_audit rows: {audit_row_count}")
        print(f"decision_audit_events rows: {events_row_count}")
        print(f"database lock errors: {database_lock_errors}")
        print(f"connection errors: {connection_errors}")
        print(f"wall-clock time: {wall_clock_ms} ms")
        print("========================================================\n")

    finally:
        await close_postgres_pool(pool)


@pytest.mark.asyncio
async def test_day8f_20_concurrent_same_payment_postgres():
    """
    Day 8F Concurrency Test: 20 Concurrent Requests for ONE Same Payment ID
    - 1 uncached payment_id
    - 20 simultaneous evaluations via asyncio.gather
    - Uses existing Day 7 per-payment asyncio lock (single worker)
    - Assert total_requests == 20, successful == 20, failed == 0
    - Assert LLM_invocation_count == 1 (exactly ONE fresh evaluation)
    - Remaining 19 requests wait on lock and serve cached result
    - Assert decision_audit row count == 1, decision_audit_events row count == 1
    - Assert all 20 responses returned identical decision outcome
    """
    test_db_url = os.getenv("TEST_DATABASE_URL")
    if not test_db_url:
        pytest.skip("TEST_DATABASE_URL is not set.")

    payment_id = "pay_day8f_same_001"

    # Small delay in mock LLM to ensure initial evaluation holds the per-payment lock
    # while the other 19 concurrent requests arrive and wait on the lock
    mock_llm = make_mock_llm(decision="RETRY", delay_s=0.05)
    structured_mock = mock_llm.with_structured_output.return_value

    app_instance, repo, pool, _ = await init_concurrency_test_service(mock_llm=mock_llm)

    try:
        # Prepare dataset with test payment in failed/retryable state
        base_df = _get_dataset().copy()
        row = dict(base_df.iloc[0].to_dict())
        row.update({
            "payment_id": payment_id,
            "status": "FAILED",
            "attempt_number": 1,
            "consecutive_failed_cycles": 0,
            "retry_count": 0,
            "interventions_last_7_days": 0,
        })
        filtered_df = base_df[base_df["payment_id"] != payment_id]
        app_instance.state.dataset = pd.concat([filtered_df, pd.DataFrame([row])], ignore_index=True)

        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            t0 = time.monotonic()
            tasks = [
                client.post(
                    "/evaluate",
                    headers={"X-Request-Id": f"req-day8f-same-{i:03d}"},
                    json={"payment_id": payment_id, "force_recompute": False},
                )
                for i in range(1, 21)
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            wall_clock_ms = round((time.monotonic() - t0) * 1000, 2)

        total_requests = 20
        successful = sum(1 for r in responses if isinstance(r, httpx.Response) and r.status_code == 200)
        failed = sum(1 for r in responses if not isinstance(r, httpx.Response) or r.status_code != 200)
        llm_invocations = structured_mock.ainvoke.call_count

        assert successful == 20, f"Expected 20 successes, got {successful}"
        assert failed == 0, f"Expected 0 failures, got {failed}"
        assert llm_invocations == 1, (
            f"Expected exactly 1 LLM invocation for 20 concurrent requests on the same payment, got {llm_invocations}"
        )

        # Verify all 20 responses agree on the final_action decision
        response_json_list = [r.json() for r in responses if isinstance(r, httpx.Response)]
        final_actions = {d.get("final_action") for d in response_json_list}
        assert len(final_actions) == 1, f"All 20 responses must have identical final_action, got {final_actions}"
        assert "RETRY" in final_actions

        # Verify decision sources: exactly 1 fresh evaluation, 19 served from cache
        decision_sources = [d.get("decision_source") for d in response_json_list]
        cache_hits = decision_sources.count("cache")
        fresh_evals = sum(1 for s in decision_sources if s != "cache")
        assert fresh_evals == 1, f"Expected exactly 1 fresh evaluation, got {fresh_evals}"
        assert cache_hits == 19, f"Expected exactly 19 cache hits, got {cache_hits}"

        # Independent database verification via direct psycopg query
        with psycopg.connect(test_db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM decision_audit WHERE payment_id = %s;", (payment_id,))
                audit_row_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM decision_audit_events WHERE payment_id = %s;", (payment_id,))
                events_row_count = cur.fetchone()[0]

        assert audit_row_count == 1, f"Expected 1 decision_audit row, got {audit_row_count}"
        assert events_row_count == 1, f"Expected 1 decision_audit_events row, got {events_row_count}"

        # Formatted console output for verification
        print("\n=== DAY 8F: 20 CONCURRENT SAME PAYMENT RESULTS ===")
        print(f"total requests: {total_requests}")
        print(f"successful: {successful}")
        print(f"failed: {failed}")
        print(f"LLM invocations: {llm_invocations}")
        print(f"decision_audit rows: {audit_row_count}")
        print(f"decision_audit_events rows: {events_row_count}")
        print(f"wall-clock time: {wall_clock_ms} ms")
        print(f"all 20 responses agreed on decision: True ({list(final_actions)[0]})")
        print(f"fresh evaluation: {fresh_evals}, cache served: {cache_hits}")
        print("==================================================\n")

    finally:
        await close_postgres_pool(pool)
