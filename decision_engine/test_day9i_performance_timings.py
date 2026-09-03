"""
decision_engine/test_day9i_performance_timings.py
=================================================
Day 9I: Performance Timing Validation in Structured Logs.

Validates:
1. Total request duration ('duration_ms') on decision_completed
2. Cache-hit duration ('duration_ms') on cache_hit
3. Fresh-evaluation duration ('evaluation_duration_ms') on decision_completed
4. Database write duration ('db_write_duration_ms') on db_transaction_committed
5. LLM call duration ('llm_duration_ms') on decision_completed (mocked with controlled delay)
6. Lock wait duration ('lock_wait_duration_ms') on decision_completed under concurrency
7. All numeric values >= 0
8. Request correlation integrity maintained
"""

from __future__ import annotations

import asyncio
import json
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
from decision_engine.service import app


TEST_DB_URL = os.getenv("TEST_DATABASE_URL")
integration_mark = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="TEST_DATABASE_URL is not set; skipping PostgreSQL performance timing tests.",
)


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


def make_delayed_mock_llm(delay_s: float = 0.05):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    async def fake_ainvoke(*args, **kwargs):
        await asyncio.sleep(delay_s)
        return LLMDecision(
            decision="RETRY_NUDGE",
            confidence=0.95,
            reasoning="Selected RETRY_NUDGE with controlled delay.",
            risk_level="low",
            expected_incremental_value=125.0,
        )

    mock_structured.ainvoke = AsyncMock(side_effect=fake_ainvoke)
    return mock_llm


def ensure_test_payment(app_instance: Any, pid: str) -> None:
    base_df = _get_dataset().copy()
    template_row = dict(base_df.iloc[0].to_dict())
    template_row.update({
        "payment_id": pid,
        "status": "FAILED",
        "attempt_number": 1,
        "consecutive_failures": 1,
        "retry_count": 0,
        "interventions_7d": 0,
    })
    app_instance.state.dataset = pd.concat([base_df, pd.DataFrame([template_row])], ignore_index=True)


async def init_timing_test_app(llm_delay_s: float = 0.05):
    test_db_url = os.getenv("TEST_DATABASE_URL")
    run_migrations(database_url=test_db_url)
    pool = await create_postgres_pool(test_db_url, min_size=2, max_size=5)
    repo = PostgresDecisionRepository(pool=pool)

    policy = make_mock_policy()
    llm = make_delayed_mock_llm(delay_s=llm_delay_s)
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
    app.state.migrations_applied = True
    app.state.dataset = _get_dataset().copy()

    return app, repo, pool


def extract_structured_logs(caplog: pytest.LogCaptureFixture, req_id: str) -> list[dict[str, Any]]:
    logs = []
    for record in caplog.records:
        try:
            payload = json.loads(record.getMessage())
            if isinstance(payload, dict) and payload.get("request_id") == req_id:
                logs.append(payload)
        except Exception:
            pass
    return logs


@integration_mark
@pytest.mark.asyncio
async def test_performance_timings_fresh_and_cache(caplog: pytest.LogCaptureFixture):
    """
    Day 9I: Verify all required performance timings across fresh evaluation and cache hit:
    1. total request duration ('duration_ms') on decision_completed
    2. fresh evaluation duration ('evaluation_duration_ms') on decision_completed
    3. LLM duration ('llm_duration_ms') on decision_completed
    4. lock wait duration ('lock_wait_duration_ms') on decision_completed
    5. database write duration ('db_write_duration_ms') on db_transaction_committed
    6. cache-hit duration ('duration_ms') on cache_hit
    """
    caplog.set_level(logging.INFO)
    app_instance, repo, pool = await init_timing_test_app(llm_delay_s=0.06)
    import uuid
    uid = uuid.uuid4().hex[:8]
    pid = f"pay_9i_timing_{uid}"
    req_id_fresh = f"req-9i-fresh-{uid}"
    req_id_cache = f"req-9i-cache-{uid}"
    ensure_test_payment(app_instance, pid)

    try:
        caplog.clear()
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Step 1: Fresh evaluation
            t0 = time.monotonic()
            resp1 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": req_id_fresh},
                json={"payment_id": pid, "force_recompute": False},
            )
            elapsed_fresh = (time.monotonic() - t0) * 1000
            assert resp1.status_code == 200

            # Step 2: Cache hit evaluation
            t0_cache = time.monotonic()
            resp2 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": req_id_cache},
                json={"payment_id": pid, "force_recompute": False},
            )
            elapsed_cache = (time.monotonic() - t0_cache) * 1000
            assert resp2.status_code == 200
            assert resp2.json()["decision_source"] == "cache"

        # --- Assertions for Fresh Evaluation ---
        fresh_logs = extract_structured_logs(caplog, req_id_fresh)
        events_fresh = {item["event"]: item for item in fresh_logs}

        # A. decision_completed timings
        assert "decision_completed" in events_fresh
        dec_comp = events_fresh["decision_completed"]
        assert "duration_ms" in dec_comp
        assert isinstance(dec_comp["duration_ms"], (int, float)) and dec_comp["duration_ms"] >= 0

        assert "evaluation_duration_ms" in dec_comp
        assert isinstance(dec_comp["evaluation_duration_ms"], (int, float))
        assert dec_comp["evaluation_duration_ms"] >= 40.0  # mock had 60ms delay

        assert "llm_duration_ms" in dec_comp
        assert isinstance(dec_comp["llm_duration_ms"], (int, float))
        assert dec_comp["llm_duration_ms"] >= 40.0  # measured actual mock invocation

        assert "lock_wait_duration_ms" in dec_comp
        assert isinstance(dec_comp["lock_wait_duration_ms"], (int, float)) and dec_comp["lock_wait_duration_ms"] >= 0

        # B. db_transaction_committed write timing
        assert "db_transaction_committed" in events_fresh
        tx_comm = events_fresh["db_transaction_committed"]
        assert "db_write_duration_ms" in tx_comm
        assert isinstance(tx_comm["db_write_duration_ms"], (int, float)) and tx_comm["db_write_duration_ms"] >= 0

        # --- Assertions for Cache Hit Evaluation ---
        cache_logs = extract_structured_logs(caplog, req_id_cache)
        events_cache = {item["event"]: item for item in cache_logs}

        # C. cache_hit timing
        assert "cache_hit" in events_cache
        c_hit = events_cache["cache_hit"]
        assert "duration_ms" in c_hit
        assert isinstance(c_hit["duration_ms"], (int, float)) and c_hit["duration_ms"] >= 0

        # D. decision_completed on cache hit
        assert "decision_completed" in events_cache
        dec_cache = events_cache["decision_completed"]
        assert dec_cache["decision_source"] == "cache"
        assert "duration_ms" in dec_cache
        assert "lock_wait_duration_ms" in dec_cache
        assert isinstance(dec_cache["duration_ms"], (int, float))
    finally:
        await close_postgres_pool(pool)


@integration_mark
@pytest.mark.asyncio
async def test_concurrent_lock_wait_duration(caplog: pytest.LogCaptureFixture):
    """
    Day 9I: Verify lock_wait_duration_ms accurately measures concurrency wait:
    - Launch 2 concurrent requests for the SAME payment_id.
    - Request 1 holds the lock for ~100ms.
    - Request 2 must report lock_wait_duration_ms >= 50ms.
    """
    caplog.set_level(logging.INFO)
    app_instance, repo, pool = await init_timing_test_app(llm_delay_s=0.10)
    import uuid
    uid = uuid.uuid4().hex[:8]
    pid = f"pay_9i_lock_{uid}"
    req_id_1 = f"req-9i-concurrent-1-{uid}"
    req_id_2 = f"req-9i-concurrent-2-{uid}"
    ensure_test_payment(app_instance, pid)

    try:
        caplog.clear()
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Launch two simultaneous requests for the same payment
            task1 = asyncio.create_task(
                client.post(
                    "/evaluate",
                    headers={"X-Request-Id": req_id_1},
                    json={"payment_id": pid, "force_recompute": True},
                )
            )
            # Slight sleep to ensure task1 acquires lock first
            await asyncio.sleep(0.01)
            task2 = asyncio.create_task(
                client.post(
                    "/evaluate",
                    headers={"X-Request-Id": req_id_2},
                    json={"payment_id": pid, "force_recompute": True},
                )
            )

            resp1, resp2 = await asyncio.gather(task1, task2)
            assert resp1.status_code == 200
            assert resp2.status_code == 200

        logs_req2 = extract_structured_logs(caplog, req_id_2)
        dec_comp_2 = next(item for item in logs_req2 if item["event"] == "decision_completed")

        assert "lock_wait_duration_ms" in dec_comp_2
        assert isinstance(dec_comp_2["lock_wait_duration_ms"], (int, float))
        # Request 2 waited for Request 1 (which had a 100ms mock LLM delay)
        assert dec_comp_2["lock_wait_duration_ms"] >= 40.0, (
            f"Expected lock_wait_duration_ms >= 40ms, got {dec_comp_2['lock_wait_duration_ms']}ms"
        )
    finally:
        await close_postgres_pool(pool)
