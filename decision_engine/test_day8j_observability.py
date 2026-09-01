"""
decision_engine/test_day8j_observability.py
===========================================
Day 8J: PostgreSQL Structured Observability Tests.

Validates the 7 structured PostgreSQL database events:
1. db_connection_acquired
2. db_transaction_started
3. db_transaction_committed
4. db_transaction_rolled_back
5. db_cache_hit
6. db_cache_miss
7. db_persistence_failed

Also validates:
- Every event contains request_id and service="python-decision-engine"
- Transaction lifecycle order on success and failure
- Cache hit / miss semantics
- Zero credential, secret, or SQL parameter leakage in logs
- Zero Azure tokens consumed (deterministic mocks)
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


async def init_obs_test_app():
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
    app.state.migrations_applied = True
    app.state.dataset = _get_dataset().copy()

    return app, repo, pool


def extract_structured_logs(caplog: pytest.LogCaptureFixture, req_id: str) -> list[dict[str, Any]]:
    logs = []
    for record in caplog.records:
        try:
            payload = json.loads(record.getMessage())
            if payload.get("request_id") == req_id:
                logs.append(payload)
        except Exception:
            pass
    return logs


# ── Test 1: Successful Evaluation Database Lifecycle ─────────────────────────

@pytest.mark.asyncio
async def test_db_observability_successful_evaluation(caplog: pytest.LogCaptureFixture):
    """
    Test 1: Normal cache-miss evaluation emits:
    - db_cache_miss
    - db_connection_acquired
    - db_transaction_started
    - db_transaction_committed
    All with request_id and service='python-decision-engine'.
    """
    caplog.set_level(logging.INFO)
    app_instance, repo, pool = await init_obs_test_app()
    pid = "pay_8j_success_001"
    req_id = "req-8j-success-obs-001"
    ensure_test_payment(app_instance, pid)

    try:
        caplog.clear()
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/evaluate",
                headers={"X-Request-Id": req_id},
                json={"payment_id": pid, "force_recompute": True},
            )
            assert resp.status_code == 200

        logs = extract_structured_logs(caplog, req_id)
        events = [item["event"] for item in logs]

        assert "db_cache_miss" in events
        assert "db_connection_acquired" in events
        assert "db_transaction_started" in events
        assert "db_transaction_committed" in events
        assert "db_transaction_rolled_back" not in events
        assert "db_persistence_failed" not in events

        # Verify event schema and correlation
        for entry in logs:
            if entry["event"].startswith("db_"):
                assert entry["service"] == "python-decision-engine"
                assert entry["request_id"] == req_id
                assert "timestamp" in entry

        # Verify transaction lifecycle ordering
        idx_conn = events.index("db_connection_acquired")
        idx_tx_start = events.index("db_transaction_started")
        idx_tx_commit = events.index("db_transaction_committed")
        assert idx_conn <= idx_tx_start < idx_tx_commit
    finally:
        await close_postgres_pool(pool)


# ── Test 2: Cache Hit Database Lifecycle ──────────────────────────────────────

@pytest.mark.asyncio
async def test_db_observability_cache_hit(caplog: pytest.LogCaptureFixture):
    """
    Test 2: Second evaluation of the same unchanged payment state emits:
    - db_connection_acquired (during get_current_decision cache lookup)
    - db_cache_hit
    Does NOT emit db_transaction_started or db_transaction_committed.
    """
    caplog.set_level(logging.INFO)
    app_instance, repo, pool = await init_obs_test_app()
    pid = "pay_8j_cache_001"
    req_id_1 = "req-8j-cache-prime-001"
    req_id_2 = "req-8j-cache-hit-002"
    ensure_test_payment(app_instance, pid)

    try:
        # Prime cache
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp1 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": req_id_1},
                json={"payment_id": pid, "force_recompute": False},
            )
            assert resp1.status_code == 200

            # Second request -> cache hit
            caplog.clear()
            resp2 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": req_id_2},
                json={"payment_id": pid, "force_recompute": False},
            )
            assert resp2.status_code == 200
            assert resp2.json()["decision_source"] == "cache"

        logs = extract_structured_logs(caplog, req_id_2)
        events = [item["event"] for item in logs]

        assert "db_cache_hit" in events
        assert "db_connection_acquired" in events
        # No transaction write events on cache hit
        assert "db_transaction_started" not in events
        assert "db_transaction_committed" not in events
        assert "db_transaction_rolled_back" not in events
        assert "db_persistence_failed" not in events
    finally:
        await close_postgres_pool(pool)


# ── Test 3: Transaction Rollback and Persistence Failure Lifecycle ────────────

@pytest.mark.asyncio
async def test_db_observability_transaction_rollback_on_write_failure(caplog: pytest.LogCaptureFixture):
    """
    Test 3: When a write fails inside a PostgreSQL transaction:
    - db_connection_acquired is emitted
    - db_transaction_started is emitted
    - db_transaction_rolled_back is emitted
    - db_persistence_failed is emitted
    - db_transaction_committed is NOT emitted
    All with the request_id.
    """
    caplog.set_level(logging.INFO)
    app_instance, repo, pool = await init_obs_test_app()
    pid = "pay_8j_rollback_001"
    req_id = "req-8j-tx-rollback-001"
    ensure_test_payment(app_instance, pid)

    try:
        # Patch cursor execute during second write (or first write) to simulate DB error inside transaction
        real_get_conn = repo._get_connection

        class FailingCursorWrapper:
            def __init__(self, real_cur):
                self._real_cur = real_cur

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def execute(self, sql, params=None):
                if "decision_audit_events" in str(sql):
                    raise psycopg.OperationalError("simulated deadlock / write failure")
                return await self._real_cur.execute(sql, params)

            def __getattr__(self, name):
                return getattr(self._real_cur, name)

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def failing_get_conn(request_id=None):
            async with real_get_conn(request_id=request_id) as conn:
                real_cursor = conn.cursor

                @asynccontextmanager
                async def failing_cursor(*args, **kwargs):
                    async with real_cursor(*args, **kwargs) as cur:
                        yield FailingCursorWrapper(cur)

                conn.cursor = failing_cursor
                yield conn

        with patch.object(repo, "_get_connection", side_effect=failing_get_conn):
            caplog.clear()
            transport = ASGITransport(app=app_instance, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/evaluate",
                    headers={"X-Request-Id": req_id},
                    json={"payment_id": pid, "force_recompute": True},
                )
                assert resp.status_code == 500

        logs = extract_structured_logs(caplog, req_id)
        events = [item["event"] for item in logs]

        assert "db_connection_acquired" in events
        assert "db_transaction_started" in events
        assert "db_transaction_rolled_back" in events
        assert "db_persistence_failed" in events
        assert "db_transaction_committed" not in events

        # Verify ordering: connection -> tx_start -> tx_rollback -> persistence_failed
        idx_conn = events.index("db_connection_acquired")
        idx_tx_start = events.index("db_transaction_started")
        idx_tx_rb = events.index("db_transaction_rolled_back")
        idx_fail = events.index("db_persistence_failed")
        assert idx_conn <= idx_tx_start < idx_tx_rb <= idx_fail

        rb_event = next(item for item in logs if item["event"] == "db_transaction_rolled_back")
        assert rb_event["error_type"] == "OperationalError"
    finally:
        await close_postgres_pool(pool)


# ── Test 4: Credential and Sensitive Data Log Scan ────────────────────────────

@pytest.mark.asyncio
async def test_db_observability_security_scan(caplog: pytest.LogCaptureFixture):
    """
    Test 4: Rigorous security scan across all emitted log records.
    Confirms zero presence of:
    - DATABASE_URL or TEST_DATABASE_URL
    - Passwords, connection strings, credentials
    - Full SQL statements with parameter values
    - Bearer tokens, API keys
    """
    caplog.set_level(logging.DEBUG)
    app_instance, repo, pool = await init_obs_test_app()
    pid = "pay_8j_sec_001"
    req_id = "req-8j-sec-scan-001"
    ensure_test_payment(app_instance, pid)

    try:
        caplog.clear()
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/evaluate",
                headers={"X-Request-Id": req_id},
                json={"payment_id": pid, "force_recompute": True},
            )
            assert resp.status_code == 200

        all_logs_text = caplog.text.lower()
        test_db_url = os.getenv("TEST_DATABASE_URL", "")

        # 1. Prohibited connection URLs and credentials
        assert "postgresql://" not in all_logs_text
        assert "postgres://" not in all_logs_text
        assert "password" not in all_logs_text
        assert "secret" not in all_logs_text
        assert "bearer" not in all_logs_text
        assert "accountkey" not in all_logs_text

        if test_db_url:
            # Masked check
            assert test_db_url.lower() not in all_logs_text

        # 2. Verify all DB events are valid JSON with required contract fields
        logs = extract_structured_logs(caplog, req_id)
        db_logs = [item for item in logs if item["event"].startswith("db_")]
        assert len(db_logs) >= 3

        for item in db_logs:
            assert "request_id" in item
            assert item["request_id"] == req_id
            assert item["service"] == "python-decision-engine"
            assert "timestamp" in item
            # Ensure no SQL string or query parameter payload leaked into log attributes
            for key, val in item.items():
                val_str = str(val).lower()
                assert "select " not in val_str
                assert "insert into" not in val_str
                assert "postgresql://" not in val_str

        print("\n=== DAY 8J SECURITY LOG SCAN RESULT ===")
        print(f"Scanned {len(caplog.records)} total log records.")
        print("Prohibited secrets/URLs detected: 0")
        print("SQL statements with parameters detected: 0")
        print("Valid structured DB events found: True")
        print("=======================================\n")
    finally:
        await close_postgres_pool(pool)
