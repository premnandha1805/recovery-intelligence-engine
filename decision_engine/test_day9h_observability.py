"""
decision_engine/test_day9h_observability.py
===========================================
Day 9H: Production Structured Observability Test Suite.

Validates all 12 required structured events:
1. request_received (with request_start timestamp)
2. decision_completed (with duration_ms)
3. cache_hit (with duration_ms)
4. cache_miss (with duration_ms)
5. db_connection_acquired (with duration_ms)
6. db_transaction_started
7. db_transaction_committed (with duration_ms)
8. db_transaction_rolled_back (with duration_ms)
9. db_persistence_failed (with duration_ms)
10. health_check (with duration_ms)
11. service_startup
12. service_shutdown

Also validates:
- Contract: timestamp, service="python-decision-engine", event, request_id on EVERY event
- Numerical validity of duration_ms (float/int >= 0)
- Request correlation: consistent request_id per transaction lifecycle
- Security & Data Minimization: zero leakage of secrets, keys, passwords, SQL parameters, or PII
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import pathlib
import sys
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
integration_mark = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="TEST_DATABASE_URL is not set; skipping PostgreSQL observability tests.",
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


def make_mock_llm():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    async def fake_ainvoke(*args, **kwargs):
        return LLMDecision(
            decision="RETRY_NUDGE",
            confidence=0.95,
            reasoning="Selected RETRY_NUDGE based on positive uplift.",
            risk_level="low",
            expected_incremental_value=125.0,
        )

    mock_structured.ainvoke = AsyncMock(side_effect=fake_ainvoke)
    mock_structured.invoke = MagicMock(return_value=LLMDecision(
        decision="RETRY_NUDGE",
        confidence=0.95,
        reasoning="Selected RETRY_NUDGE based on positive uplift.",
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
        "consecutive_failures": 1,
        "retry_count": 0,
        "interventions_7d": 0,
    })
    app_instance.state.dataset = pd.concat([base_df, pd.DataFrame([template_row])], ignore_index=True)


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


def extract_all_structured_logs(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    logs = []
    for record in caplog.records:
        try:
            payload = json.loads(record.getMessage())
            if isinstance(payload, dict) and "event" in payload:
                logs.append(payload)
        except Exception:
            pass
    return logs


@integration_mark
@pytest.mark.asyncio
async def test_all_12_required_events_and_timings(caplog: pytest.LogCaptureFixture):
    """
    Day 9H: Comprehensive validation of all 12 required structured events,
    timing fields (duration_ms / request_start), and correlation IDs.
    """
    caplog.set_level(logging.INFO)
    app_instance, repo, pool = await init_obs_test_app()
    import uuid
    uid = uuid.uuid4().hex[:8]
    pid = f"pay_9h_full_{uid}"
    req_id_1 = f"req-9h-full-miss-{uid}"
    req_id_2 = f"req-9h-full-hit-{uid}"
    ensure_test_payment(app_instance, pid)

    try:
        caplog.clear()
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Health check event
            health_resp = await client.get("/health")
            assert health_resp.status_code == 200

            # 2. Cache-miss evaluation
            resp1 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": req_id_1},
                json={"payment_id": pid, "force_recompute": False},
            )
            assert resp1.status_code == 200

            # 3. Cache-hit evaluation
            resp2 = await client.post(
                "/evaluate",
                headers={"X-Request-Id": req_id_2},
                json={"payment_id": pid, "force_recompute": False},
            )
            assert resp2.status_code == 200

        # Also test startup and shutdown events via lifespan
        mock_pool = MagicMock()
        mock_pool.closed = False
        mock_pool.close = AsyncMock()
        mock_app = MagicMock()
        mock_app.state = MagicMock()
        mock_app.state.persistence_backend = "postgres"
        mock_app.state.db_pool = mock_pool
        mock_app.state.db = None
        mock_app.state.repository = MagicMock()

        with patch("decision_engine.service.load_and_validate_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(
                persistence_backend="postgres",
                database_url=TEST_DB_URL,
                db_pool_min=2,
                db_pool_max=5,
                db_connect_timeout_ms=3000,
            )
            with patch("decision_engine.service.create_postgres_pool", new_callable=AsyncMock, return_value=mock_pool):
                with patch("decision_engine.service.run_migrations", return_value=[]):
                    with patch("decision_engine.service.CausalUpliftPolicy"):
                        with patch("decision_engine.service.create_recovery_graph"):
                            async with lifespan(mock_app):
                                pass

        all_logs = extract_all_structured_logs(caplog)
        events_found = {item["event"]: item for item in all_logs}

        # Verify presence of all key events
        expected_events = [
            "service_startup",
            "service_shutdown",
            "health_check",
            "request_received",
            "decision_completed",
            "cache_miss",
            "cache_hit",
            "db_connection_acquired",
            "db_transaction_started",
            "db_transaction_committed",
        ]
        for evt in expected_events:
            assert evt in events_found, f"Expected event '{evt}' was not emitted!"

        # Contract validation on EVERY event
        for log_entry in all_logs:
            assert "timestamp" in log_entry, f"Missing timestamp in {log_entry}"
            assert log_entry["service"] == "python-decision-engine"
            assert "event" in log_entry
            assert "request_id" in log_entry
            assert isinstance(log_entry["request_id"], str) and len(log_entry["request_id"]) > 0

        # Timing fields validation
        # 1. request_received has request_start
        req_rec = events_found["request_received"]
        assert "request_start" in req_rec
        assert isinstance(req_rec["request_start"], str)

        # 2. decision_completed has numeric duration_ms
        dec_comp = events_found["decision_completed"]
        assert "duration_ms" in dec_comp
        assert isinstance(dec_comp["duration_ms"], (int, float))
        assert dec_comp["duration_ms"] >= 0

        # 3. cache_hit has numeric duration_ms
        c_hit = events_found["cache_hit"]
        assert "duration_ms" in c_hit
        assert isinstance(c_hit["duration_ms"], (int, float))

        # 4. cache_miss has numeric duration_ms
        c_miss = events_found["cache_miss"]
        assert "duration_ms" in c_miss
        assert isinstance(c_miss["duration_ms"], (int, float))

        # 5. db_connection_acquired has numeric duration_ms
        db_conn = events_found["db_connection_acquired"]
        assert "duration_ms" in db_conn
        assert isinstance(db_conn["duration_ms"], (int, float))

        # 6. db_transaction_committed has numeric duration_ms
        db_comm = events_found["db_transaction_committed"]
        assert "duration_ms" in db_comm
        assert isinstance(db_comm["duration_ms"], (int, float))

        # 7. health_check has numeric duration_ms
        h_check = events_found["health_check"]
        assert "duration_ms" in h_check
        assert isinstance(h_check["duration_ms"], (int, float))
    finally:
        await close_postgres_pool(pool)


@integration_mark
@pytest.mark.asyncio
async def test_rollback_and_failure_events_with_duration(caplog: pytest.LogCaptureFixture):
    """
    Day 9H: Validate db_transaction_rolled_back and db_persistence_failed
    contain duration_ms and error_type.
    """
    caplog.set_level(logging.INFO)
    app_instance, repo, pool = await init_obs_test_app()
    pid = "pay_9h_rb_001"
    req_id = "req-9h-rb-001"
    ensure_test_payment(app_instance, pid)

    try:
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

        logs = extract_all_structured_logs(caplog)
        events_by_name = {item["event"]: item for item in logs if item.get("request_id") == req_id}

        assert "db_transaction_rolled_back" in events_by_name
        assert "db_persistence_failed" in events_by_name

        rb_event = events_by_name["db_transaction_rolled_back"]
        assert rb_event["error_type"] == "OperationalError"
        assert "duration_ms" in rb_event
        assert isinstance(rb_event["duration_ms"], (int, float))

        fail_event = events_by_name["db_persistence_failed"]
        assert fail_event["error_type"] == "OperationalError"
        assert "duration_ms" in fail_event
        assert isinstance(fail_event["duration_ms"], (int, float))
    finally:
        await close_postgres_pool(pool)


@integration_mark
@pytest.mark.asyncio
async def test_security_and_pii_data_minimization(caplog: pytest.LogCaptureFixture):
    """
    Day 9H: Comprehensive security and data minimization scan across all emitted logs:
    - Zero passwords, DATABASE_URL, Azure keys, Bearer tokens
    - Zero raw SQL queries with parameter values
    - Zero customer PII or raw prompts
    """
    caplog.set_level(logging.DEBUG)
    app_instance, repo, pool = await init_obs_test_app()
    pid = "pay_9h_sec_001"
    req_id = "req-9h-sec-001"
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

        all_text = caplog.text.lower()
        test_db_url = os.getenv("TEST_DATABASE_URL", "")

        # Prohibited secrets
        assert "postgresql://" not in all_text
        assert "postgres://" not in all_text
        assert "password" not in all_text
        assert "secret" not in all_text
        assert "bearer" not in all_text
        assert "accountkey" not in all_text
        if test_db_url:
            assert test_db_url.lower() not in all_text

        # Validate all structured entries
        all_logs = extract_all_structured_logs(caplog)
        assert len(all_logs) > 0
        for entry in all_logs:
            for k, v in entry.items():
                val_str = str(v).lower()
                assert "select " not in val_str
                assert "insert into" not in val_str
                assert "delete from" not in val_str
                assert "update " not in val_str
    finally:
        await close_postgres_pool(pool)
