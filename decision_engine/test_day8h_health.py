"""
decision_engine/test_day8h_health.py
====================================
Day 8H: PostgreSQL Health / Readiness Check Tests.

Verifies:
- Test A: PostgreSQL healthy → database=ok
- Test B: PostgreSQL unavailable → database=unavailable, status=degraded
- Test C: Health probe uses short ~2s timeout, not 8s decision timeout
- Test D: No credentials, stack traces, or DATABASE_URL leakage
- Test E: SQLite mode still works (no PostgreSQL dependency required)
- Test F: NestJS forwards Python database degradation correctly (architecture check)
- Test G: PostgreSQL recovery returns database=ok again
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from dotenv import load_dotenv
load_dotenv(".env.test")
load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pandas as pd
import psycopg
import pytest
from httpx import ASGITransport, AsyncClient

from decision_engine.context_node import _get_dataset
from decision_engine.graph import create_recovery_graph
from decision_engine.persistence import (
    PostgresDecisionRepository,
    SqliteDecisionRepository,
    create_postgres_pool,
    close_postgres_pool,
    check_pool_health,
    HEALTH_PROBE_TIMEOUT_S,
)
from decision_engine.persistence.migrate import run_migrations
from decision_engine.reasoning_node import LLMDecision
from decision_engine.service import app, lifespan


TEST_DB_URL = os.getenv("TEST_DATABASE_URL")


# ── Mock Helpers ─────────────────────────────────────────────────────────────

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


async def init_health_test_app():
    """Initialize FastAPI service wired to real PostgreSQL test pool and mock policy/LLM."""
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


# ── Test A: PostgreSQL healthy → database=ok ─────────────────────────────────

@pytest.mark.asyncio
async def test_day8h_test_a_postgres_healthy():
    """
    Test A: When PostgreSQL is healthy, /health returns:
    - HTTP 200
    - status = "ok"
    - dependencies.database = "ok"
    - dependencies.decision_engine = "ok"
    """
    app_instance, repo, pool = await init_health_test_app()
    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["dependencies"]["database"] == "ok"
            assert data["dependencies"]["decision_engine"] == "ok"
    finally:
        await close_postgres_pool(pool)


# ── Test B: PostgreSQL unavailable → database=unavailable ────────────────────

@pytest.mark.asyncio
async def test_day8h_test_b_postgres_unavailable():
    """
    Test B: When PostgreSQL pool is closed/unavailable, /health returns:
    - HTTP 200 (engine itself is initialized)
    - status = "degraded"
    - dependencies.database = "unavailable"
    - dependencies.decision_engine = "ok"
    """
    app_instance, repo, pool = await init_health_test_app()
    try:
        # Close the pool to simulate database unavailability
        await close_postgres_pool(pool)

        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "degraded"
            assert data["dependencies"]["database"] == "unavailable"
            assert data["dependencies"]["decision_engine"] == "ok"
    finally:
        pass  # Pool already closed


# ── Test C: Health probe uses short timeout ──────────────────────────────────

@pytest.mark.asyncio
async def test_day8h_test_c_short_timeout():
    """
    Test C: Health probe timeout is ~2 seconds, not the 8-second decision timeout.
    Verify that:
    - HEALTH_PROBE_TIMEOUT_S == 2.0
    - When the pool connection hangs, the probe completes within ~3s (2s timeout + overhead)
    - The probe does NOT wait for 8 seconds
    """
    # Verify configured constant
    assert HEALTH_PROBE_TIMEOUT_S == 2.0

    # Simulate a hanging pool connection
    mock_pool = MagicMock()
    mock_pool.closed = False

    async def hanging_connection(*args, **kwargs):
        await asyncio.sleep(30)  # Hang longer than any timeout

    # Make pool.connection() return a context manager that hangs
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(side_effect=hanging_connection)
    mock_cm.__aexit__ = AsyncMock()
    mock_pool.connection.return_value = mock_cm

    t0 = time.monotonic()
    result = await check_pool_health(mock_pool, timeout_s=2.0)
    elapsed = time.monotonic() - t0

    assert result == "unavailable"
    assert elapsed < 4.0, f"Health probe took {elapsed:.2f}s, expected < 4s"
    assert elapsed >= 1.5, f"Health probe completed too quickly ({elapsed:.2f}s), timeout may not be active"

    print(f"\n=== DAY 8H: Health probe timeout test ===")
    print(f"configured timeout: {HEALTH_PROBE_TIMEOUT_S}s")
    print(f"measured duration: {elapsed:.2f}s")
    print(f"8-second decision timeout NOT used: True")
    print(f"==========================================\n")


# ── Test D: No credential/stack trace leakage ────────────────────────────────

@pytest.mark.asyncio
async def test_day8h_test_d_no_leakage():
    """
    Test D: Health responses contain no credentials, DATABASE_URL, passwords,
    stack traces, bearer tokens, API keys, or filesystem paths.
    """
    app_instance, repo, pool = await init_health_test_app()
    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Check healthy response
            resp_ok = await client.get("/health")
            text_ok = resp_ok.text.lower()
            assert "password" not in text_ok
            assert "traceback" not in text_ok
            assert "database_url" not in text_ok
            assert "bearer" not in text_ok
            assert "api_key" not in text_ok
            assert "postgresql://" not in text_ok
            assert "postgres://" not in text_ok

        # Close pool and check degraded response
        await close_postgres_pool(pool)
        pool = None  # prevent double-close in finally

        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp_degraded = await client.get("/health")
            text_degraded = resp_degraded.text.lower()
            assert "password" not in text_degraded
            assert "traceback" not in text_degraded
            assert "database_url" not in text_degraded
            assert "bearer" not in text_degraded
            assert "api_key" not in text_degraded
            assert "postgresql://" not in text_degraded
            assert "postgres://" not in text_degraded
            assert "stack" not in text_degraded
    finally:
        if pool is not None:
            await close_postgres_pool(pool)


# ── Test E: SQLite mode still works ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_day8h_test_e_sqlite_mode(tmp_path: pathlib.Path):
    """
    Test E: When PERSISTENCE_BACKEND=sqlite, /health works without PostgreSQL.
    Returns status=ok with database=ok (no PG dependency to check).
    """
    test_db_path = str(tmp_path / "sqlite_health.db")
    with patch.dict(os.environ, {"PERSISTENCE_BACKEND": "sqlite"}):
        with patch("decision_engine.service.DEFAULT_AUDIT_DB_PATH", test_db_path):
            async with lifespan(app):
                assert app.state.persistence_backend == "sqlite"
                assert isinstance(app.state.repository, SqliteDecisionRepository)

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/health")
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["status"] == "ok"
                    assert data["dependencies"]["database"] == "ok"
                    assert data["dependencies"]["decision_engine"] == "ok"


# ── Test F: NestJS architecture compatibility ────────────────────────────────

@pytest.mark.asyncio
async def test_day8h_test_f_nestjs_compatibility():
    """
    Test F: NestJS compatibility — verify Python health response format.
    NestJS checkHealth() checks `parsed.status === 'ok'`.
    When DB is healthy, Python returns status="ok" → NestJS sees ok → reports decision_engine=ok.
    When DB is down, Python returns status="degraded" → NestJS sees non-ok → reports decision_engine=unreachable.
    No NestJS code change required.
    """
    app_instance, repo, pool = await init_health_test_app()
    try:
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Healthy: NestJS will see status="ok" and report decision_engine=ok
            resp_ok = await client.get("/health")
            data_ok = resp_ok.json()
            assert data_ok["status"] == "ok"  # NestJS: parsed.status === 'ok' → true

        # Close pool → degraded
        await close_postgres_pool(pool)
        pool = None

        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp_degraded = await client.get("/health")
            data_degraded = resp_degraded.json()
            # NestJS: parsed.status === 'ok' → false → reports 'unreachable'
            assert data_degraded["status"] == "degraded"
            assert data_degraded["status"] != "ok"
    finally:
        if pool is not None:
            await close_postgres_pool(pool)


# ── Test G: PostgreSQL recovery ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_day8h_test_g_postgres_recovery():
    """
    Test G: PostgreSQL recovery — after pool is closed, creating a new pool restores health.
    This tests whether the app can recover if the pool is replaced.
    Note: The current fail-fast pool lifecycle requires a new pool creation; it does not
    auto-reconnect a closed pool.
    """
    app_instance, repo, pool = await init_health_test_app()
    test_db_url = os.getenv("TEST_DATABASE_URL")

    try:
        # 1. Verify healthy
        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp1 = await client.get("/health")
            assert resp1.json()["status"] == "ok"
            assert resp1.json()["dependencies"]["database"] == "ok"

        # 2. Close pool → degraded
        await close_postgres_pool(pool)

        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp2 = await client.get("/health")
            assert resp2.json()["status"] == "degraded"
            assert resp2.json()["dependencies"]["database"] == "unavailable"

        # 3. Create new pool → recovered
        new_pool = await create_postgres_pool(test_db_url, min_size=2, max_size=5)
        app_instance.state.db_pool = new_pool
        app_instance.state.repository = PostgresDecisionRepository(pool=new_pool)

        transport = ASGITransport(app=app_instance)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp3 = await client.get("/health")
            assert resp3.json()["status"] == "ok"
            assert resp3.json()["dependencies"]["database"] == "ok"

        await close_postgres_pool(new_pool)
    finally:
        pass  # Pools handled above


# ── Unit tests for check_pool_health ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_pool_health_none_pool():
    """check_pool_health(None) returns 'unavailable'."""
    result = await check_pool_health(None)
    assert result == "unavailable"


@pytest.mark.asyncio
async def test_check_pool_health_closed_pool():
    """check_pool_health on a closed pool returns 'unavailable'."""
    mock_pool = MagicMock()
    mock_pool.closed = True
    result = await check_pool_health(mock_pool)
    assert result == "unavailable"


@pytest.mark.asyncio
async def test_check_pool_health_real_pool():
    """check_pool_health on a real healthy pool returns 'ok'."""
    test_db_url = os.getenv("TEST_DATABASE_URL")
    if not test_db_url:
        pytest.skip("TEST_DATABASE_URL is not set.")

    pool = await create_postgres_pool(test_db_url, min_size=1, max_size=2)
    try:
        result = await check_pool_health(pool)
        assert result == "ok"
    finally:
        await close_postgres_pool(pool)


@pytest.mark.asyncio
async def test_check_pool_health_exception_returns_unavailable():
    """check_pool_health returns 'unavailable' on any exception, never propagates."""
    mock_pool = MagicMock()
    mock_pool.closed = False

    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(side_effect=ConnectionError("simulated connection failure"))
    mock_cm.__aexit__ = AsyncMock()
    mock_pool.connection.return_value = mock_cm

    result = await check_pool_health(mock_pool)
    assert result == "unavailable"
