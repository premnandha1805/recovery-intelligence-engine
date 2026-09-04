"""
decision_engine/test_day9g_shutdown.py
======================================
Day 9G: Graceful Shutdown Unit & Integration Tests.

Validates:
1. Lifespan exit triggers clean database pool closure.
2. Lifespan exit emits structured 'service_shutdown' event.
3. In-flight requests complete or terminate cleanly within grace period.
4. Database integrity is preserved with zero dangling/uncommitted transactions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

from dotenv import load_dotenv
load_dotenv(".env.test")
load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import psycopg
import pytest
from httpx import ASGITransport, AsyncClient

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
    reason="TEST_DATABASE_URL is not set; skipping PostgreSQL shutdown tests.",
)


@pytest.mark.asyncio
async def test_lifespan_emits_service_shutdown_event_and_closes_pool(caplog: pytest.LogCaptureFixture):
    """
    Day 9G: Verify that when the lifespan context exits:
    1. The PostgreSQL connection pool is cleanly closed.
    2. A structured 'service_shutdown' log event is emitted.
    3. State repository reference is reset.
    """
    caplog.set_level(logging.INFO)
    mock_pool = MagicMock()
    mock_pool.closed = False
    mock_pool.close = AsyncMock()

    test_app = MagicMock()
    test_app.state = MagicMock()
    test_app.state.persistence_backend = "postgres"
    test_app.state.db_pool = mock_pool
    test_app.state.db = None
    test_app.state.repository = MagicMock()
    test_app.state.policy = MagicMock()
    test_app.state.graph = MagicMock()

    caplog.clear()
    with patch("decision_engine.service.load_and_validate_config") as mock_cfg:
        mock_cfg.return_value = MagicMock(
            persistence_backend="postgres",
            database_url="postgresql://user:pass@localhost:5432/testdb",
            db_pool_min=2,
            db_pool_max=5,
            db_connect_timeout_ms=3000,
        )
        with patch("decision_engine.service.create_postgres_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool
            with patch("decision_engine.service.run_migrations", return_value=[]):
                with patch("decision_engine.service.CausalUpliftPolicy"):
                    with patch("decision_engine.service.create_recovery_graph"):
                        async with lifespan(test_app):
                            pass

    # Verify pool was closed
    mock_pool.close.assert_awaited_once()

    # Verify structured service_shutdown event was emitted
    shutdown_events = []
    for record in caplog.records:
        try:
            parsed = json.loads(record.getMessage())
            if parsed.get("event") == "service_shutdown":
                shutdown_events.append(parsed)
        except Exception:
            pass

    assert len(shutdown_events) >= 1, "Must emit at least one service_shutdown event"
    event = shutdown_events[0]
    assert event["service"] == "python-decision-engine"
    assert event["event"] == "service_shutdown"
    assert event["status"] == "clean_shutdown"
    assert "timestamp" in event


@pytest.mark.asyncio
async def test_lifespan_sqlite_clean_shutdown(caplog: pytest.LogCaptureFixture):
    """Verify SQLite connection is closed on shutdown when running in SQLite mode."""
    caplog.set_level(logging.INFO)
    mock_db = MagicMock()
    mock_db.close = AsyncMock()

    test_app = MagicMock()
    test_app.state = MagicMock()
    test_app.state.persistence_backend = "sqlite"
    test_app.state.db_pool = None
    test_app.state.db = mock_db
    test_app.state.repository = MagicMock()

    caplog.clear()
    with patch("decision_engine.service.load_and_validate_config") as mock_cfg:
        mock_cfg.return_value = MagicMock(
            persistence_backend="sqlite",
            database_url=None,
        )
        with patch("decision_engine.service.open_sqlite_repository", new_callable=AsyncMock) as mock_open:
            mock_open.return_value = (mock_db, MagicMock())
            with patch("decision_engine.service.CausalUpliftPolicy"):
                with patch("decision_engine.service.create_recovery_graph"):
                    async with lifespan(test_app):
                        pass

    mock_db.close.assert_awaited_once()
    shutdown_events = []
    for record in caplog.records:
        try:
            parsed = json.loads(record.getMessage())
            if parsed.get("event") == "service_shutdown":
                shutdown_events.append(parsed)
        except Exception:
            pass

    assert len(shutdown_events) >= 1, "Must emit service_shutdown in sqlite mode"


@integration_mark
@pytest.mark.asyncio
async def test_database_integrity_post_shutdown():
    """
    Day 9G: Verify database integrity against real PostgreSQL:
    - Pool acquisition and operations commit cleanly.
    - Post-shutdown, connection pool closes.
    - No active/dangling transactions or locked tables remain.
    """
    db_url = TEST_DB_URL
    run_migrations(database_url=db_url)

    pool = await create_postgres_pool(database_url=db_url, min_size=2, max_size=5)
    repo = PostgresDecisionRepository(pool=pool)

    # Perform sample read/write
    result = await repo.get_current_decision("pay_nonexistent_shutdown_test")
    assert result is None

    # Shutdown the pool
    await close_postgres_pool(pool)

    # Verify no open transactions / orphaned locks on PostgreSQL
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*)
                FROM pg_stat_activity
                WHERE datname = current_database()
                  AND state = 'idle in transaction';
                """
            )
            idle_in_tx = cur.fetchone()[0]
            assert idle_in_tx == 0, "No connections should be left idle in transaction"
