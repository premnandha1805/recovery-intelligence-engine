"""
decision_engine/persistence/test_pool.py
========================================
Comprehensive test suite for Day 8C PostgreSQL AsyncConnectionPool lifecycle:
- Unit tests: Configuration validation, default values, error handling, password sanitization.
- Integration tests: Pool creation, connection checkout, concurrency, timeout, shutdown,
  PostgresDecisionRepository compatibility, and FastAPI lifespan integration.
"""

from __future__ import annotations

import asyncio
import datetime
import os
import sys
import time
from typing import Any
from unittest.mock import patch
import pytest

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from psycopg_pool import AsyncConnectionPool
from decision_engine.persistence.migrate import run_migrations
from decision_engine.persistence.postgres import (
    PostgresDecisionRepository,
    create_postgres_pool,
    close_postgres_pool,
    get_pool_config,
    sanitize_database_url,
    validate_database_url,
)
from decision_engine.service import app, lifespan
from dotenv import load_dotenv

load_dotenv(".env.test")
load_dotenv()

TEST_DB_URL = os.getenv("TEST_DATABASE_URL")
integration_mark = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="TEST_DATABASE_URL is not set; skipping PostgreSQL pool integration tests",
)


# ===========================================================================
# 1. Configuration & Unit Tests (No Database Required)
# ===========================================================================

def test_1_default_values() -> None:
    """1. Verify default values: min=2, max=12, timeout=3000ms."""
    with patch.dict(os.environ, {"DATABASE_URL": "postgresql://user:pass@localhost:5432/db"}, clear=False):
        # Remove any override env vars
        os.environ.pop("DB_POOL_MIN", None)
        os.environ.pop("DB_POOL_MAX", None)
        os.environ.pop("DB_CONNECT_TIMEOUT_MS", None)

        cfg = get_pool_config()
        assert cfg["min_size"] == 2
        assert cfg["max_size"] == 12
        assert cfg["timeout_ms"] == 3000
        assert cfg["database_url"] == "postgresql://user:pass@localhost:5432/db"


def test_2_invalid_database_url() -> None:
    """2. Reject missing, empty, or wrong-scheme DATABASE_URL."""
    # Missing / None
    with pytest.raises(ValueError, match="DATABASE_URL is required"):
        validate_database_url(None)

    with pytest.raises(ValueError, match="DATABASE_URL is required"):
        validate_database_url("   ")

    # Wrong scheme
    with pytest.raises(ValueError, match="Invalid DATABASE_URL scheme"):
        validate_database_url("mysql://user:pass@localhost/db")

    with pytest.raises(ValueError, match="Invalid DATABASE_URL scheme"):
        validate_database_url("sqlite:///test.db")


def test_3_invalid_db_pool_min() -> None:
    """3. Reject invalid DB_POOL_MIN (< 1 or non-integer)."""
    valid_url = "postgresql://localhost/db"
    with pytest.raises(ValueError, match="DB_POOL_MIN must be at least 1"):
        get_pool_config(database_url=valid_url, min_size=0)

    with pytest.raises(ValueError, match="DB_POOL_MIN must be at least 1"):
        get_pool_config(database_url=valid_url, min_size=-2)

    with pytest.raises(ValueError, match="DB_POOL_MIN must be a valid integer"):
        get_pool_config(database_url=valid_url, min_size="invalid")  # type: ignore


def test_4_invalid_db_pool_max() -> None:
    """4. Reject invalid DB_POOL_MAX (non-integer)."""
    valid_url = "postgresql://localhost/db"
    with pytest.raises(ValueError, match="DB_POOL_MAX must be a valid integer"):
        get_pool_config(database_url=valid_url, max_size="not_an_int")  # type: ignore


def test_5_min_greater_than_max_rejection() -> None:
    """5. Reject configuration where min_size > max_size."""
    valid_url = "postgresql://localhost/db"
    with pytest.raises(ValueError, match="cannot be less than DB_POOL_MIN"):
        get_pool_config(database_url=valid_url, min_size=15, max_size=10)


def test_6_invalid_connect_timeout() -> None:
    """6. Reject invalid DB_CONNECT_TIMEOUT_MS (<= 0 or non-integer)."""
    valid_url = "postgresql://localhost/db"
    with pytest.raises(ValueError, match="DB_CONNECT_TIMEOUT_MS must be a positive integer"):
        get_pool_config(database_url=valid_url, timeout_ms=0)

    with pytest.raises(ValueError, match="DB_CONNECT_TIMEOUT_MS must be a positive integer"):
        get_pool_config(database_url=valid_url, timeout_ms=-500)

    with pytest.raises(ValueError, match="DB_CONNECT_TIMEOUT_MS must be a valid integer"):
        get_pool_config(database_url=valid_url, timeout_ms="fast")  # type: ignore


def test_7_sanitized_url_never_exposes_password() -> None:
    """7. Verify sanitized URL masks plain text password."""
    raw = "postgresql://admin_user:SuperSecretPassword99!@10.0.0.1:5432/proddb"
    sanitized = sanitize_database_url(raw)
    assert "SuperSecretPassword99!" not in sanitized
    assert "admin_user:***@" in sanitized
    assert "10.0.0.1:5432/proddb" in sanitized


@pytest.mark.asyncio
async def test_fail_fast_on_unreachable_host() -> None:
    """Verify pool creation fails fast within timeout when host is unreachable."""
    # 192.0.2.1 is TEST-NET-1 (RFC 5737), non-routable blackhole
    unreachable_url = "postgresql://test:pwd@192.0.2.1:54329/testdb"
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="PostgreSQL pool initialization failed"):
        await create_postgres_pool(
            database_url=unreachable_url,
            timeout_ms=500,  # Fast 500ms budget
        )
    elapsed = time.monotonic() - start
    # Ensure total elapsed time remains bounded by the budget (plus small margin for OS network stack)
    assert elapsed < 3.0


# ===========================================================================
# 2. PostgreSQL Integration Tests (Require TEST_DATABASE_URL)
# ===========================================================================

@integration_mark
@pytest.mark.asyncio
async def test_8_real_pool_opens_successfully() -> None:
    """8. Real pool opens successfully against PostgreSQL test instance."""
    pool = await create_postgres_pool(database_url=TEST_DB_URL, min_size=2, max_size=5, timeout_ms=5000)
    try:
        assert isinstance(pool, AsyncConnectionPool)
        assert not pool.closed
    finally:
        await close_postgres_pool(pool)
    assert pool.closed


@integration_mark
@pytest.mark.asyncio
async def test_9_select_1_succeeds() -> None:
    """9. SELECT 1 query executes and returns 1 over borrowed connection."""
    pool = await create_postgres_pool(database_url=TEST_DB_URL, min_size=2, max_size=5)
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1 AS ready;")
                res = await cur.fetchone()
                assert res[0] == 1
    finally:
        await close_postgres_pool(pool)


@integration_mark
@pytest.mark.asyncio
async def test_10_pool_configured_min_max_sizes() -> None:
    """10. Pool respects configured min and max sizes."""
    pool = await create_postgres_pool(database_url=TEST_DB_URL, min_size=3, max_size=7)
    try:
        assert pool.min_size == 3
        assert pool.max_size == 7
    finally:
        await close_postgres_pool(pool)


@integration_mark
@pytest.mark.asyncio
async def test_11_connection_borrowing_and_return() -> None:
    """11. Connection can be borrowed, used, and returned to pool cleanly."""
    pool = await create_postgres_pool(database_url=TEST_DB_URL, min_size=2, max_size=4)
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT current_user;")
                user = await cur.fetchone()
                assert user is not None

        # Borrow again to verify connection was recycled
        async with pool.connection() as conn2:
            async with conn2.cursor() as cur2:
                await cur2.execute("SELECT current_database();")
                db = await cur2.fetchone()
                assert db is not None
    finally:
        await close_postgres_pool(pool)


@integration_mark
@pytest.mark.asyncio
async def test_12_multiple_concurrent_borrows() -> None:
    """12. Multiple concurrent tasks can borrow connections simultaneously."""
    pool = await create_postgres_pool(database_url=TEST_DB_URL, min_size=2, max_size=6)

    async def worker(worker_id: int) -> int:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT %s::int AS id;", (worker_id,))
                row = await cur.fetchone()
                await asyncio.sleep(0.05)
                return row[0]

    try:
        results = await asyncio.gather(*(worker(i) for i in range(5)))
        assert sorted(results) == [0, 1, 2, 3, 4]
    finally:
        await close_postgres_pool(pool)


@integration_mark
@pytest.mark.asyncio
async def test_13_pool_reaches_configured_max_without_failures() -> None:
    """13. Pool handles concurrent loads up to max_size (12) without connection failure."""
    pool = await create_postgres_pool(database_url=TEST_DB_URL, min_size=2, max_size=12)

    async def query_db(n: int) -> int:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT %s + 10;", (n,))
                row = await cur.fetchone()
                return row[0]

    try:
        tasks = [query_db(i) for i in range(12)]
        outputs = await asyncio.gather(*tasks)
        assert len(outputs) == 12
        assert sorted(outputs) == [i + 10 for i in range(12)]
    finally:
        await close_postgres_pool(pool)


@integration_mark
@pytest.mark.asyncio
async def test_14_pool_closes_cleanly() -> None:
    """14. Pool closes cleanly and marks pool.closed = True."""
    pool = await create_postgres_pool(database_url=TEST_DB_URL, min_size=2, max_size=4)
    assert not pool.closed
    await close_postgres_pool(pool)
    assert pool.closed

    # Safe double-close idempotency
    await close_postgres_pool(pool)
    assert pool.closed


@integration_mark
@pytest.mark.asyncio
async def test_15_connection_cannot_be_borrowed_after_closure() -> None:
    """15. Connection cannot be borrowed once pool is closed."""
    pool = await create_postgres_pool(database_url=TEST_DB_URL, min_size=2, max_size=4)
    await close_postgres_pool(pool)

    with pytest.raises(Exception):
        async with pool.connection():
            pass


@integration_mark
@pytest.mark.asyncio
async def test_postgres_decision_repository_with_pool() -> None:
    """Verify PostgresDecisionRepository operates seamlessly over pooled connections."""
    run_migrations(database_url=TEST_DB_URL)
    pool = await create_postgres_pool(database_url=TEST_DB_URL, min_size=2, max_size=4)
    repo = PostgresDecisionRepository(pool=pool)
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        pay_id = "test_pooled_pay_001"
        await repo.save_current_decision(
            payment_id=pay_id,
            decision_id="dec_pool_001",
            final_action="RETRY",
            decision_source="pooled_repo_test",
            evaluated_at=now,
            llm_proposed_decision="RETRY",
            error=None,
        )

        dec = await repo.get_current_decision(pay_id)
        assert dec is not None
        assert dec["payment_id"] == pay_id
        assert dec["final_action"] == "RETRY"
        assert dec["decision_source"] == "pooled_repo_test"
    finally:
        await close_postgres_pool(pool)


# ===========================================================================
# 3. FastAPI Lifespan Integration Tests
# ===========================================================================

@integration_mark
@pytest.mark.asyncio
async def test_16_fastapi_lifespan_postgres_enabled() -> None:
    """
    Test FastAPI lifespan when PERSISTENCE_BACKEND=postgres:
    - pool is created at startup
    - app.state.db_pool is active and usable
    - pool is cleanly closed at shutdown
    """
    env_overrides = {
        "PERSISTENCE_BACKEND": "postgres",
        "DATABASE_URL": TEST_DB_URL,
    }
    with patch.dict(os.environ, env_overrides, clear=False):
        async with lifespan(app):
            assert hasattr(app.state, "db_pool")
            assert app.state.db_pool is not None
            assert isinstance(app.state.db_pool, AsyncConnectionPool)
            assert not app.state.db_pool.closed

            # Verify pool query execution
            async with app.state.db_pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 100;")
                    assert (await cur.fetchone())[0] == 100

        # After lifespan exits, pool must be closed
        assert app.state.db_pool.closed


@pytest.mark.asyncio
async def test_17_fastapi_lifespan_sqlite_default_regression() -> None:
    """
    Test FastAPI lifespan with default PERSISTENCE_BACKEND=sqlite:
    - does NOT require DATABASE_URL
    - app.state.db_pool is None
    - SQLite database is initialized and functional
    """
    env_overrides = {
        "PERSISTENCE_BACKEND": "sqlite",
    }
    with patch.dict(os.environ, env_overrides, clear=False):
        os.environ.pop("DATABASE_URL", None)
        os.environ.pop("POSTGRES_URL", None)

        async with lifespan(app):
            assert hasattr(app.state, "db_pool")
            assert app.state.db_pool is None
            assert hasattr(app.state, "db")
            assert app.state.db is not None
            # SQLite query check
            async with app.state.db.execute("SELECT 1;") as cur:
                assert (await cur.fetchone())[0] == 1
