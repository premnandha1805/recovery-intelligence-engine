"""
decision_engine/persistence/postgres.py
=======================================
PostgreSQL implementation of DecisionRepository using psycopg v3 async APIs.

- Parameterized SQL only (zero raw string interpolation)
- Uses psycopg.types.json.Jsonb for native JSONB adaptation
- Preserves native DOUBLE PRECISION and TIMESTAMPTZ types
- Independent single-operation commits for existing interface methods
- Day 8D: save_decision_with_event() performs both writes atomically
  inside ONE psycopg v3 `async with conn.transaction():` block.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import datetime
import logging
import os
import re
import time
from typing import Any, AsyncIterator, Optional, Union
import uuid

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from decision_engine.structured_logger import emit_log

logger = logging.getLogger("decision_engine.persistence.postgres")


# ── Configuration & Sanitization Helpers ─────────────────────────────────────

def sanitize_database_url(url: str) -> str:
    """Mask credentials in database URL to prevent secret leakage in logs/errors."""
    if not url:
        return ""
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)


def validate_database_url(url: str | None) -> str:
    """
    Validate that database URL is present, non-empty, and uses a valid postgres scheme.
    Raises ValueError if invalid. Redacts credentials and never outputs raw URL.
    """
    if not url or not isinstance(url, str) or not url.strip():
        raise ValueError("DATABASE_URL is required but was not provided.")
    clean_url = url.strip()
    if not (clean_url.startswith("postgresql://") or clean_url.startswith("postgres://")):
        raise ValueError(
            "Invalid DATABASE_URL scheme. Must start with 'postgresql://' or 'postgres://'."
        )
    return clean_url


def get_pool_config(
    database_url: Optional[str] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    timeout_ms: Optional[int] = None,
) -> dict[str, Any]:
    """
    Resolve and validate PostgreSQL connection pool configuration from arguments or environment.

    Variables:
        DATABASE_URL           PostgreSQL connection URI (required)
        DB_POOL_MIN            default 2 (>= 1, <= DB_POOL_MAX)
        DB_POOL_MAX            default 12 (>= DB_POOL_MIN)
        DB_CONNECT_TIMEOUT_MS  default 3000 (> 0)
    """
    raw_url = database_url if database_url is not None else (
        os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or os.getenv("TEST_DATABASE_URL")
    )
    valid_url = validate_database_url(raw_url)

    # DB_POOL_MIN: integer >= 1
    raw_min = min_size if min_size is not None else os.getenv("DB_POOL_MIN", "2")
    try:
        min_val = int(raw_min)
    except (TypeError, ValueError):
        raise ValueError(f"DB_POOL_MIN must be a valid integer, got {raw_min!r}")
    if min_val < 1:
        raise ValueError(f"DB_POOL_MIN must be at least 1, got {min_val}")

    # DB_POOL_MAX: integer >= DB_POOL_MIN
    # DB_POOL_MAX=12 is intentional:
    # app.state.llm_semaphore limits LLM-bound work to 5 concurrent
    # requests. Cache-hit requests still require database access but
    # do not consume an LLM semaphore slot. A pool of 12 provides
    # headroom for those requests and avoids unnecessary connection
    # starvation without being arbitrary.
    raw_max = max_size if max_size is not None else os.getenv("DB_POOL_MAX", "12")
    try:
        max_val = int(raw_max)
    except (TypeError, ValueError):
        raise ValueError(f"DB_POOL_MAX must be a valid integer, got {raw_max!r}")
    if max_val < min_val:
        raise ValueError(f"DB_POOL_MAX ({max_val}) cannot be less than DB_POOL_MIN ({min_val})")

    # DB_CONNECT_TIMEOUT_MS: positive integer
    raw_timeout = timeout_ms if timeout_ms is not None else os.getenv("DB_CONNECT_TIMEOUT_MS", "3000")
    try:
        timeout_val = int(raw_timeout)
    except (TypeError, ValueError):
        raise ValueError(f"DB_CONNECT_TIMEOUT_MS must be a valid integer, got {raw_timeout!r}")
    if timeout_val <= 0:
        raise ValueError(f"DB_CONNECT_TIMEOUT_MS must be a positive integer, got {timeout_val}")

    return {
        "database_url": valid_url,
        "min_size": min_val,
        "max_size": max_val,
        "timeout_ms": timeout_val,
    }


# ── Pool Creation & Lifecycle ────────────────────────────────────────────────

async def create_postgres_pool(
    database_url: Optional[str] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    timeout_ms: Optional[int] = None,
) -> AsyncConnectionPool:
    """
    Create, open, and verify an asynchronous psycopg PostgreSQL connection pool.

    The startup sequence MUST be:
        validate config
            ↓
        create AsyncConnectionPool(open=False)
            ↓
        open pool
            ↓
        borrow one connection
            ↓
        SELECT 1
            ↓
        pool ready

    Fails fast if config is invalid, opening times out, or verification fails.
    """
    try:
        config = get_pool_config(
            database_url=database_url,
            min_size=min_size,
            max_size=max_size,
            timeout_ms=timeout_ms,
        )
    except Exception as exc:
        raise RuntimeError(
            f"PostgreSQL pool initialization failed: invalid configuration: {exc}. Service refusing to start."
        ) from exc

    url = config["database_url"]
    min_conn = config["min_size"]
    max_conn = config["max_size"]
    total_timeout_sec = config["timeout_ms"] / 1000.0

    deadline = time.monotonic() + total_timeout_sec
    safe_url = sanitize_database_url(url)

    pool: Optional[AsyncConnectionPool] = None
    try:
        # Create pool with open=False
        pool = AsyncConnectionPool(
            conninfo=url,
            min_size=min_conn,
            max_size=max_conn,
            open=False,
            timeout=total_timeout_sec,
            connection_class=psycopg.AsyncConnection,
        )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Startup budget exceeded before opening pool at {safe_url}")

        # Explicitly open pool bounded by monotonic deadline
        await asyncio.wait_for(pool.open(wait=True, timeout=remaining), timeout=remaining)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Startup budget exceeded while awaiting pool readiness at {safe_url}")

        # Borrow one connection and run SELECT 1
        async with pool.connection(timeout=remaining) as conn:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Startup budget exceeded before ping verification at {safe_url}")
            async with conn.cursor() as cur:
                await asyncio.wait_for(cur.execute("SELECT 1;"), timeout=remaining)
                row = await cur.fetchone()
                if not row or row[0] != 1:
                    raise RuntimeError(f"Verification ping SELECT 1 returned unexpected result at {safe_url}")

        return pool

    except Exception as exc:
        if pool is not None:
            try:
                await pool.close(timeout=1.0)
            except Exception:
                pass
        safe_msg = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(
            f"PostgreSQL pool initialization failed: {safe_msg}. Service refusing to start."
        ) from exc


async def close_postgres_pool(pool: Optional[AsyncConnectionPool]) -> None:
    """
    Cleanly close an active AsyncConnectionPool at shutdown.
    Tolerates uninitialized or already-closed pools safely.
    """
    if pool is None:
        return
    try:
        if not getattr(pool, "closed", False):
            await pool.close(timeout=3.0)
    except Exception as exc:
        logger.warning(f"Error during PostgreSQL connection pool shutdown: {exc}")


# ── Health Probe ─────────────────────────────────────────────────────────────

HEALTH_PROBE_TIMEOUT_S = 2.0


async def check_pool_health(
    pool: Optional[AsyncConnectionPool],
    timeout_s: float = HEALTH_PROBE_TIMEOUT_S,
) -> str:
    """
    Lightweight PostgreSQL health probe using the existing connection pool.

    Acquires a connection from the pool, executes ``SELECT 1``, and returns
    ``"ok"`` if successful within *timeout_s* seconds, or ``"unavailable"``
    on any failure (closed pool, network error, timeout).

    Security: never exposes DATABASE_URL, credentials, or stack traces.
    """
    if pool is None:
        return "unavailable"

    if getattr(pool, "closed", False):
        return "unavailable"

    try:
        async with asyncio.timeout(timeout_s):
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1;")
                    row = await cur.fetchone()
                    if row and row[0] == 1:
                        return "ok"
                    return "unavailable"
    except Exception:
        # Catch all: timeout, connection errors, pool exhaustion, etc.
        # Never leak exception details to caller.
        return "unavailable"


# ── Decision Repository ──────────────────────────────────────────────────────

class PostgresDecisionRepository:
    """
    PostgreSQL concrete implementation of DecisionRepository.

    Backed strictly by psycopg v3 async APIs (no asyncpg).
    Supports either an AsyncConnectionPool or standalone AsyncConnection / URL.
    Implements:
    - get_current_decision(payment_id) -> dict | None
    - save_current_decision(**kwargs) -> UPSERT into decision_audit
    - append_decision_event(**kwargs) -> INSERT into decision_audit_events
    - get_events(payment_id) -> list[dict] ordered by evaluated_at ASC
    """

    def __init__(
        self,
        connection_or_pool_or_url: Optional[Union[AsyncConnectionPool, psycopg.AsyncConnection[Any], str]] = None,
        *,
        pool: Optional[AsyncConnectionPool] = None,
        database_url: Optional[str] = None,
    ) -> None:
        """
        Initialize repository with an AsyncConnectionPool, active psycopg AsyncConnection,
        or database URL string.
        """
        self._pool: Optional[AsyncConnectionPool] = None
        self._connection: Optional[psycopg.AsyncConnection[Any]] = None
        self._database_url: Optional[str] = None

        if pool is not None:
            self._pool = pool
        elif isinstance(connection_or_pool_or_url, AsyncConnectionPool):
            self._pool = connection_or_pool_or_url
        elif isinstance(connection_or_pool_or_url, str):
            self._database_url = connection_or_pool_or_url
        elif connection_or_pool_or_url is not None:
            self._connection = connection_or_pool_or_url
        else:
            self._database_url = database_url or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")

    @asynccontextmanager
    async def _get_cursor(self, request_id: Optional[str] = None) -> AsyncIterator[psycopg.AsyncCursor[Any]]:
        """Yield an async cursor configured with dict_row factory."""
        t0 = time.monotonic()
        if self._pool is not None:
            async with self._pool.connection() as conn:
                duration_ms = round((time.monotonic() - t0) * 1000, 2)
                if request_id:
                    emit_log(logger, logging.INFO, "db_connection_acquired", request_id, duration_ms=duration_ms)
                async with conn.cursor(row_factory=dict_row) as cur:
                    yield cur
                if not conn.autocommit:
                    await conn.commit()
        elif self._connection is not None:
            duration_ms = round((time.monotonic() - t0) * 1000, 2)
            if request_id:
                emit_log(logger, logging.INFO, "db_connection_acquired", request_id, duration_ms=duration_ms)
            async with self._connection.cursor(row_factory=dict_row) as cur:
                yield cur
            if not getattr(self._connection, "autocommit", False):
                await self._connection.commit()
        elif self._database_url is not None:
            async with await psycopg.AsyncConnection.connect(
                self._database_url, row_factory=dict_row
            ) as conn:
                duration_ms = round((time.monotonic() - t0) * 1000, 2)
                if request_id:
                    emit_log(logger, logging.INFO, "db_connection_acquired", request_id, duration_ms=duration_ms)
                async with conn.cursor(row_factory=dict_row) as cur:
                    yield cur
                if not conn.autocommit:
                    await conn.commit()
        else:
            raise ValueError(
                "Neither an active AsyncConnectionPool, AsyncConnection, nor valid database_url is available."
            )

    @asynccontextmanager
    async def _get_connection(self, request_id: Optional[str] = None) -> AsyncIterator[psycopg.AsyncConnection[Any]]:
        """
        Yield a raw async psycopg connection (without auto-cursor).
        Used by atomic operations that manage their own transaction boundary
        via `async with conn.transaction()`.
        """
        t0 = time.monotonic()
        if self._pool is not None:
            async with self._pool.connection() as conn:
                duration_ms = round((time.monotonic() - t0) * 1000, 2)
                if request_id:
                    emit_log(logger, logging.INFO, "db_connection_acquired", request_id, duration_ms=duration_ms)
                yield conn
        elif self._connection is not None:
            duration_ms = round((time.monotonic() - t0) * 1000, 2)
            if request_id:
                emit_log(logger, logging.INFO, "db_connection_acquired", request_id, duration_ms=duration_ms)
            yield self._connection
        elif self._database_url is not None:
            async with await psycopg.AsyncConnection.connect(self._database_url) as conn:
                duration_ms = round((time.monotonic() - t0) * 1000, 2)
                if request_id:
                    emit_log(logger, logging.INFO, "db_connection_acquired", request_id, duration_ms=duration_ms)
                yield conn
        else:
            raise ValueError(
                "Neither an active AsyncConnectionPool, AsyncConnection, nor valid database_url is available."
            )

    async def get_current_decision(
        self, payment_id: str, request_id: Optional[str] = None
    ) -> dict[str, Any] | None:
        """
        Retrieve the latest decision audit record for payment_id.
        Returns a Python dictionary with native JSONB dicts, or None if not found.
        """
        query = """
        SELECT
            payment_id,
            decision_id,
            request_id,
            raw_arm_probabilities,
            raw_arm_net_values,
            llm_proposed_decision,
            llm_confidence,
            llm_reasoning,
            llm_risk_level,
            expected_incremental_value,
            guardrail_verdict,
            guardrail_reason,
            final_action,
            decision_source,
            error,
            evaluated_at,
            state_fingerprint
        FROM decision_audit
        WHERE payment_id = %(payment_id)s;
        """
        async with self._get_cursor(request_id=request_id) as cur:
            await cur.execute(query, {"payment_id": payment_id})
            row = await cur.fetchone()
            return dict(row) if row is not None else None

    async def save_current_decision(self, **kwargs: Any) -> None:
        """
        Persist or update the current decision state using PostgreSQL ON CONFLICT UPSERT.
        """
        payment_id = kwargs.get("payment_id")
        if not payment_id:
            raise ValueError("payment_id is required for save_current_decision")

        decision_id = kwargs.get("decision_id") or f"dec_{payment_id}"
        evaluated_at = kwargs.get("evaluated_at") or datetime.datetime.now(datetime.timezone.utc)

        # Native JSONB adaptation via psycopg Jsonb wrapper
        raw_probs = kwargs.get("raw_arm_probabilities")
        raw_net = kwargs.get("raw_arm_net_values")
        probs_val = Jsonb(raw_probs) if raw_probs is not None else None
        net_val = Jsonb(raw_net) if raw_net is not None else None

        params = {
            "payment_id": payment_id,
            "decision_id": decision_id,
            "request_id": kwargs.get("request_id"),
            "raw_arm_probabilities": probs_val,
            "raw_arm_net_values": net_val,
            "llm_proposed_decision": kwargs.get("llm_proposed_decision"),
            "llm_confidence": kwargs.get("llm_confidence"),
            "llm_reasoning": kwargs.get("llm_reasoning"),
            "llm_risk_level": kwargs.get("llm_risk_level"),
            "expected_incremental_value": kwargs.get("expected_incremental_value"),
            "guardrail_verdict": kwargs.get("guardrail_verdict"),
            "guardrail_reason": kwargs.get("guardrail_reason"),
            "final_action": kwargs.get("final_action", "WAIT"),
            "decision_source": kwargs.get("decision_source", "unknown"),
            "error": kwargs.get("error"),
            "evaluated_at": evaluated_at,
            "state_fingerprint": kwargs.get("state_fingerprint"),
        }

        query = """
        INSERT INTO decision_audit (
            payment_id,
            decision_id,
            request_id,
            raw_arm_probabilities,
            raw_arm_net_values,
            llm_proposed_decision,
            llm_confidence,
            llm_reasoning,
            llm_risk_level,
            expected_incremental_value,
            guardrail_verdict,
            guardrail_reason,
            final_action,
            decision_source,
            error,
            evaluated_at,
            state_fingerprint
        ) VALUES (
            %(payment_id)s,
            %(decision_id)s,
            %(request_id)s,
            %(raw_arm_probabilities)s,
            %(raw_arm_net_values)s,
            %(llm_proposed_decision)s,
            %(llm_confidence)s,
            %(llm_reasoning)s,
            %(llm_risk_level)s,
            %(expected_incremental_value)s,
            %(guardrail_verdict)s,
            %(guardrail_reason)s,
            %(final_action)s,
            %(decision_source)s,
            %(error)s,
            %(evaluated_at)s,
            %(state_fingerprint)s
        )
        ON CONFLICT (payment_id) DO UPDATE SET
            decision_id = EXCLUDED.decision_id,
            request_id = EXCLUDED.request_id,
            raw_arm_probabilities = EXCLUDED.raw_arm_probabilities,
            raw_arm_net_values = EXCLUDED.raw_arm_net_values,
            llm_proposed_decision = EXCLUDED.llm_proposed_decision,
            llm_confidence = EXCLUDED.llm_confidence,
            llm_reasoning = EXCLUDED.llm_reasoning,
            llm_risk_level = EXCLUDED.llm_risk_level,
            expected_incremental_value = EXCLUDED.expected_incremental_value,
            guardrail_verdict = EXCLUDED.guardrail_verdict,
            guardrail_reason = EXCLUDED.guardrail_reason,
            final_action = EXCLUDED.final_action,
            decision_source = EXCLUDED.decision_source,
            error = EXCLUDED.error,
            evaluated_at = EXCLUDED.evaluated_at,
            state_fingerprint = EXCLUDED.state_fingerprint;
        """

        request_id = kwargs.get("request_id")
        try:
            async with self._get_cursor(request_id=request_id) as cur:
                await cur.execute(query, params)
        except Exception as exc:
            if request_id:
                emit_log(
                    logger,
                    logging.ERROR,
                    "db_persistence_failed",
                    request_id,
                    error_type=type(exc).__name__,
                )
            raise

    async def append_decision_event(self, **kwargs: Any) -> None:
        """
        Append an immutable audit event record into decision_audit_events.
        """
        payment_id = kwargs.get("payment_id")
        if not payment_id:
            raise ValueError("payment_id is required for append_decision_event")

        decision_id = kwargs.get("decision_id") or str(uuid.uuid4())
        evaluated_at = kwargs.get("evaluated_at") or datetime.datetime.now(datetime.timezone.utc)
        request_id = kwargs.get("request_id")

        params = {
            "decision_id": decision_id,
            "payment_id": payment_id,
            "request_id": request_id,
            "evaluated_at": evaluated_at,
            "decision_source": kwargs.get("decision_source"),
            "final_action": kwargs.get("final_action", "WAIT"),
            "model_decision": kwargs.get("model_decision"),
            "llm_proposed_decision": kwargs.get("llm_proposed_decision"),
            "guardrail_overridden": kwargs.get("guardrail_overridden"),
            "guardrail_reason": kwargs.get("guardrail_reason"),
            "state_fingerprint": kwargs.get("state_fingerprint"),
        }

        query = """
        INSERT INTO decision_audit_events (
            decision_id,
            payment_id,
            request_id,
            evaluated_at,
            decision_source,
            final_action,
            model_decision,
            llm_proposed_decision,
            guardrail_overridden,
            guardrail_reason,
            state_fingerprint
        ) VALUES (
            %(decision_id)s,
            %(payment_id)s,
            %(request_id)s,
            %(evaluated_at)s,
            %(decision_source)s,
            %(final_action)s,
            %(model_decision)s,
            %(llm_proposed_decision)s,
            %(guardrail_overridden)s,
            %(guardrail_reason)s,
            %(state_fingerprint)s
        );
        """

        try:
            async with self._get_cursor(request_id=request_id) as cur:
                await cur.execute(query, params)
        except Exception as exc:
            if request_id:
                emit_log(
                    logger,
                    logging.ERROR,
                    "db_persistence_failed",
                    request_id,
                    error_type=type(exc).__name__,
                )
            raise

    async def get_events(
        self, payment_id: str, request_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve all audit events for payment_id ordered chronologically by evaluated_at ASC.
        """
        query = """
        SELECT
            decision_id,
            payment_id,
            request_id,
            evaluated_at,
            decision_source,
            final_action,
            model_decision,
            llm_proposed_decision,
            guardrail_overridden,
            guardrail_reason,
            state_fingerprint
        FROM decision_audit_events
        WHERE payment_id = %(payment_id)s
        ORDER BY evaluated_at ASC;
        """
        async with self._get_cursor(request_id=request_id) as cur:
            await cur.execute(query, {"payment_id": payment_id})
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

    # ── Day 8D: Atomic Combined Operation ─────────────────────────────────────

    async def save_decision_with_event(self, **kwargs: Any) -> None:
        """
        Atomically persist a current-decision record AND an audit event record
        within a SINGLE psycopg v3 transaction.

        Both writes (INSERT/UPSERT into decision_audit, INSERT into
        decision_audit_events) are executed inside one
        `async with conn.transaction():` block. If the second write fails,
        psycopg rolls back the entire transaction — including the first write.

        Parameters (kwargs)
        -------------------
        payment_id         : str   (required)
        decision_id        : str   (auto-generated if absent)
        event_decision_id  : str   (separate PK for events row; auto-generated)
        All other fields mirror save_current_decision() + append_decision_event().

        Raises
        ------
        ValueError  if payment_id is missing.
        Any psycopg error propagates unmodified so callers can inspect it;
        the transaction is already rolled back by psycopg's context manager.
        """
        payment_id = kwargs.get("payment_id")
        if not payment_id:
            raise ValueError("payment_id is required for save_decision_with_event")

        decision_id = kwargs.get("decision_id") or f"dec_{payment_id}"
        # event_decision_id is the PK for the decision_audit_events row.
        # Callers may supply it explicitly (useful in failure-injection tests).
        event_decision_id = kwargs.get("event_decision_id") or str(uuid.uuid4())
        evaluated_at = kwargs.get("evaluated_at") or datetime.datetime.now(datetime.timezone.utc)
        request_id = kwargs.get("request_id")

        raw_probs = kwargs.get("raw_arm_probabilities")
        raw_net = kwargs.get("raw_arm_net_values")
        probs_val = Jsonb(raw_probs) if raw_probs is not None else None
        net_val = Jsonb(raw_net) if raw_net is not None else None

        decision_params = {
            "payment_id": payment_id,
            "decision_id": decision_id,
            "request_id": request_id,
            "raw_arm_probabilities": probs_val,
            "raw_arm_net_values": net_val,
            "llm_proposed_decision": kwargs.get("llm_proposed_decision"),
            "llm_confidence": kwargs.get("llm_confidence"),
            "llm_reasoning": kwargs.get("llm_reasoning"),
            "llm_risk_level": kwargs.get("llm_risk_level"),
            "expected_incremental_value": kwargs.get("expected_incremental_value"),
            "guardrail_verdict": kwargs.get("guardrail_verdict"),
            "guardrail_reason": kwargs.get("guardrail_reason"),
            "final_action": kwargs.get("final_action", "WAIT"),
            "decision_source": kwargs.get("decision_source", "unknown"),
            "error": kwargs.get("error"),
            "evaluated_at": evaluated_at,
            "state_fingerprint": kwargs.get("state_fingerprint"),
        }

        event_params = {
            "decision_id": event_decision_id,
            "payment_id": payment_id,
            "request_id": request_id,
            "evaluated_at": evaluated_at,
            "decision_source": kwargs.get("decision_source"),
            "final_action": kwargs.get("final_action", "WAIT"),
            "model_decision": kwargs.get("model_decision"),
            "llm_proposed_decision": kwargs.get("llm_proposed_decision"),
            "guardrail_overridden": kwargs.get("guardrail_overridden"),
            "guardrail_reason": kwargs.get("guardrail_reason"),
            "state_fingerprint": kwargs.get("state_fingerprint"),
        }

        upsert_decision_sql = """
        INSERT INTO decision_audit (
            payment_id, decision_id, request_id,
            raw_arm_probabilities, raw_arm_net_values,
            llm_proposed_decision, llm_confidence, llm_reasoning, llm_risk_level,
            expected_incremental_value, guardrail_verdict, guardrail_reason,
            final_action, decision_source, error, evaluated_at, state_fingerprint
        ) VALUES (
            %(payment_id)s, %(decision_id)s, %(request_id)s,
            %(raw_arm_probabilities)s, %(raw_arm_net_values)s,
            %(llm_proposed_decision)s, %(llm_confidence)s, %(llm_reasoning)s,
            %(llm_risk_level)s, %(expected_incremental_value)s,
            %(guardrail_verdict)s, %(guardrail_reason)s,
            %(final_action)s, %(decision_source)s, %(error)s,
            %(evaluated_at)s, %(state_fingerprint)s
        )
        ON CONFLICT (payment_id) DO UPDATE SET
            decision_id = EXCLUDED.decision_id,
            request_id = EXCLUDED.request_id,
            raw_arm_probabilities = EXCLUDED.raw_arm_probabilities,
            raw_arm_net_values = EXCLUDED.raw_arm_net_values,
            llm_proposed_decision = EXCLUDED.llm_proposed_decision,
            llm_confidence = EXCLUDED.llm_confidence,
            llm_reasoning = EXCLUDED.llm_reasoning,
            llm_risk_level = EXCLUDED.llm_risk_level,
            expected_incremental_value = EXCLUDED.expected_incremental_value,
            guardrail_verdict = EXCLUDED.guardrail_verdict,
            guardrail_reason = EXCLUDED.guardrail_reason,
            final_action = EXCLUDED.final_action,
            decision_source = EXCLUDED.decision_source,
            error = EXCLUDED.error,
            evaluated_at = EXCLUDED.evaluated_at,
            state_fingerprint = EXCLUDED.state_fingerprint;
        """

        insert_event_sql = """
        INSERT INTO decision_audit_events (
            decision_id, payment_id, request_id, evaluated_at,
            decision_source, final_action, model_decision, llm_proposed_decision,
            guardrail_overridden, guardrail_reason, state_fingerprint
        ) VALUES (
            %(decision_id)s, %(payment_id)s, %(request_id)s, %(evaluated_at)s,
            %(decision_source)s, %(final_action)s, %(model_decision)s,
            %(llm_proposed_decision)s, %(guardrail_overridden)s,
            %(guardrail_reason)s, %(state_fingerprint)s
        );
        """

        t_op_start = time.monotonic()
        try:
            async with self._get_connection(request_id=request_id) as conn:
                t_tx_start = time.monotonic()
                try:
                    async with conn.transaction():
                        if request_id:
                            emit_log(logger, logging.INFO, "db_transaction_started", request_id)
                        # Write 1 of 2: upsert current decision into decision_audit
                        async with conn.cursor() as cur:
                            await cur.execute(upsert_decision_sql, decision_params)

                        # Write 2 of 2: append audit event into decision_audit_events
                        # If this raises, psycopg rolls back the entire transaction,
                        # including the decision_audit row written above.
                        async with conn.cursor() as cur:
                            await cur.execute(insert_event_sql, event_params)
                    tx_duration_ms = round((time.monotonic() - t_tx_start) * 1000, 2)
                    if request_id:
                        emit_log(logger, logging.INFO, "db_transaction_committed", request_id, duration_ms=tx_duration_ms)
                except Exception as exc:
                    tx_duration_ms = round((time.monotonic() - t_tx_start) * 1000, 2)
                    if request_id:
                        emit_log(
                            logger,
                            logging.ERROR,
                            "db_transaction_rolled_back",
                            request_id,
                            error_type=type(exc).__name__,
                            duration_ms=tx_duration_ms,
                        )
                    raise
        except Exception as exc:
            op_duration_ms = round((time.monotonic() - t_op_start) * 1000, 2)
            if request_id:
                emit_log(
                    logger,
                    logging.ERROR,
                    "db_persistence_failed",
                    request_id,
                    error_type=type(exc).__name__,
                    duration_ms=op_duration_ms,
                )
            raise
