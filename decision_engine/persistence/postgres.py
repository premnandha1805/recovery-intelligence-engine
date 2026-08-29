"""
decision_engine/persistence/postgres.py
=======================================
PostgreSQL implementation of DecisionRepository using psycopg v3 async APIs.

- Parameterized SQL only (zero raw string interpolation)
- Uses psycopg.types.json.Jsonb for native JSONB adaptation
- Preserves native DOUBLE PRECISION and TIMESTAMPTZ types
- Independent single-operation commits (transaction orchestration is Day 8D)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import datetime
import os
from typing import Any, AsyncIterator, Optional, Union
import uuid

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


class PostgresDecisionRepository:
    """
    PostgreSQL concrete implementation of DecisionRepository.

    Backed strictly by psycopg v3 async APIs (no asyncpg).
    Implements:
    - get_current_decision(payment_id) -> dict | None
    - save_current_decision(**kwargs) -> UPSERT into decision_audit
    - append_decision_event(**kwargs) -> INSERT into decision_audit_events
    - get_events(payment_id) -> list[dict] ordered by evaluated_at ASC
    """

    def __init__(
        self,
        connection_or_url: Optional[Union[psycopg.AsyncConnection[Any], str]] = None,
        *,
        database_url: Optional[str] = None,
    ) -> None:
        """
        Initialize repository with an active psycopg AsyncConnection, connection pool,
        or database URL string.
        """
        if isinstance(connection_or_url, str):
            self._connection: Optional[psycopg.AsyncConnection[Any]] = None
            self._database_url: Optional[str] = connection_or_url
        elif connection_or_url is not None:
            self._connection = connection_or_url
            self._database_url = None
        else:
            self._connection = None
            self._database_url = database_url or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")

    @asynccontextmanager
    async def _get_cursor(self) -> AsyncIterator[psycopg.AsyncCursor[Any]]:
        """Yield an async cursor configured with dict_row factory."""
        if self._connection is not None:
            async with self._connection.cursor(row_factory=dict_row) as cur:
                yield cur
            if not getattr(self._connection, "autocommit", False):
                await self._connection.commit()
        elif self._database_url is not None:
            async with await psycopg.AsyncConnection.connect(
                self._database_url, row_factory=dict_row
            ) as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    yield cur
                if not conn.autocommit:
                    await conn.commit()
        else:
            raise ValueError(
                "Neither an active AsyncConnection nor a valid database_url is available."
            )

    async def get_current_decision(self, payment_id: str) -> dict[str, Any] | None:
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
        async with self._get_cursor() as cur:
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

        async with self._get_cursor() as cur:
            await cur.execute(query, params)

    async def append_decision_event(self, **kwargs: Any) -> None:
        """
        Append an immutable audit event record into decision_audit_events.
        """
        payment_id = kwargs.get("payment_id")
        if not payment_id:
            raise ValueError("payment_id is required for append_decision_event")

        decision_id = kwargs.get("decision_id") or str(uuid.uuid4())
        evaluated_at = kwargs.get("evaluated_at") or datetime.datetime.now(datetime.timezone.utc)

        params = {
            "decision_id": decision_id,
            "payment_id": payment_id,
            "request_id": kwargs.get("request_id"),
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

        async with self._get_cursor() as cur:
            await cur.execute(query, params)

    async def get_events(self, payment_id: str) -> list[dict[str, Any]]:
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
        async with self._get_cursor() as cur:
            await cur.execute(query, {"payment_id": payment_id})
            rows = await cur.fetchall()
            return [dict(r) for r in rows]
