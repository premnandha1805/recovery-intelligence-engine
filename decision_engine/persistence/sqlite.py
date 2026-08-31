"""
decision_engine/persistence/sqlite.py
======================================
SQLite concrete implementation of DecisionRepository protocol.
Wraps existing verified Day 7 async_save_decision_audit for atomicity.
"""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
from typing import Any, Optional
import aiosqlite

from decision_engine.audit import (
    CREATE_EVENTS_INDEX_SQL,
    CREATE_EVENTS_TABLE_SQL,
    CREATE_TABLE_SQL,
    DEFAULT_AUDIT_DB_PATH,
    INSERT_EVENT_SQL,
    UPSERT_SQL,
    async_save_decision_audit,
)

logger = logging.getLogger("decision_engine.persistence.sqlite")


async def open_sqlite_repository(
    db_path: str | pathlib.Path | None = None,
) -> tuple[aiosqlite.Connection, SqliteDecisionRepository]:
    """
    Open an aiosqlite connection with WAL mode and table initialization,
    and return (db, SqliteDecisionRepository(db)).
    """
    path_str = str(db_path) if db_path is not None else str(DEFAULT_AUDIT_DB_PATH)
    pathlib.Path(path_str).parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(path_str)
    await db.execute("PRAGMA journal_mode=WAL;")
    await db.execute("PRAGMA synchronous=NORMAL;")
    await db.execute("PRAGMA busy_timeout=5000;")
    await db.execute(CREATE_TABLE_SQL)
    async with db.execute("PRAGMA table_info(decision_audit)") as cur:
        cols = [c[1] for c in await cur.fetchall()]
        if "request_id" not in cols:
            await db.execute("ALTER TABLE decision_audit ADD COLUMN request_id TEXT;")
        if "state_fingerprint" not in cols:
            await db.execute("ALTER TABLE decision_audit ADD COLUMN state_fingerprint TEXT;")
    await db.execute(CREATE_EVENTS_TABLE_SQL)
    await db.execute(CREATE_EVENTS_INDEX_SQL)
    await db.commit()
    return db, SqliteDecisionRepository(db)


class SqliteDecisionRepository:
    """
    SQLite implementation of DecisionRepository.
    Backed by an aiosqlite.Connection and delegates atomic writes to async_save_decision_audit.
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def get_current_decision(
        self, payment_id: str, request_id: Optional[str] = None
    ) -> dict[str, Any] | None:
        """
        Retrieve latest decision audit record for payment_id from SQLite.
        """
        query = "SELECT * FROM decision_audit WHERE payment_id = ?"
        async with self.db.execute(query, (payment_id,)) as cursor:
            cursor.row_factory = aiosqlite.Row
            row = await cursor.fetchone()
            if row is None:
                return None
            res = dict(row)

            # Unpack JSON strings if present for consistency with Postgres/in-memory
            for json_field in ("raw_arm_probabilities", "raw_arm_net_values"):
                val = res.get(json_field)
                if isinstance(val, str):
                    try:
                        res[json_field] = json.loads(val)
                    except Exception:
                        pass

            # Map SQLite column names to canonical schema if needed
            if "error_status" in res and "error" not in res:
                res["error"] = res["error_status"]
            if "timestamp" in res and "evaluated_at" not in res:
                res["evaluated_at"] = res["timestamp"]

            return res

    async def save_current_decision(self, **kwargs: Any) -> None:
        """
        Persist or update the current decision state with UPSERT semantics.
        """
        payment_id = kwargs.get("payment_id")
        if not payment_id:
            raise ValueError("payment_id is required for save_current_decision")

        decision_id = kwargs.get("decision_id") or f"dec_{payment_id}"
        request_id = kwargs.get("request_id")
        raw_probs = json.dumps(kwargs.get("raw_arm_probabilities") or {})
        raw_net = json.dumps(kwargs.get("raw_arm_net_values") or {})
        llm_proposed = kwargs.get("llm_proposed_decision")
        llm_conf = kwargs.get("llm_confidence")
        llm_reas = kwargs.get("llm_reasoning")
        llm_risk = kwargs.get("llm_risk_level")
        exp_val = kwargs.get("expected_incremental_value")
        g_verdict = kwargs.get("guardrail_verdict")
        g_reason = kwargs.get("guardrail_reason")
        final_action = kwargs.get("final_action", "WAIT")
        decision_source = kwargs.get("decision_source", "unknown")
        error = kwargs.get("error")
        timestamp = (kwargs.get("evaluated_at") or datetime.datetime.now(datetime.timezone.utc)).isoformat()
        state_fingerprint = kwargs.get("state_fingerprint")

        await self.db.execute(
            UPSERT_SQL,
            (
                payment_id,
                decision_id,
                request_id,
                raw_probs,
                raw_net,
                llm_proposed,
                llm_conf,
                llm_reas,
                llm_risk,
                exp_val,
                g_verdict,
                g_reason,
                final_action,
                decision_source,
                error,
                timestamp,
                state_fingerprint,
            ),
        )
        await self.db.commit()

    async def append_decision_event(self, **kwargs: Any) -> None:
        """
        Append an immutable audit event record into decision_audit_events.
        """
        payment_id = kwargs.get("payment_id")
        if not payment_id:
            raise ValueError("payment_id is required for append_decision_event")

        import uuid
        decision_id = kwargs.get("decision_id") or str(uuid.uuid4())
        request_id = kwargs.get("request_id")
        evaluated_at = (kwargs.get("evaluated_at") or datetime.datetime.now(datetime.timezone.utc)).isoformat()
        decision_source = kwargs.get("decision_source")
        final_action = kwargs.get("final_action", "WAIT")
        model_decision = kwargs.get("model_decision")
        llm_decision = kwargs.get("llm_proposed_decision")
        guardrail_overridden = kwargs.get("guardrail_overridden")
        guardrail_reason = kwargs.get("guardrail_reason")
        state_fingerprint = kwargs.get("state_fingerprint")

        await self.db.execute(
            INSERT_EVENT_SQL,
            (
                decision_id,
                payment_id,
                request_id,
                evaluated_at,
                decision_source,
                final_action,
                model_decision,
                llm_decision,
                guardrail_overridden,
                guardrail_reason,
                state_fingerprint,
            ),
        )
        await self.db.commit()

    async def get_events(
        self, payment_id: str, request_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve all audit events for payment_id ordered chronologically.
        """
        query = "SELECT * FROM decision_audit_events WHERE payment_id = ? ORDER BY evaluated_at ASC"
        async with self.db.execute(query, (payment_id,)) as cursor:
            cursor.row_factory = aiosqlite.Row
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def save_decision_with_event(self, **kwargs: Any) -> None:
        """
        Atomically persist current decision and append audit event.
        MANDATORY: Wraps the existing verified Day 7 async_save_decision_audit function.
        """
        payment_id = kwargs.get("payment_id")
        if not payment_id:
            raise ValueError("payment_id is required for save_decision_with_event")

        state = kwargs.get("state")
        if state is None:
            llm_decision = {
                "decision": kwargs.get("llm_proposed_decision", "WAIT"),
                "confidence": kwargs.get("llm_confidence", 1.0),
                "reasoning": kwargs.get("llm_reasoning", ""),
                "risk_level": kwargs.get("llm_risk_level", "medium"),
                "expected_incremental_value": kwargs.get("expected_incremental_value", 0.0),
                "decision_source": kwargs.get("decision_source", "llm"),
            }
            guardrail_result = {
                "status": kwargs.get("guardrail_verdict", "passed"),
                "reason": kwargs.get("guardrail_reason", ""),
                "overridden": kwargs.get("guardrail_overridden", False),
            }
            state = {
                "payment_id": payment_id,
                "decision_id": kwargs.get("decision_id"),
                "arm_probabilities": kwargs.get("raw_arm_probabilities") or {},
                "arm_net_values": kwargs.get("raw_arm_net_values") or {},
                "llm_decision": llm_decision,
                "guardrail_result": guardrail_result,
                "final_action": kwargs.get("final_action", "WAIT"),
                "error": kwargs.get("error"),
                "state_fingerprint": kwargs.get("state_fingerprint"),
            }

        request_id = kwargs.get("request_id")
        await async_save_decision_audit(state, self.db, request_id=request_id)
