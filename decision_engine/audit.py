"""
decision_engine/audit.py
========================
SQLite Audit persistence for the Recovery Decision Engine.

Persists full end-to-end reasoning-to-execution chains into decision_engine/audit.db
with strict UPSERT semantics keyed by payment_id.
Reuses the canonical Decision dataclass from models.schemas.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import sqlite3
from typing import Any, Optional

from models.schemas import Action, Decision
from decision_engine.state import RecoveryState

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_AUDIT_DB_PATH = _REPO_ROOT / "decision_engine" / "audit.db"

# Schema definition with PRIMARY KEY on payment_id to enforce UPSERT
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS decision_audit (
    payment_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    raw_arm_probabilities TEXT,
    raw_arm_net_values TEXT,
    llm_proposed_decision TEXT,
    llm_confidence REAL,
    llm_reasoning TEXT,
    llm_risk_level TEXT,
    expected_incremental_value REAL,
    guardrail_verdict TEXT,
    guardrail_reason TEXT,
    final_action TEXT NOT NULL,
    decision_source TEXT NOT NULL,
    error_status TEXT,
    timestamp TEXT NOT NULL
);
"""

UPSERT_SQL = """
INSERT INTO decision_audit (
    payment_id,
    decision_id,
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
    error_status,
    timestamp
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(payment_id) DO UPDATE SET
    decision_id=excluded.decision_id,
    raw_arm_probabilities=excluded.raw_arm_probabilities,
    raw_arm_net_values=excluded.raw_arm_net_values,
    llm_proposed_decision=excluded.llm_proposed_decision,
    llm_confidence=excluded.llm_confidence,
    llm_reasoning=excluded.llm_reasoning,
    llm_risk_level=excluded.llm_risk_level,
    expected_incremental_value=excluded.expected_incremental_value,
    guardrail_verdict=excluded.guardrail_verdict,
    guardrail_reason=excluded.guardrail_reason,
    final_action=excluded.final_action,
    decision_source=excluded.decision_source,
    error_status=excluded.error_status,
    timestamp=excluded.timestamp;
"""


def init_audit_db(db_path: pathlib.Path | str | None = None) -> str:
    """Ensure SQLite audit database and table exist."""
    path_str = str(db_path) if db_path is not None else str(DEFAULT_AUDIT_DB_PATH)
    if path_str != ":memory:":
        parent = pathlib.Path(path_str).parent
        parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path_str) as conn:
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()
    return path_str


def save_decision_audit(
    state: RecoveryState,
    db_path: pathlib.Path | str | None = None,
) -> Decision:
    """
    Persist full decision record to SQLite audit database using UPSERT semantics.

    Parameters
    ----------
    state : RecoveryState
        Completed workflow state.
    db_path : pathlib.Path | str, optional
        Custom database path (for test isolation).

    Returns
    -------
    Decision
        Instantiated canonical Decision object from models.schemas.
    """
    path_str = init_audit_db(db_path)

    payment_id = state.get("payment_id") or "UNKNOWN_PAYMENT"
    decision_id = f"dec_{payment_id}"

    # Extract state fields
    error = state.get("error")
    is_error_path = bool(error)

    llm_decision = state.get("llm_decision", {})
    guardrail_result = state.get("guardrail_result", {})

    final_action_raw = state.get("final_action", "WAIT")
    try:
        action_enum = Action(final_action_raw)
    except (ValueError, KeyError):
        action_enum = Action.WAIT

    if is_error_path:
        llm_proposed = "N/A — error path"
        llm_confidence = 0.0
        llm_reasoning = f"Error path: {error}"
        llm_risk = "none"
        expected_val = 0.0
        guardrail_verdict = "N/A — error path"
        guardrail_reason = "Bypassed due to error"
        decision_source = "error_path"
    else:
        llm_proposed = llm_decision.get("decision", "WAIT")
        llm_confidence = float(llm_decision.get("confidence", 1.0))
        llm_reasoning = llm_decision.get("reasoning", "")
        llm_risk = llm_decision.get("risk_level", "medium")
        expected_val = float(llm_decision.get("expected_incremental_value", 0.0))
        guardrail_verdict = guardrail_result.get("status", "passed")
        guardrail_reason = guardrail_result.get("reason", "")
        decision_source = llm_decision.get("decision_source", "llm")

    raw_arm_probas_json = json.dumps(state.get("arm_probabilities", {}))
    raw_arm_net_json = json.dumps(state.get("arm_net_values", {}))
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Reuse existing canonical Decision dataclass
    decision_entity = Decision(
        decision_id=decision_id,
        payment_id=payment_id,
        action=action_enum,
        confidence=llm_confidence,
        expected_incremental_value=expected_val,
        numeric_source=decision_source,
        reasoning=llm_reasoning,
        risk_level=llm_risk,
        model_version="gpt-4.1-mini",
    )

    with sqlite3.connect(path_str) as conn:
        conn.execute(
            UPSERT_SQL,
            (
                payment_id,
                decision_id,
                raw_arm_probas_json,
                raw_arm_net_json,
                llm_proposed,
                llm_confidence,
                llm_reasoning,
                llm_risk,
                expected_val,
                guardrail_verdict,
                guardrail_reason,
                action_enum.value,
                decision_source,
                error,
                now_iso,
            ),
        )
        conn.commit()

    return decision_entity


def get_audit_record(
    payment_id: str,
    db_path: pathlib.Path | str | None = None,
) -> Optional[dict[str, Any]]:
    """Retrieve an audit record by payment_id."""
    path_str = str(db_path) if db_path is not None else str(DEFAULT_AUDIT_DB_PATH)
    if not pathlib.Path(path_str).exists() and path_str != ":memory:":
        return None

    with sqlite3.connect(path_str) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM decision_audit WHERE payment_id = ?",
            (payment_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def get_audit_row_count(db_path: pathlib.Path | str | None = None) -> int:
    """Count total records in decision_audit table."""
    path_str = str(db_path) if db_path is not None else str(DEFAULT_AUDIT_DB_PATH)
    if not pathlib.Path(path_str).exists() and path_str != ":memory:":
        return 0

    with sqlite3.connect(path_str) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM decision_audit")
        return int(cur.fetchone()[0])


def get_audit_records_count_for_payment(
    payment_id: str,
    db_path: pathlib.Path | str | None = None,
) -> int:
    """Count records for a specific payment_id to verify uniqueness."""
    path_str = str(db_path) if db_path is not None else str(DEFAULT_AUDIT_DB_PATH)
    if not pathlib.Path(path_str).exists() and path_str != ":memory:":
        return 0

    with sqlite3.connect(path_str) as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM decision_audit WHERE payment_id = ?",
            (payment_id,),
        )
        return int(cur.fetchone()[0])
