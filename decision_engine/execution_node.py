"""
decision_engine/execution_node.py
=================================
Mock execution node and SQLite audit persistence for the Recovery Decision Engine.
"""

from __future__ import annotations

import pathlib
from typing import Any
from decision_engine.audit import save_decision_audit
from decision_engine.state import RecoveryState


def execution_node(
    state: RecoveryState,
    db_path: pathlib.Path | str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """
    Perform mock execution of the final action and persist audit record to SQLite.

    Parameters
    ----------
    state : RecoveryState
        Completed workflow state after guardrails (or after error short-circuit).
    db_path : pathlib.Path | str, optional
        Custom SQLite audit database path.
    request_id : str, optional
        HTTP request correlation ID.

    Returns
    -------
    dict[str, Any]
        State update containing execution audit event and final_action.
    """
    final_action = state.get("final_action", "WAIT")
    error = state.get("error")
    req_id = request_id or state.get("request_id")

    # Persist decision to SQLite audit database
    save_decision_audit(state, db_path=db_path, request_id=req_id)

    status_str = "error_halted" if error else "executed"

    audit_event = {
        "node": "execution_node",
        "status": status_str,
        "final_action": final_action,
        "payment_id": state.get("payment_id"),
    }
    if req_id:
        audit_event["request_id"] = req_id
    if error:
        audit_event["error"] = error

    return {
        "final_action": final_action,
        "audit_trail": [audit_event],
    }
