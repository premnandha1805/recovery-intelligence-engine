"""
decision_engine/guardrail_node.py
=================================
Deterministic Guardrail Node in the Recovery Decision Engine LangGraph workflow.

Enforces authoritative safety rules, caps, and transition validation over the
LLM's proposed action using decision_engine.guardrails.check.
"""

from __future__ import annotations

from typing import Any
import decision_engine.guardrails as guardrails
from decision_engine.state import RecoveryState


def guardrail_node(state: RecoveryState) -> dict[str, Any]:
    """
    Validate proposed LLM recovery action against deterministic guardrails.

    Parameters
    ----------
    state : RecoveryState
        Current workflow state.

    Returns
    -------
    dict[str, Any]
        State update containing guardrail_result, final_action, and audit_trail.
    """
    # If prior error, preserve error state and bypass guardrails
    if state.get("error"):
        return {
            "final_action": state.get("final_action", "WAIT"),
            "audit_trail": [
                {
                    "node": "guardrail_node",
                    "status": "skipped",
                    "reason": f"Bypassed due to prior error: {state.get('error')}",
                }
            ],
        }

    llm_decision = state.get("llm_decision", {})
    proposed_action = llm_decision.get("decision", "WAIT")
    payment_context = state.get("payment_context", {})
    customer_history = state.get("customer_history", {})

    # Authoritative guardrail check
    result = guardrails.check(
        proposed_action,
        payment_context,
        customer_history,
    )

    final_action_str = result.final_action.value if hasattr(result.final_action, "value") else str(result.final_action)
    status_str = "overridden" if result.overridden else "passed"

    guardrail_result_dict = {
        "status": status_str,
        "proposed_action": proposed_action,
        "final_action": final_action_str,
        "overridden": result.overridden,
        "reason": result.reason,
    }

    audit_event = {
        "node": "guardrail_node",
        "status": status_str,
        "proposed_action": proposed_action,
        "final_action": final_action_str,
        "overridden": result.overridden,
        "reason": result.reason,
    }

    return {
        "guardrail_result": guardrail_result_dict,
        "final_action": final_action_str,
        "audit_trail": [audit_event],
    }
