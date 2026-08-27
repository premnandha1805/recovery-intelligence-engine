"""
decision_engine/state.py
========================
LangGraph state definition for Day 6 Recovery Intelligence Engine.

Maintains strict separation between:
- observable_features (9 canonical ML features)
- payment_context (transaction and cycle state)
- customer_history (behavioral engagement and historical metrics)
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class RecoveryState(TypedDict, total=False):
    """
    State object passed between LangGraph nodes during recovery action determination.

    Attributes
    ----------
    payment_id : str
        Unique identifier for the payment.
    observable_features : dict
        The 9 observable features defined by ml.dataset.OBSERVABLE_FEATURES.
    payment_context : dict
        Payment-specific status, billing cycle, attempt counts.
    customer_history : dict
        Customer-level history, cumulative statistics, engagement metrics.
    arm_probabilities : dict
        Estimated recovery probabilities for WAIT, RETRY, RETRY_NUDGE, ESCALATE.
    arm_net_values : dict
        Estimated net expected values (INR) for each arm.
    permitted_actions : list
        Actions permitted by structural validity checks.
    llm_decision : dict
        Decision, confidence, reasoning, and risk level from the reasoning node.
    guardrail_result : dict
        Structured guardrail output (wired in subsequent tasks).
    final_action : str
        The final action selected for execution.
    error : Optional[str]
        Error description if any node encounters an invalid state.
    audit_trail : Annotated[list, operator.add]
        Append-only log of lifecycle audit events across all nodes.
    """

    payment_id: str
    observable_features: dict[str, Any]
    payment_context: dict[str, Any]
    customer_history: dict[str, Any]
    arm_probabilities: dict[str, float]
    arm_net_values: dict[str, float]
    permitted_actions: list[str]
    llm_decision: dict[str, Any]
    guardrail_result: dict[str, Any]
    final_action: str
    error: Optional[str]
    audit_trail: Annotated[list[dict[str, Any]], operator.add]
