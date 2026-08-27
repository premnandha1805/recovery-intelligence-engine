"""
decision_engine/estimation_node.py
==================================
Estimation node consuming ml.decision.CausalUpliftPolicy.

Computes potential recovery probabilities and net values across all four
treatment arms (WAIT, RETRY, RETRY_NUDGE, ESCALATE) and performs structural
action filtering.
"""

from __future__ import annotations

from typing import Any, Mapping
import pandas as pd

from decision_engine.state import RecoveryState
from decision_engine.guardrails import MAX_RETRIES_PER_BILLING_CYCLE
from models.schemas import Action
from policy.cost_config import ACTION_COSTS
from ml.dataset import OBSERVABLE_FEATURES

# Lazy global policy instance to avoid re-fitting models across calls
_POLICY_INSTANCE: Any = None


def get_policy() -> Any:
    """Lazily load and cache the CausalUpliftPolicy."""
    global _POLICY_INSTANCE
    if _POLICY_INSTANCE is None:
        from ml.decision import CausalUpliftPolicy
        _POLICY_INSTANCE = CausalUpliftPolicy()
    return _POLICY_INSTANCE


def estimation_node(
    state: RecoveryState,
    policy: Any = None,
) -> dict[str, Any]:
    """
    Compute causal treatment effects and net financial values.

    Parameters
    ----------
    state : RecoveryState
        Current LangGraph workflow state.
    policy : CausalUpliftPolicy, optional
        Injected policy instance (useful for unit testing with mocks).

    Returns
    -------
    dict[str, Any]
        Partial state dictionary update.
    """
    # 1. Skip if previous node set an error
    error = state.get("error")
    if error:
        return {
            "audit_trail": [
                {
                    "node": "estimation_node",
                    "status": "skipped",
                    "reason": f"Skipped due to prior error: {error}",
                }
            ]
        }

    observable_features = state.get("observable_features", {})
    payment_context = state.get("payment_context", {})

    missing_cols = [c for c in OBSERVABLE_FEATURES if c not in observable_features]
    if missing_cols:
        err = f"Missing required observable features: {missing_cols}"
        return {
            "error": err,
            "audit_trail": [
                {
                    "node": "estimation_node",
                    "status": "error",
                    "reason": err,
                }
            ],
        }

    # 2. Execute CausalUpliftPolicy estimation
    active_policy = policy if policy is not None else get_policy()

    df_single = pd.DataFrame([observable_features])[OBSERVABLE_FEATURES]
    probas = active_policy.t_learner.predict_proba(df_single).iloc[0]

    amount = float(observable_features.get("amount", 1.0))

    arm_probabilities = {
        "WAIT": float(probas["WAIT"]),
        "RETRY": float(probas["RETRY"]),
        "RETRY_NUDGE": float(probas["RETRY_NUDGE"]),
        "ESCALATE": float(probas["ESCALATE"]),
    }

    arm_net_values = {
        "WAIT": float((arm_probabilities["WAIT"] * amount) - float(ACTION_COSTS[Action.WAIT])),
        "RETRY": float((arm_probabilities["RETRY"] * amount) - float(ACTION_COSTS[Action.RETRY])),
        "RETRY_NUDGE": float((arm_probabilities["RETRY_NUDGE"] * amount) - float(ACTION_COSTS[Action.RETRY_NUDGE])),
        "ESCALATE": float((arm_probabilities["ESCALATE"] * amount) - float(ACTION_COSTS[Action.ESCALATE])),
    }

    # 3. Structural action filtering (sanity checks from payment context)
    permitted = ["WAIT", "RETRY", "RETRY_NUDGE", "ESCALATE"]

    status = str(payment_context.get("status", "")).upper()
    if status in ("RECOVERED", "SUCCESS", "COMPLETED"):
        permitted = ["WAIT"]

    retry_count = int(payment_context.get("retry_count_current_cycle", 0))
    if retry_count >= MAX_RETRIES_PER_BILLING_CYCLE and "RETRY" in permitted:
        permitted.remove("RETRY")

    # Safe fallback guarantee
    if "WAIT" not in permitted:
        permitted.append("WAIT")

    return {
        "arm_probabilities": arm_probabilities,
        "arm_net_values": arm_net_values,
        "permitted_actions": permitted,
        "audit_trail": [
            {
                "node": "estimation_node",
                "status": "success",
                "permitted_actions": permitted,
            }
        ],
    }
