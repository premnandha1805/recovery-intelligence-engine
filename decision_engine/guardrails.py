"""
decision_engine/guardrails.py
==============================
Deterministic Guardrails Layer for Recovery Decision Engine.

This module enforces hard safety constraints, retry limits, escalation caps,
and state transition rules on proposed recovery actions before execution.

Determinism & Scope
-------------------
- 100% deterministic rules with explicit, documented precedence.
- Consumes the existing `Action` enum from `models.schemas`.
- Does NOT recalculate probabilities, ML estimates, or policy logic.
- Returns a structured `GuardrailResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from models.schemas import Action

# ── Explicit Named Constants ────────────────────────────────────────────────
MAX_RETRIES_PER_BILLING_CYCLE: int = 3
MAX_INTERVENTIONS_PER_WINDOW: int = 2
INTERVENTION_WINDOW_DAYS: int = 7
MAX_LIFETIME_ESCALATIONS: int = 1
MAX_CONSECUTIVE_FAILURES: int = 3

# Actions classified as active interventions (consuming customer goodwill or cost)
INTERVENTION_ACTIONS: set[Action] = {
    Action.RETRY,
    Action.RETRY_NUDGE,
    Action.ESCALATE,
}

# Terminal or recovered payment status strings representing a resolved state
RECOVERED_STATUSES: set[str] = {
    "RECOVERED",
    "SUCCESS",
    "recovered",
    "success",
    "COMPLETED",
    "completed",
}


@dataclass(frozen=True)
class GuardrailResult:
    """
    Structured outcome of guardrail evaluation.

    Attributes
    ----------
    final_action : Action
        The safe recovery action to take after applying guardrails.
    overridden : bool
        True if the proposed action was overridden by a guardrail rule.
    reason : str
        Detailed explanation of why the action was allowed or overridden.
    """

    final_action: Action
    overridden: bool
    reason: str


def _extract_field(container: Any, field_names: list[str], default: Any = None) -> Any:
    """
    Helper to extract a field from a dict, Mapping, or dataclass/object instance.
    """
    if container is None:
        return default

    if isinstance(container, Mapping):
        for name in field_names:
            if name in container:
                return container[name]
    else:
        for name in field_names:
            if hasattr(container, name):
                return getattr(container, name)

    return default


def _coerce_action(action_input: Action | str) -> Action:
    """
    Coerce a string or Action enum input cleanly to an Action enum instance.
    """
    if isinstance(action_input, Action):
        return action_input
    if isinstance(action_input, str):
        try:
            return Action(action_input)
        except ValueError:
            try:
                return Action[action_input.upper()]
            except KeyError:
                raise ValueError(f"Invalid Action label or enum value: {action_input!r}")
    raise TypeError(f"Expected Action enum or string, got {type(action_input).__name__}")


def apply_guardrails(
    proposed_action: Action | str,
    payment_context: Any = None,
    customer_history: Any = None,
) -> GuardrailResult:
    """
    Evaluate a proposed recovery action against deterministic safety guardrails.

    Precedence Order
    ----------------
    1. Invalid State Transition (Payment already recovered/succeeded)
    2. Consecutive Failure Stop Rule (Consecutive failures >= MAX_CONSECUTIVE_FAILURES)
    3. Escalation Cap (Lifetime escalations >= MAX_LIFETIME_ESCALATIONS)
    4. Retry Limit (Cycle retries >= MAX_RETRIES_PER_BILLING_CYCLE)
    5. Intervention Window Limit (Interventions in window >= MAX_INTERVENTIONS_PER_WINDOW)

    Parameters
    ----------
    proposed_action : Action | str
        Action proposed by upstream policy or engine.
    payment_context : Any, optional
        Dictionary or object with payment fields (e.g. status, retry count).
    customer_history : Any, optional
        Dictionary or object with customer history (e.g. lifetime_escalations, interventions).

    Returns
    -------
    GuardrailResult
        Structured dataclass containing final_action, overridden flag, and reason.
    """
    action = _coerce_action(proposed_action)

    # Combine field extraction across payment_context and customer_history
    status = str(
        _extract_field(payment_context, ["status", "state", "payment_status"], "")
    ).strip()

    consecutive_failures = int(
        _extract_field(
            payment_context,
            ["consecutive_failures", "consecutive_failed_cycles", "recent_consecutive_failures"],
            _extract_field(
                customer_history,
                ["consecutive_failures", "consecutive_failed_cycles", "recent_consecutive_failures"],
                0,
            ),
        )
        or 0
    )

    lifetime_escalations = int(
        _extract_field(
            customer_history,
            ["lifetime_escalations", "escalation_count", "total_escalations"],
            _extract_field(
                payment_context,
                ["lifetime_escalations", "escalation_count"],
                0,
            ),
        )
        or 0
    )

    retry_count = int(
        _extract_field(
            payment_context,
            ["retry_count_current_cycle", "retries_this_cycle", "cycle_retry_count", "attempt_number"],
            _extract_field(
                customer_history,
                ["retry_count_current_cycle", "retries_this_cycle"],
                0,
            ),
        )
        or 0
    )

    interventions_in_window = int(
        _extract_field(
            customer_history,
            ["interventions_last_7_days", "interventions_in_window", "recent_interventions_count"],
            _extract_field(
                payment_context,
                ["interventions_last_7_days", "interventions_in_window"],
                0,
            ),
        )
        or 0
    )

    # -------------------------------------------------------------------------
    # RULE 1 — Invalid State Transition
    # -------------------------------------------------------------------------
    if status in RECOVERED_STATUSES:
        if action in INTERVENTION_ACTIONS:
            return GuardrailResult(
                final_action=Action.WAIT,
                overridden=True,
                reason=f"Invalid state transition: payment is already {status!r}. Overriding {action.value} to Action.WAIT.",
            )

    # -------------------------------------------------------------------------
    # RULE 2 — Consecutive Failure Stop Rule
    # -------------------------------------------------------------------------
    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        if action != Action.STOP:
            return GuardrailResult(
                final_action=Action.STOP,
                overridden=True,
                reason=(
                    f"Consecutive failure limit reached ({consecutive_failures} >= {MAX_CONSECUTIVE_FAILURES}). "
                    f"Overriding {action.value} to Action.STOP."
                ),
            )
        return GuardrailResult(
            final_action=Action.STOP,
            overridden=False,
            reason=f"Action.STOP matches consecutive failure stop rule ({consecutive_failures} failures).",
        )

    # -------------------------------------------------------------------------
    # RULE 3 — Escalation Cap
    # -------------------------------------------------------------------------
    if action == Action.ESCALATE and lifetime_escalations >= MAX_LIFETIME_ESCALATIONS:
        return GuardrailResult(
            final_action=Action.WAIT,
            overridden=True,
            reason=(
                f"Lifetime escalation cap reached ({lifetime_escalations} >= {MAX_LIFETIME_ESCALATIONS}). "
                f"Overriding Action.ESCALATE to Action.WAIT."
            ),
        )

    # -------------------------------------------------------------------------
    # RULE 4 — Max Retries Per Billing Cycle
    # -------------------------------------------------------------------------
    if action == Action.RETRY and retry_count >= MAX_RETRIES_PER_BILLING_CYCLE:
        return GuardrailResult(
            final_action=Action.WAIT,
            overridden=True,
            reason=(
                f"Maximum retries per billing cycle reached ({retry_count} >= {MAX_RETRIES_PER_BILLING_CYCLE}). "
                f"Overriding Action.RETRY to Action.WAIT."
            ),
        )

    # -------------------------------------------------------------------------
    # RULE 5 — Max Interventions Per Window
    # -------------------------------------------------------------------------
    if action in INTERVENTION_ACTIONS and interventions_in_window >= MAX_INTERVENTIONS_PER_WINDOW:
        return GuardrailResult(
            final_action=Action.WAIT,
            overridden=True,
            reason=(
                f"Maximum interventions in {INTERVENTION_WINDOW_DAYS}-day window reached "
                f"({interventions_in_window} >= {MAX_INTERVENTIONS_PER_WINDOW}). "
                f"Overriding {action.value} to Action.WAIT."
            ),
        )

    # -------------------------------------------------------------------------
    # PASS-THROUGH — No guardrail violations
    # -------------------------------------------------------------------------
    return GuardrailResult(
        final_action=action,
        overridden=False,
        reason=f"Proposed action {action.value} passed all deterministic guardrails.",
    )


# Alias matching Day 6D graph specification: guardrails.check(...)
check = apply_guardrails

