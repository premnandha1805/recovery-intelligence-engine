"""
decision_engine/test_guardrails.py
==================================
Unit tests for Day 6 Deterministic Guardrails.
"""

from __future__ import annotations

import pytest

from models.schemas import Action
from decision_engine.guardrails import (
    GuardrailResult,
    apply_guardrails,
    MAX_RETRIES_PER_BILLING_CYCLE,
    MAX_INTERVENTIONS_PER_WINDOW,
    INTERVENTION_WINDOW_DAYS,
    MAX_LIFETIME_ESCALATIONS,
    MAX_CONSECUTIVE_FAILURES,
)


def test_escalation_cap_override_mandatory_demo():
    """
    REQUIRED DEMO TEST 1: Escalation Cap Override
    Assert proposed_action = ESCALATE with lifetime_escalations >= MAX_LIFETIME_ESCALATIONS
    overrides ESCALATE and reason contains 'escalation'.
    """
    proposed_action = Action.ESCALATE
    customer_history = {"lifetime_escalations": MAX_LIFETIME_ESCALATIONS}
    payment_context = {"status": "failed"}

    result = apply_guardrails(proposed_action, payment_context, customer_history)

    assert isinstance(result, GuardrailResult)
    assert result.overridden is True
    assert result.final_action != Action.ESCALATE
    assert result.final_action == Action.WAIT
    assert "escalation" in result.reason.lower()


def test_retry_limit_override():
    """
    Test Rule 4: Maximum retries per billing cycle override.
    """
    proposed_action = Action.RETRY
    payment_context = {"retry_count_current_cycle": MAX_RETRIES_PER_BILLING_CYCLE}

    result = apply_guardrails(proposed_action, payment_context=payment_context)

    assert result.overridden is True
    assert result.final_action != Action.RETRY
    assert result.final_action == Action.WAIT
    assert "retries" in result.reason.lower() or "retry" in result.reason.lower()


def test_intervention_window_override():
    """
    Test Rule 5: Maximum interventions in 7-day window override.
    """
    proposed_action = Action.RETRY_NUDGE
    customer_history = {"interventions_last_7_days": MAX_INTERVENTIONS_PER_WINDOW}

    result = apply_guardrails(proposed_action, customer_history=customer_history)

    assert result.overridden is True
    assert result.final_action == Action.WAIT
    assert "7-day" in result.reason.lower() or "intervention" in result.reason.lower()


def test_consecutive_failure_stop_behavior():
    """
    Test Rule 2: Consecutive failure stop rule.
    When consecutive failures >= MAX_CONSECUTIVE_FAILURES, forces Action.STOP.
    """
    proposed_action = Action.RETRY
    payment_context = {"consecutive_failures": MAX_CONSECUTIVE_FAILURES}

    result = apply_guardrails(proposed_action, payment_context=payment_context)

    assert result.overridden is True
    assert result.final_action == Action.STOP
    assert "consecutive" in result.reason.lower() or "stop" in result.reason.lower()


def test_invalid_transition_from_recovered():
    """
    Test Rule 1: Invalid transition from RECOVERED state.
    """
    proposed_action = Action.RETRY
    payment_context = {"status": "RECOVERED"}

    result = apply_guardrails(proposed_action, payment_context=payment_context)

    assert result.overridden is True
    assert result.final_action == Action.WAIT
    assert "recovered" in result.reason.lower()


def test_guardrail_pass_through():
    """
    Test pass-through when no guardrail rules are violated.
    """
    proposed_action = Action.RETRY
    payment_context = {"status": "failed", "retry_count_current_cycle": 0}
    customer_history = {"lifetime_escalations": 0, "interventions_last_7_days": 0}

    result = apply_guardrails(proposed_action, payment_context, customer_history)

    assert result.overridden is False
    assert result.final_action == Action.RETRY
    assert "passed" in result.reason.lower()


def test_guardrail_determinism():
    """
    Test 100% deterministic execution: repeated identical invocations produce identical output.
    """
    proposed_action = Action.ESCALATE
    customer_history = {"lifetime_escalations": 2}
    payment_context = {"status": "failed"}

    first_result = apply_guardrails(proposed_action, payment_context, customer_history)

    for _ in range(100):
        res = apply_guardrails(proposed_action, payment_context, customer_history)
        assert res == first_result


def test_multiple_violations_precedence_order():
    """
    Test precedent evaluation order when multiple guardrails are triggered simultaneously:
    1. State transition (RECOVERED) takes priority over consecutive failures, escalation cap, etc.
    """
    proposed_action = Action.ESCALATE
    # Triggers Rule 1 (RECOVERED), Rule 2 (consecutive_failures=3), Rule 3 (escalations=1)
    payment_context = {
        "status": "RECOVERED",
        "consecutive_failures": 5,
    }
    customer_history = {"lifetime_escalations": 2}

    result = apply_guardrails(proposed_action, payment_context, customer_history)

    # Rule 1 must win over Rule 2 and Rule 3
    assert result.overridden is True
    assert "state transition" in result.reason.lower()
