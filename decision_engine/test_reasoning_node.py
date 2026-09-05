"""
decision_engine/test_reasoning_node.py
======================================
Unit tests for the Azure AI Foundry reasoning node.
Completely mocked — requires no network access and no API keys.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from decision_engine.reasoning_node import (
    LLMDecision,
    reasoning_node,
    _execute_fallback,
)
from decision_engine.state import RecoveryState


def create_sample_state() -> RecoveryState:
    """Helper to generate a valid estimation-complete state for reasoning tests."""
    return {
        "payment_id": "pay_test_001",
        "observable_features": {
            "amount": 1000.0,
            "attempt_number": 1,
            "dynamic_success_rate": 0.5,
            "cumulative_failures": 1,
            "consecutive_failed_cycles": 0,
            "notification_engagement_score": 0.8,
            "contact_response_score": 0.8,
            "payment_method": "card",
            "failure_reason": "insufficient_funds",
        },
        "payment_context": {"status": "failed", "retry_count_current_cycle": 0},
        "customer_history": {"lifetime_escalations": 0},
        "arm_probabilities": {
            "WAIT": 0.20,
            "RETRY": 0.60,
            "RETRY_NUDGE": 0.75,
            "ESCALATE": 0.80,
        },
        "arm_net_values": {
            "WAIT": 200.0,
            "RETRY": 595.0,
            "RETRY_NUDGE": 735.0,
            "ESCALATE": 550.0,
        },
        "permitted_actions": ["WAIT", "RETRY", "RETRY_NUDGE", "ESCALATE"],
        "error": None,
        "audit_trail": [],
    }


def test_valid_structured_llm_response():
    """Test 6: Valid structured LLM response succeeds on Attempt 1."""
    state = create_sample_state()

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    mock_structured.invoke.return_value = LLMDecision(
        decision="RETRY_NUDGE",
        confidence=0.92,
        reasoning="High notification engagement justifies RETRY_NUDGE with highest net value.",
        risk_level="low",
    )

    update = reasoning_node(state, llm=mock_llm)

    assert update["final_action"] == "RETRY_NUDGE"
    assert update["llm_decision"]["decision"] == "RETRY_NUDGE"
    assert update["llm_decision"]["confidence"] == 0.92
    assert update["llm_decision"]["decision_source"] == "llm"
    # Test 7: expected_incremental_value populated from arm_net_values
    assert update["llm_decision"]["expected_incremental_value"] == 735.0
    assert len(update["audit_trail"]) == 1
    assert update["audit_trail"][0]["status"] == "success"
    assert update["audit_trail"][0]["attempt"] == 1


def test_expected_incremental_value_overwritten_from_arm_net_values():
    """Test 7: Numerical expected_incremental_value MUST come from Python/state, not LLM."""
    state = create_sample_state()

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    mock_structured.invoke.return_value = LLMDecision(
        decision="WAIT",
        confidence=0.85,
        reasoning="Natural recovery is sufficient.",
        risk_level="low",
    )

    update = reasoning_node(state, llm=mock_llm)
    assert update["llm_decision"]["expected_incremental_value"] == state["arm_net_values"]["WAIT"]
    assert update["llm_decision"]["expected_incremental_value"] == 200.0


def test_llm_decision_outside_permitted_actions_triggers_retry():
    """Test 8 & 9: Decision outside permitted_actions triggers Attempt 2 retry."""
    state = create_sample_state()
    # Permit only WAIT and RETRY
    state["permitted_actions"] = ["WAIT", "RETRY"]

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    # Attempt 1: returns ESCALATE (not in permitted_actions)
    # Attempt 2: returns RETRY (permitted)
    mock_structured.invoke.side_effect = [
        LLMDecision(
            decision="ESCALATE",
            confidence=0.9,
            reasoning="Try escalation",
            risk_level="high",
        ),
        LLMDecision(
            decision="RETRY",
            confidence=0.85,
            reasoning="Corrected to RETRY within permitted actions",
            risk_level="low",
        ),
    ]

    update = reasoning_node(state, llm=mock_llm)

    assert update["final_action"] == "RETRY"
    assert update["llm_decision"]["decision"] == "RETRY"
    assert update["llm_decision"]["decision_source"] == "llm_retry_success"
    assert update["llm_decision"]["expected_incremental_value"] == 595.0
    assert mock_structured.invoke.call_count == 2
    assert update["audit_trail"][0]["attempt"] == 2


def test_invalid_response_twice_triggers_deterministic_fallback():
    """Test 10: When LLM fails both attempts, trigger deterministic argmax fallback."""
    state = create_sample_state()
    state["permitted_actions"] = ["WAIT", "RETRY"]

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    # Attempt 1 & 2 both raise exceptions (e.g. rate limit, validation error)
    mock_structured.invoke.side_effect = [
        RuntimeError("Foundry timeout"),
        ValueError("Pydantic validation failure"),
    ]

    update = reasoning_node(state, llm=mock_llm)

    # Permitted actions are ["WAIT", "RETRY"]
    # Net values: WAIT=200.0, RETRY=595.0 -> argmax is RETRY
    assert update["final_action"] == "RETRY"
    assert update["llm_decision"]["decision"] == "RETRY"
    assert update["llm_decision"]["decision_source"] == "fallback_no_llm"
    assert update["llm_decision"]["reasoning"] == "LLM fallback — using argmax directly"
    assert update["llm_decision"]["expected_incremental_value"] == 595.0
    assert update["audit_trail"][0]["status"] == "fallback"
    assert update["audit_trail"][0]["decision_source"] == "fallback_no_llm"


def test_reasoning_node_skipped_on_prior_error():
    """Test 11: Error route bypasses reasoning node."""
    state = create_sample_state()
    state["error"] = "Missing payment record"

    mock_llm = MagicMock()
    update = reasoning_node(state, llm=mock_llm)

    mock_llm.with_structured_output.assert_not_called()
    assert update["final_action"] == "WAIT"
    assert update["audit_trail"][0]["status"] == "skipped"
