"""
decision_engine/test_graph.py
=============================
Unit and end-to-end tests for LangGraph state and graph flow.
Includes the 3 required demonstration cases (Case A, Case B, Case C).
All tests mock external services and run completely offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pandas as pd
import pytest

from decision_engine.state import RecoveryState
from decision_engine.context_node import context_node
from decision_engine.estimation_node import estimation_node
from decision_engine.reasoning_node import LLMDecision
from decision_engine.graph import create_recovery_graph


# ── Fixtures & Mock Helpers ──────────────────────────────────────────────────

def make_mock_policy():
    """Mock CausalUpliftPolicy to return deterministic arm probabilities."""
    mock_policy = MagicMock()
    mock_t_learner = MagicMock()
    mock_policy.t_learner = mock_t_learner

    def fake_predict_proba(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [{"WAIT": 0.25, "RETRY": 0.65, "RETRY_NUDGE": 0.80, "ESCALATE": 0.85}],
            index=df.index,
        )

    mock_t_learner.predict_proba.side_effect = fake_predict_proba
    return mock_policy


def make_mock_llm(decisions: list[LLMDecision] | None = None, raise_errors: list[Exception] | None = None):
    """Mock LangChain chat model with structured output support."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    if raise_errors:
        mock_structured.invoke.side_effect = raise_errors
    elif decisions:
        mock_structured.invoke.side_effect = decisions
    else:
        mock_structured.invoke.return_value = LLMDecision(
            decision="RETRY_NUDGE",
            confidence=0.90,
            reasoning="High probability and positive incremental uplift.",
            risk_level="low",
        )

    return mock_llm


# ── Individual Component Tests ───────────────────────────────────────────────

def test_recovery_state_structure():
    """Test 1: Validate RecoveryState fields and typing contract."""
    state: RecoveryState = {
        "payment_id": "pay_test",
        "observable_features": {},
        "payment_context": {},
        "customer_history": {},
        "arm_probabilities": {},
        "arm_net_values": {},
        "permitted_actions": [],
        "llm_decision": {},
        "guardrail_result": {},
        "final_action": "WAIT",
        "error": None,
        "audit_trail": [{"node": "init", "status": "ok"}],
    }
    assert state["payment_id"] == "pay_test"
    assert isinstance(state["audit_trail"], list)


def test_context_node_success_with_seed777_payment():
    """Test 2: Context node successfully loads real Seed-777 payment data."""
    state: RecoveryState = {"payment_id": "pay_000001_a1", "audit_trail": []}
    update = context_node(state)

    assert update["error"] is None
    assert update["payment_id"] == "pay_000001_a1"
    assert "amount" in update["observable_features"]
    assert "payment_method" in update["observable_features"]
    assert "status" in update["payment_context"]
    assert "customer_id" in update["customer_history"]
    # Verify strict separation
    assert "payment_context" not in update["observable_features"]
    assert len(update["audit_trail"]) == 1
    assert update["audit_trail"][0]["status"] == "success"


def test_context_node_invalid_payment_id():
    """Test 3: Context node sets state['error'] on missing or unknown payment_id without crashing."""
    # Case: Empty string
    state_empty: RecoveryState = {"payment_id": "", "audit_trail": []}
    up1 = context_node(state_empty)
    assert up1["error"] is not None
    assert up1["audit_trail"][0]["status"] == "error"

    # Case: Unknown payment ID
    state_unknown: RecoveryState = {"payment_id": "pay_non_existent_999999", "audit_trail": []}
    up2 = context_node(state_unknown)
    assert up2["error"] is not None
    assert "not found" in up2["error"].lower()
    assert up2["audit_trail"][0]["status"] == "error"


def test_estimation_node_success():
    """Test 4: Estimation node computes probabilities and net values via policy."""
    mock_policy = make_mock_policy()
    state: RecoveryState = {
        "payment_id": "pay_test",
        "observable_features": {
            "amount": 500.0,
            "attempt_number": 1,
            "dynamic_success_rate": 0.5,
            "cumulative_failures": 0,
            "consecutive_failed_cycles": 0,
            "notification_engagement_score": 0.7,
            "contact_response_score": 0.7,
            "payment_method": "card",
            "failure_reason": "network_error",
        },
        "payment_context": {"status": "failed", "retry_count_current_cycle": 0},
        "audit_trail": [],
    }

    update = estimation_node(state, policy=mock_policy)

    assert "WAIT" in update["arm_probabilities"]
    assert "RETRY" in update["arm_probabilities"]
    assert "RETRY_NUDGE" in update["arm_probabilities"]
    assert "ESCALATE" in update["arm_probabilities"]

    assert "WAIT" in update["arm_net_values"]
    assert "RETRY" in update["arm_net_values"]
    assert "RETRY_NUDGE" in update["arm_net_values"]
    assert "ESCALATE" in update["arm_net_values"]

    assert "WAIT" in update["permitted_actions"]
    assert len(update["audit_trail"]) == 1
    assert update["audit_trail"][0]["status"] == "success"


def test_estimation_node_skipped_when_error_exists():
    """Test 5: Estimation node preserves error and skips calling policy."""
    mock_policy = make_mock_policy()
    state: RecoveryState = {
        "payment_id": "invalid",
        "error": "Payment ID not found",
        "audit_trail": [],
    }

    update = estimation_node(state, policy=mock_policy)

    mock_policy.t_learner.predict_proba.assert_not_called()
    assert update["audit_trail"][0]["status"] == "skipped"


def test_graph_compiles():
    """Test 13: LangGraph compiles without syntax or schema errors."""
    graph = create_recovery_graph(policy=make_mock_policy(), llm=make_mock_llm())
    assert graph is not None


def test_audit_trail_preserves_entries_across_nodes():
    """Test 12: Ensure operator.add reducer accumulates audit events across nodes."""
    mock_policy = make_mock_policy()
    mock_llm = make_mock_llm()
    graph = create_recovery_graph(policy=mock_policy, llm=mock_llm)

    init_state: RecoveryState = {
        "payment_id": "pay_000001_a1",
        "audit_trail": [{"node": "test_runner", "status": "start"}],
    }

    final_state = graph.invoke(init_state)

    nodes_in_trail = [e["node"] for e in final_state["audit_trail"]]
    assert "test_runner" in nodes_in_trail
    assert "context_node" in nodes_in_trail
    assert "estimation_node" in nodes_in_trail
    assert "reasoning_node" in nodes_in_trail
    assert len(final_state["audit_trail"]) >= 4


def test_graph_determinism():
    """Test Determinism: Repeated execution with identical state and mock output produces identical results."""
    mock_policy = make_mock_policy()
    mock_llm = make_mock_llm()
    graph = create_recovery_graph(policy=mock_policy, llm=mock_llm)

    state1 = graph.invoke({"payment_id": "pay_000001_a1", "audit_trail": []})
    state2 = graph.invoke({"payment_id": "pay_000001_a1", "audit_trail": []})

    assert state1["final_action"] == state2["final_action"]
    assert state1["arm_probabilities"] == state2["arm_probabilities"]
    assert state1["arm_net_values"] == state2["arm_net_values"]
    assert state1["llm_decision"]["expected_incremental_value"] == state2["llm_decision"]["expected_incremental_value"]


# ── THREE REQUIRED END-TO-END DEMONSTRATION CASES ────────────────────────────

def test_case_a_normal_payment_end_to_end():
    """
    CASE A — NORMAL PAYMENT:
    - Valid Seed-777 payment
    - context succeeds
    - estimation succeeds
    - reasoning node calls mocked LLM
    - structured result is produced
    - expected_incremental_value is populated from arm_net_values
    - audit_trail contains node events
    """
    mock_policy = make_mock_policy()
    mock_llm = make_mock_llm(
        decisions=[
            LLMDecision(
                decision="RETRY_NUDGE",
                confidence=0.88,
                reasoning="Moderate amount with high engagement favors RETRY_NUDGE.",
                risk_level="low",
            )
        ]
    )

    graph = create_recovery_graph(policy=mock_policy, llm=mock_llm)
    input_state: RecoveryState = {
        "payment_id": "pay_000001_a1",
        "audit_trail": [],
    }

    result = graph.invoke(input_state)

    # Verifications
    assert result.get("error") is None
    assert result["payment_id"] == "pay_000001_a1"
    assert result["final_action"] == "RETRY_NUDGE"
    assert result["llm_decision"]["decision"] == "RETRY_NUDGE"
    assert result["llm_decision"]["confidence"] == 0.88
    assert result["llm_decision"]["decision_source"] == "llm"
    # Must match python arm_net_values
    assert result["llm_decision"]["expected_incremental_value"] == result["arm_net_values"]["RETRY_NUDGE"]

    # Audit events
    event_nodes = [ev["node"] for ev in result["audit_trail"]]
    assert "context_node" in event_nodes
    assert "estimation_node" in event_nodes
    assert "reasoning_node" in event_nodes


def test_case_b_llm_fails_twice_end_to_end():
    """
    CASE B — LLM FAILS TWICE:
    - Mocked reasoning model raises errors twice
    - First attempt fails, second attempt fails
    - fallback_no_llm is set
    - fallback decision = argmax over permitted arm_net_values
    - expected_incremental_value comes from arm_net_values
    - audit_trail records the fallback
    - graph does not crash
    """
    mock_policy = make_mock_policy()
    mock_llm = make_mock_llm(
        raise_errors=[
            RuntimeError("Foundry service unavailable (attempt 1)"),
            RuntimeError("Rate limit / malformed JSON (attempt 2)"),
        ]
    )

    graph = create_recovery_graph(policy=mock_policy, llm=mock_llm)
    input_state: RecoveryState = {
        "payment_id": "pay_000001_a1",
        "audit_trail": [],
    }

    result = graph.invoke(input_state)

    # Verifications
    assert result.get("error") is None
    assert result["llm_decision"]["decision_source"] == "fallback_no_llm"
    assert result["llm_decision"]["reasoning"] == "LLM fallback — using argmax directly"

    # Argmax over arm_net_values
    expected_argmax = max(result["permitted_actions"], key=lambda a: result["arm_net_values"][a])
    assert result["final_action"] == expected_argmax
    assert result["llm_decision"]["decision"] == expected_argmax
    assert result["llm_decision"]["expected_incremental_value"] == result["arm_net_values"][expected_argmax]

    # Audit check
    fallback_events = [ev for ev in result["audit_trail"] if ev.get("status") == "fallback"]
    assert len(fallback_events) >= 1
    assert fallback_events[0]["node"] == "reasoning_node"


def test_case_c_malformed_or_missing_payment_id_end_to_end():
    """
    CASE C — MALFORMED / MISSING PAYMENT ID:
    - Invalid or missing payment_id
    - context_node sets state["error"]
    - estimation_node does NOT invoke CausalUpliftPolicy
    - reasoning_node is skipped
    - final_action = WAIT
    - audit_trail explains the error
    - graph completes without crashing
    """
    mock_policy = make_mock_policy()
    mock_llm = make_mock_llm()

    graph = create_recovery_graph(policy=mock_policy, llm=mock_llm)
    input_state: RecoveryState = {
        "payment_id": "non_existent_payment_xyz",
        "audit_trail": [],
    }

    result = graph.invoke(input_state)

    # Verifications
    assert result["error"] is not None
    assert "not found" in result["error"].lower()
    mock_policy.t_learner.predict_proba.assert_not_called()
    mock_llm.with_structured_output.assert_not_called()

    # Final action defaulted to safe WAIT
    assert result["final_action"] == "WAIT"

    # Audit events
    event_nodes = [ev["node"] for ev in result["audit_trail"]]
    assert "context_node" in event_nodes
    assert "estimation_node" in event_nodes
    assert "error_fallback" in event_nodes
    assert "reasoning_node" not in event_nodes
