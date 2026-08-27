"""
decision_engine/test_graph.py
=============================
Unit and end-to-end tests for LangGraph state and graph flow.
Includes the 3 required demonstration cases (Case A, Case B, Case C)
with full reasoning -> guardrail -> execution -> SQLite audit persistence.
All tests mock external services and run completely offline.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock
import pandas as pd
import pytest

from decision_engine.state import RecoveryState
from decision_engine.context_node import context_node
from decision_engine.estimation_node import estimation_node
from decision_engine.reasoning_node import LLMDecision
from decision_engine.graph import create_recovery_graph
from decision_engine.audit import (
    init_audit_db,
    get_audit_record,
    get_audit_row_count,
    get_audit_records_count_for_payment,
)


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


def test_graph_compiles(tmp_path):
    """Test 13: LangGraph compiles without syntax or schema errors."""
    test_db = tmp_path / "test_compile.db"
    graph = create_recovery_graph(
        policy=make_mock_policy(),
        llm=make_mock_llm(),
        db_path=test_db,
    )
    assert graph is not None


def test_audit_trail_preserves_entries_across_nodes(tmp_path):
    """Test 12: Ensure operator.add reducer accumulates audit events across nodes."""
    test_db = tmp_path / "test_trail.db"
    mock_policy = make_mock_policy()
    mock_llm = make_mock_llm()
    graph = create_recovery_graph(policy=mock_policy, llm=mock_llm, db_path=test_db)

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
    assert "guardrail_node" in nodes_in_trail
    assert "execution_node" in nodes_in_trail
    assert len(final_state["audit_trail"]) >= 6


def test_graph_determinism(tmp_path):
    """Test Determinism: Repeated execution with identical state and mock output produces identical results."""
    test_db = tmp_path / "test_det.db"
    mock_policy = make_mock_policy()
    mock_llm = make_mock_llm()
    graph = create_recovery_graph(policy=mock_policy, llm=mock_llm, db_path=test_db)

    state1 = graph.invoke({"payment_id": "pay_000001_a1", "audit_trail": []})
    state2 = graph.invoke({"payment_id": "pay_000001_a1", "audit_trail": []})

    assert state1["final_action"] == state2["final_action"]
    assert state1["arm_probabilities"] == state2["arm_probabilities"]
    assert state1["arm_net_values"] == state2["arm_net_values"]
    assert state1["llm_decision"]["expected_incremental_value"] == state2["llm_decision"]["expected_incremental_value"]
    assert state1["guardrail_result"] == state2["guardrail_result"]


# ── THREE REQUIRED END-TO-END DEMONSTRATION CASES ────────────────────────────

def test_case_a_normal_payment_end_to_end(tmp_path):
    """
    CASE A — NORMAL PAYMENT:
    - Valid Seed-777 payment
    - context succeeds
    - estimation succeeds
    - reasoning node calls mocked LLM
    - guardrail evaluates and PASSES
    - execution node runs and persists to SQLite
    - expected_incremental_value is populated from arm_net_values
    - audit_trail contains all node events
    """
    test_db = tmp_path / "case_a.db"
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

    graph = create_recovery_graph(policy=mock_policy, llm=mock_llm, db_path=test_db)
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
    assert result["llm_decision"]["expected_incremental_value"] == result["arm_net_values"]["RETRY_NUDGE"]

    # Guardrail passed
    assert result["guardrail_result"]["status"] == "passed"
    assert result["guardrail_result"]["overridden"] is False
    assert result["guardrail_result"]["final_action"] == "RETRY_NUDGE"

    # Audit events
    event_nodes = [ev["node"] for ev in result["audit_trail"]]
    assert "context_node" in event_nodes
    assert "estimation_node" in event_nodes
    assert "reasoning_node" in event_nodes
    assert "guardrail_node" in event_nodes
    assert "execution_node" in event_nodes

    # SQLite DB record
    rec = get_audit_record("pay_000001_a1", db_path=test_db)
    assert rec is not None
    assert rec["payment_id"] == "pay_000001_a1"
    assert rec["final_action"] == "RETRY_NUDGE"
    assert rec["llm_proposed_decision"] == "RETRY_NUDGE"
    assert rec["guardrail_verdict"] == "passed"
    assert rec["expected_incremental_value"] == result["arm_net_values"]["RETRY_NUDGE"]


def test_case_b_guardrail_override_end_to_end(tmp_path):
    """
    CASE B — GUARDRAIL OVERRIDE:
    - LLM proposes ESCALATE
    - Customer history has lifetime_escalations >= 1
    - Guardrail OVERRIDES ESCALATE to WAIT
    - Final action is WAIT
    - Override reason is preserved and recorded in SQLite
    """
    test_db = tmp_path / "case_b.db"
    mock_policy = make_mock_policy()
    mock_llm = make_mock_llm(
        decisions=[
            LLMDecision(
                decision="ESCALATE",
                confidence=0.95,
                reasoning="High amount transaction, recommending immediate escalation.",
                risk_level="high",
            )
        ]
    )

    graph = create_recovery_graph(policy=mock_policy, llm=mock_llm, db_path=test_db)
    # Inject customer_history with lifetime_escalations = 1 to trigger Guardrail 3
    input_state: RecoveryState = {
        "payment_id": "pay_000001_a1",
        "customer_history": {"lifetime_escalations": 1, "customer_id": "cust_000001"},
        "payment_context": {"status": "failed", "retry_count_current_cycle": 0},
        "audit_trail": [],
    }

    result = graph.invoke(input_state)

    # Verifications
    assert result["llm_decision"]["decision"] == "ESCALATE"
    assert result["guardrail_result"]["status"] == "overridden"
    assert result["guardrail_result"]["overridden"] is True
    assert result["guardrail_result"]["proposed_action"] == "ESCALATE"
    assert result["guardrail_result"]["final_action"] == "WAIT"
    assert "escalation" in result["guardrail_result"]["reason"].lower()
    assert result["final_action"] == "WAIT"

    # SQLite DB record verification
    rec = get_audit_record("pay_000001_a1", db_path=test_db)
    assert rec is not None
    assert rec["llm_proposed_decision"] == "ESCALATE"
    assert rec["final_action"] == "WAIT"
    assert rec["guardrail_verdict"] == "overridden"
    assert "escalation" in rec["guardrail_reason"].lower()


def test_case_c_malformed_or_missing_payment_id_end_to_end(tmp_path):
    """
    CASE C — MALFORMED / MISSING PAYMENT ID:
    - Invalid or missing payment_id
    - context_node sets state["error"]
    - estimation_node skips CausalUpliftPolicy
    - reasoning_node and guardrail_node are bypassed
    - execution_node runs and persists error record
    - final_action = WAIT
    """
    test_db = tmp_path / "case_c.db"
    mock_policy = make_mock_policy()
    mock_llm = make_mock_llm()

    graph = create_recovery_graph(policy=mock_policy, llm=mock_llm, db_path=test_db)
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
    assert "execution_node" in event_nodes
    assert "reasoning_node" not in event_nodes
    assert "guardrail_node" not in event_nodes

    # SQLite DB record verification
    rec = get_audit_record("non_existent_payment_xyz", db_path=test_db)
    assert rec is not None
    assert rec["error_status"] is not None
    assert rec["final_action"] == "WAIT"
    assert rec["llm_proposed_decision"] == "N/A — error path"
    assert rec["guardrail_verdict"] == "N/A — error path"
