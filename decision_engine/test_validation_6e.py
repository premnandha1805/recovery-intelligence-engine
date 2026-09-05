"""
decision_engine/test_validation_6e.py
=====================================
Day 6E Final Decision Engine Validation & Regression Suite.

Exhaustively verifies:
1. Guardrail escalation-cap override and preservation through guardrail_node.
2. LLM structured-output Pydantic validation and value population.
3. Invalid LLM response attempt 1 -> retry attempt 2 success.
4. Invalid LLM response twice -> deterministic argmax fallback.
5. Out-of-permitted-actions rejection, correction retry, and fallback.
6. End-to-end LangGraph deterministic reproducibility.
7. Error path short-circuit and execution persistence.
8. SQLite UPSERT strict uniqueness constraint via direct SQL queries.

All tests are 100% offline, deterministic, and use isolated mocks/temporary databases.
"""

from __future__ import annotations

import pathlib
import sqlite3
from unittest.mock import MagicMock
import pandas as pd
import pytest

from models.schemas import Action, Decision
from decision_engine.guardrails import (
    MAX_LIFETIME_ESCALATIONS,
    check,
    apply_guardrails,
)
from decision_engine.guardrail_node import guardrail_node
from decision_engine.reasoning_node import (
    LLMDecision,
    reasoning_node,
)
from decision_engine.state import RecoveryState
from decision_engine.graph import create_recovery_graph
from decision_engine.audit import (
    get_audit_record,
    get_audit_row_count,
    get_audit_records_count_for_payment,
)


# ── Mock Helpers ─────────────────────────────────────────────────────────────

def _make_mock_policy():
    """Create a mock CausalUpliftPolicy returning fixed deterministic arm probabilities."""
    mock_policy = MagicMock()
    mock_t_learner = MagicMock()
    mock_policy.t_learner = mock_t_learner

    def fake_predict_proba(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [{"WAIT": 0.20, "RETRY": 0.60, "RETRY_NUDGE": 0.75, "ESCALATE": 0.85}],
            index=df.index,
        )

    mock_t_learner.predict_proba.side_effect = fake_predict_proba
    return mock_policy


# ── 1. Guardrail Override Test (Escalation Cap) ──────────────────────────────

def test_guardrail_escalation_cap_override_and_preservation():
    """
    Day 6E Section 2:
    Given: proposed_action = Action.ESCALATE and lifetime_escalations >= MAX_LIFETIME_ESCALATIONS
    Assert:
      - result.overridden is True
      - result.final_action != Action.ESCALATE (overridden to Action.WAIT)
      - result.reason contains 'escalation'
      - override is preserved when flowing through guardrail_node.
    """
    # 1. Test actual guardrails.check / apply_guardrails interface
    result = check(
        Action.ESCALATE,
        {"status": "failed", "retry_count_current_cycle": 0},
        {"lifetime_escalations": MAX_LIFETIME_ESCALATIONS},
    )

    assert result.overridden is True
    assert result.final_action != Action.ESCALATE
    assert result.final_action == Action.WAIT
    assert "escalation" in result.reason.lower()

    # 2. Test preservation through guardrail_node
    state: RecoveryState = {
        "payment_id": "pay_esc_test",
        "llm_decision": {
            "decision": "ESCALATE",
            "confidence": 0.95,
            "reasoning": "Account requires high-touch outreach.",
            "risk_level": "high",
        },
        "payment_context": {"status": "failed", "retry_count_current_cycle": 0},
        "customer_history": {"lifetime_escalations": MAX_LIFETIME_ESCALATIONS},
        "audit_trail": [],
    }

    node_output = guardrail_node(state)

    assert node_output["final_action"] == "WAIT"
    assert node_output["guardrail_result"]["status"] == "overridden"
    assert node_output["guardrail_result"]["overridden"] is True
    assert node_output["guardrail_result"]["proposed_action"] == "ESCALATE"
    assert node_output["guardrail_result"]["final_action"] == "WAIT"
    assert "escalation" in node_output["guardrail_result"]["reason"].lower()

    # Audit event verification
    trail = node_output["audit_trail"]
    assert len(trail) == 1
    assert trail[0]["node"] == "guardrail_node"
    assert trail[0]["status"] == "overridden"
    assert trail[0]["proposed_action"] == "ESCALATE"
    assert trail[0]["final_action"] == "WAIT"


# ── 2. LLM Structured-Output Validation Tests ────────────────────────────────

def test_llm_structured_output_validation():
    """
    Day 6E Section 3:
    Mock LLM returning valid structured output.
    Verify:
      - Pydantic schema validation succeeds
      - Decision is in permitted_actions
      - Confidence is in [0.0, 1.0]
      - expected_incremental_value is NOT taken from the LLM
      - expected_incremental_value is populated from state['arm_net_values'][decision]
      - decision_source is 'llm'
    """
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    mock_structured.invoke.return_value = LLMDecision(
        decision="RETRY_NUDGE",
        confidence=0.92,
        reasoning="High uplift net value justifies nudge.",
        risk_level="low",
    )

    state: RecoveryState = {
        "payment_id": "pay_struct_01",
        "observable_features": {"amount": 1000.0, "dynamic_success_rate": 0.6},
        "arm_probabilities": {"WAIT": 0.20, "RETRY": 0.60, "RETRY_NUDGE": 0.75, "ESCALATE": 0.85},
        "arm_net_values": {"WAIT": 200.0, "RETRY": 580.0, "RETRY_NUDGE": 715.0, "ESCALATE": 510.0},
        "permitted_actions": ["WAIT", "RETRY", "RETRY_NUDGE"],
        "audit_trail": [],
    }

    result = reasoning_node(state, llm=mock_llm)

    assert result["final_action"] == "RETRY_NUDGE"
    assert result["llm_decision"]["decision"] == "RETRY_NUDGE"
    assert result["llm_decision"]["decision"] in state["permitted_actions"]
    assert 0.0 <= result["llm_decision"]["confidence"] <= 1.0
    assert result["llm_decision"]["confidence"] == 0.92
    assert result["llm_decision"]["decision_source"] == "llm"
    # Strict verification: expected_incremental_value taken from arm_net_values, not LLM
    assert result["llm_decision"]["expected_incremental_value"] == 715.0
    assert result["llm_decision"]["expected_incremental_value"] == state["arm_net_values"]["RETRY_NUDGE"]

    # Audit trail verifies Attempt 1 success
    assert result["audit_trail"][0]["status"] == "success"
    assert result["audit_trail"][0]["attempt"] == 1


# ── 3. Invalid-Response -> Retry Test ────────────────────────────────────────

def test_invalid_response_triggers_retry_success():
    """
    Day 6E Section 4:
    Attempt 1: raises a validation/structured-output error
    Attempt 2: returns a valid structured response
    Assert:
      - Exactly two LLM invocations occurred
      - Retry happened
      - Final result comes from Attempt 2
      - No fallback_no_llm source recorded
      - expected_incremental_value comes from arm_net_values
      - audit_trail records the retry success
    """
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    # Attempt 1 fails, Attempt 2 succeeds
    mock_structured.invoke.side_effect = [
        ValueError("Pydantic JSON validation failed: missing field 'reasoning'"),
        LLMDecision(
            decision="RETRY",
            confidence=0.85,
            reasoning="Attempt 2 valid decision after error correction.",
            risk_level="low",
        ),
    ]

    state: RecoveryState = {
        "payment_id": "pay_retry_01",
        "observable_features": {"amount": 500.0},
        "arm_probabilities": {"WAIT": 0.15, "RETRY": 0.70},
        "arm_net_values": {"WAIT": 75.0, "RETRY": 340.0},
        "permitted_actions": ["WAIT", "RETRY"],
        "audit_trail": [],
    }

    result = reasoning_node(state, llm=mock_llm)

    # Exactly 2 invocations occurred
    assert mock_structured.invoke.call_count == 2

    # Final action from Attempt 2
    assert result["final_action"] == "RETRY"
    assert result["llm_decision"]["decision"] == "RETRY"
    assert result["llm_decision"]["decision_source"] == "llm_retry_success"
    assert result["llm_decision"]["decision_source"] != "fallback_no_llm"
    assert result["llm_decision"]["expected_incremental_value"] == 340.0

    # Audit trail verifies Attempt 2 success
    assert result["audit_trail"][0]["status"] == "success"
    assert result["audit_trail"][0]["attempt"] == 2
    assert result["audit_trail"][0]["decision_source"] == "llm_retry_success"


# ── 4. Invalid-Response Twice -> Fallback Test ────────────────────────────────

def test_invalid_response_twice_triggers_deterministic_fallback():
    """
    Day 6E Section 5:
    Mock LLM fails structured validation twice.
    Assert:
      - Exactly two LLM invocations occurred
      - No third attempt
      - decision_source == 'fallback_no_llm'
      - Selected action == deterministic argmax over permitted arm_net_values
      - expected_incremental_value equals selected arm's arm_net_values entry
      - Fallback reasoning and event recorded in audit_trail
      - Pipeline does not raise an exception
    """
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    # Fails both attempt 1 and attempt 2
    mock_structured.invoke.side_effect = [
        RuntimeError("Foundry gateway timeout (attempt 1)"),
        RuntimeError("Foundry rate limit exceeded (attempt 2)"),
    ]

    state: RecoveryState = {
        "payment_id": "pay_fallback_01",
        "observable_features": {"amount": 800.0},
        "arm_probabilities": {"WAIT": 0.20, "RETRY": 0.60, "RETRY_NUDGE": 0.80, "ESCALATE": 0.70},
        "arm_net_values": {"WAIT": 160.0, "RETRY": 460.0, "RETRY_NUDGE": 615.0, "ESCALATE": 390.0},
        "permitted_actions": ["WAIT", "RETRY", "RETRY_NUDGE"],
        "audit_trail": [],
    }

    result = reasoning_node(state, llm=mock_llm)

    # Exactly 2 invocations
    assert mock_structured.invoke.call_count == 2

    # Deterministic argmax over permitted actions
    # Among ["WAIT": 160, "RETRY": 460, "RETRY_NUDGE": 615], argmax is RETRY_NUDGE
    assert result["final_action"] == "RETRY_NUDGE"
    assert result["llm_decision"]["decision"] == "RETRY_NUDGE"
    assert result["llm_decision"]["decision_source"] == "fallback_no_llm"
    assert result["llm_decision"]["expected_incremental_value"] == 615.0
    assert "fallback" in result["llm_decision"]["reasoning"].lower()

    # Audit trail verifies fallback event
    assert result["audit_trail"][0]["status"] == "fallback"
    assert result["audit_trail"][0]["decision_source"] == "fallback_no_llm"
    assert result["audit_trail"][0]["fallback_action"] == "RETRY_NUDGE"


# ── 5. Out-of-Permitted-Actions Test ──────────────────────────────────────────

def test_out_of_permitted_actions_rejection_and_retry():
    """
    Day 6E Section 6:
    Part A: Attempt 1 returns an action not in permitted_actions.
            Attempt 2 returns a valid permitted action.
    Assert:
      - Attempt 1 rejected despite syntactically valid JSON
      - Correction retry occurs
      - Attempt 2 succeeds
    """
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    # Permitted actions are WAIT, RETRY.
    # Attempt 1 returns ESCALATE (not permitted).
    # Attempt 2 returns RETRY (permitted).
    mock_structured.invoke.side_effect = [
        LLMDecision(
            decision="ESCALATE",
            confidence=0.90,
            reasoning="Recommending escalate outside permitted set.",
            risk_level="high",
        ),
        LLMDecision(
            decision="RETRY",
            confidence=0.85,
            reasoning="Corrected to permitted action RETRY.",
            risk_level="low",
        ),
    ]

    state: RecoveryState = {
        "payment_id": "pay_unpermitted_01",
        "arm_probabilities": {"WAIT": 0.2, "RETRY": 0.7, "ESCALATE": 0.9},
        "arm_net_values": {"WAIT": 20.0, "RETRY": 65.0, "ESCALATE": 85.0},
        "permitted_actions": ["WAIT", "RETRY"],  # ESCALATE is not permitted!
        "audit_trail": [],
    }

    result = reasoning_node(state, llm=mock_llm)

    assert mock_structured.invoke.call_count == 2
    assert result["final_action"] == "RETRY"
    assert result["llm_decision"]["decision"] == "RETRY"
    assert result["llm_decision"]["decision_source"] == "llm_retry_success"
    assert result["llm_decision"]["expected_incremental_value"] == 65.0


def test_out_of_permitted_actions_both_attempts_triggers_fallback():
    """
    Day 6E Section 6:
    Part B: Both Attempt 1 and Attempt 2 return unpermitted actions.
    Assert:
      - Fallback occurs
      - Fallback chooses argmax strictly from permitted_actions
      - Unpermitted action is NEVER selected
    """
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    mock_structured.invoke.side_effect = [
        LLMDecision(
            decision="ESCALATE",
            confidence=0.95,
            reasoning="Attempt 1 unpermitted.",
            risk_level="high",
        ),
        LLMDecision(
            decision="ESCALATE",
            confidence=0.95,
            reasoning="Attempt 2 still unpermitted.",
            risk_level="high",
        ),
    ]

    state: RecoveryState = {
        "payment_id": "pay_unpermitted_02",
        "arm_probabilities": {"WAIT": 0.2, "RETRY": 0.7, "ESCALATE": 0.9},
        "arm_net_values": {"WAIT": 20.0, "RETRY": 65.0, "ESCALATE": 999.0},
        "permitted_actions": ["WAIT", "RETRY"],  # ESCALATE is forbidden
        "audit_trail": [],
    }

    result = reasoning_node(state, llm=mock_llm)

    assert mock_structured.invoke.call_count == 2
    assert result["llm_decision"]["decision_source"] == "fallback_no_llm"
    # Even though ESCALATE had 999.0 net value, it MUST NOT be chosen because it's not permitted!
    assert result["final_action"] == "RETRY"
    assert result["final_action"] in state["permitted_actions"]
    assert result["final_action"] != "ESCALATE"
    assert result["llm_decision"]["expected_incremental_value"] == 65.0


# ── 6. End-to-End Graph Determinism ──────────────────────────────────────────

def test_end_to_end_graph_determinism_strict(tmp_path):
    """
    Day 6E Section 7:
    Run the full graph twice with identical inputs and fixed mock outputs.
    Assert deterministic equality for:
      - arm_probabilities
      - arm_net_values
      - permitted_actions
      - llm_decision (excluding runtime timestamps)
      - guardrail_result
      - final_action
      - semantic sequence of node events in audit_trail
    """
    test_db1 = tmp_path / "det1.db"
    test_db2 = tmp_path / "det2.db"

    mock_policy = _make_mock_policy()
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.return_value = LLMDecision(
        decision="RETRY_NUDGE",
        confidence=0.90,
        reasoning="Deterministic choice based on optimal net value.",
        risk_level="low",
    )

    graph1 = create_recovery_graph(policy=mock_policy, llm=mock_llm, db_path=test_db1)
    graph2 = create_recovery_graph(policy=mock_policy, llm=mock_llm, db_path=test_db2)

    input_data = {
        "payment_id": "pay_000001_a1",
        "audit_trail": [],
    }

    state1 = graph1.invoke(input_data)
    state2 = graph2.invoke(input_data)

    # Deterministic equality across all decision-relevant fields
    assert state1["final_action"] == state2["final_action"]
    assert state1["arm_probabilities"] == state2["arm_probabilities"]
    assert state1["arm_net_values"] == state2["arm_net_values"]
    assert state1["permitted_actions"] == state2["permitted_actions"]
    assert state1["llm_decision"]["decision"] == state2["llm_decision"]["decision"]
    assert state1["llm_decision"]["confidence"] == state2["llm_decision"]["confidence"]
    assert state1["llm_decision"]["expected_incremental_value"] == state2["llm_decision"]["expected_incremental_value"]
    assert state1["guardrail_result"] == state2["guardrail_result"]

    # Semantic audit trail equivalence (nodes and statuses match in order)
    trail1 = [(ev["node"], ev.get("status")) for ev in state1["audit_trail"]]
    trail2 = [(ev["node"], ev.get("status")) for ev in state2["audit_trail"]]
    assert trail1 == trail2
    assert trail1 == [
        ("context_node", "success"),
        ("estimation_node", "success"),
        ("reasoning_node", "success"),
        ("guardrail_node", "passed"),
        ("execution_node", "executed"),
    ]


# ── 7. Error Path Regression Test ────────────────────────────────────────────

def test_error_path_regression_skips_reasoning_and_guardrails(tmp_path):
    """
    Day 6E Section 8:
    Verify malformed/missing payment_id case:
      context_node -> error -> estimation skipped -> reasoning skipped
      -> guardrail skipped -> execution/error audit -> final_action = WAIT
    Assert graph completes without raising an exception.
    """
    test_db = tmp_path / "err_reg.db"
    mock_policy = _make_mock_policy()
    mock_llm = MagicMock()

    graph = create_recovery_graph(policy=mock_policy, llm=mock_llm, db_path=test_db)

    # Invalid payment ID
    result = graph.invoke({"payment_id": "malformed_pid_empty_or_bad", "audit_trail": []})

    # Assertions
    assert result["error"] is not None
    assert "not found" in result["error"].lower()
    assert result["final_action"] == "WAIT"

    # Verify external models were never called
    mock_policy.t_learner.predict_proba.assert_not_called()
    mock_llm.with_structured_output.assert_not_called()

    # Verify node traversal
    trail_nodes = [ev["node"] for ev in result["audit_trail"]]
    assert "context_node" in trail_nodes
    assert "estimation_node" in trail_nodes
    assert "error_fallback" in trail_nodes
    assert "execution_node" in trail_nodes
    assert "reasoning_node" not in trail_nodes
    assert "guardrail_node" not in trail_nodes

    # Verify SQLite record captures error path
    rec = get_audit_record("malformed_pid_empty_or_bad", db_path=test_db)
    assert rec is not None
    assert rec["final_action"] == "WAIT"
    assert rec["error_status"] is not None
    assert rec["llm_proposed_decision"] == "N/A — error path"
    assert rec["guardrail_verdict"] == "N/A — error path"


# ── 8. SQLite UPSERT Regression via Direct Queries ────────────────────────────

def test_sqlite_upsert_regression_direct_queries(tmp_path):
    """
    Day 6E Section 9:
    Verify existing Day 6D audit behavior:
      1. Run graph for a payment_id.
      2. Query SQLite and confirm exactly 1 row.
      3. Run same payment_id again.
      4. Confirm there is still exactly 1 row.
      5. Confirm row was updated rather than duplicated.
    Uses direct SQLite queries.
    """
    db_path = tmp_path / "upsert_reg.db"
    mock_policy = _make_mock_policy()
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    # Return different decisions across the 2 runs to test in-place UPDATE
    mock_structured.invoke.side_effect = [
        LLMDecision(decision="RETRY", confidence=0.80, reasoning="Run 1", risk_level="low"),
        LLMDecision(decision="RETRY_NUDGE", confidence=0.95, reasoning="Run 2 updated", risk_level="medium"),
    ]

    graph = create_recovery_graph(policy=mock_policy, llm=mock_llm, db_path=db_path)
    payment_id = "pay_000001_a1"

    # --- Run 1 ---
    graph.invoke({"payment_id": payment_id, "audit_trail": []})

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM decision_audit WHERE payment_id = ?", (payment_id,))
        count1 = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM decision_audit")
        total1 = cur.fetchone()[0]
        cur.execute("SELECT payment_id, final_action, llm_confidence, timestamp FROM decision_audit WHERE payment_id = ?", (payment_id,))
        row1 = dict(cur.fetchone())

    assert count1 == 1
    assert total1 == 1
    assert row1["final_action"] == "RETRY"
    assert row1["llm_confidence"] == 0.80

    # --- Run 2 (Same payment_id) ---
    graph.invoke({"payment_id": payment_id, "audit_trail": []})

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM decision_audit WHERE payment_id = ?", (payment_id,))
        count2 = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM decision_audit")
        total2 = cur.fetchone()[0]
        cur.execute("SELECT payment_id, final_action, llm_confidence, timestamp FROM decision_audit WHERE payment_id = ?", (payment_id,))
        row2 = dict(cur.fetchone())

    # Strict UPSERT assertions: exactly 1 row, updated in-place
    assert count2 == 1
    assert total2 == 1
    assert row2["final_action"] == "RETRY_NUDGE"
    assert row2["llm_confidence"] == 0.95
    assert row2["payment_id"] == payment_id
