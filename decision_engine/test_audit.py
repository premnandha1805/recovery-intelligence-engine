"""
decision_engine/test_audit.py
=============================
Unit tests for SQLite audit persistence, UPSERT semantics, and guardrail/execution integration.
All tests run completely offline and use isolated temporary databases.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock
import pandas as pd
import pytest

from models.schemas import Action, Decision
from decision_engine.state import RecoveryState
from decision_engine.audit import (
    init_audit_db,
    save_decision_audit,
    get_audit_record,
    get_audit_row_count,
    get_audit_records_count_for_payment,
)
from decision_engine.guardrail_node import guardrail_node
from decision_engine.execution_node import execution_node
from decision_engine.reasoning_node import LLMDecision
from decision_engine.graph import create_recovery_graph


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


def make_mock_llm(decisions: list[LLMDecision] | None = None):
    """Mock LangChain chat model with structured output support."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    if decisions:
        mock_structured.invoke.side_effect = decisions
    else:
        mock_structured.invoke.return_value = LLMDecision(
            decision="RETRY_NUDGE",
            confidence=0.90,
            reasoning="High probability and positive incremental uplift.",
            risk_level="low",
        )

    return mock_llm


def test_sqlite_database_initialization(tmp_path):
    """Test 5: Verify SQLite database and table auto-initialization."""
    db_path = tmp_path / "test_init.db"
    assert not db_path.exists()

    init_audit_db(db_path)
    assert db_path.exists()

    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='decision_audit'")
        assert cur.fetchone() is not None


def test_decision_record_persistence(tmp_path):
    """Test 6: Verify decision record persistence and reuse of canonical Decision entity."""
    db_path = tmp_path / "test_persist.db"
    state: RecoveryState = {
        "payment_id": "pay_test_persist_01",
        "arm_probabilities": {"WAIT": 0.1, "RETRY": 0.7, "RETRY_NUDGE": 0.8, "ESCALATE": 0.6},
        "arm_net_values": {"WAIT": 10.0, "RETRY": 65.0, "RETRY_NUDGE": 80.0, "ESCALATE": 50.0},
        "llm_decision": {
            "decision": "RETRY_NUDGE",
            "confidence": 0.95,
            "reasoning": "Highest incremental net value",
            "risk_level": "low",
            "expected_incremental_value": 80.0,
            "decision_source": "llm",
        },
        "guardrail_result": {
            "status": "passed",
            "proposed_action": "RETRY_NUDGE",
            "final_action": "RETRY_NUDGE",
            "overridden": False,
            "reason": "Passed all rules",
        },
        "final_action": "RETRY_NUDGE",
        "error": None,
        "audit_trail": [],
    }

    decision_obj = save_decision_audit(state, db_path=db_path)

    # Reuses canonical Decision entity
    assert isinstance(decision_obj, Decision)
    assert decision_obj.payment_id == "pay_test_persist_01"
    assert decision_obj.action == Action.RETRY_NUDGE
    assert decision_obj.expected_incremental_value == 80.0
    assert decision_obj.model_version == "gpt-4.1-mini"

    # Database query verification
    rec = get_audit_record("pay_test_persist_01", db_path=db_path)
    assert rec is not None
    assert rec["payment_id"] == "pay_test_persist_01"
    assert rec["final_action"] == "RETRY_NUDGE"
    assert rec["llm_proposed_decision"] == "RETRY_NUDGE"
    assert rec["guardrail_verdict"] == "passed"
    assert rec["decision_source"] == "llm"


def test_upsert_behavior_no_duplicates(tmp_path):
    """Test 7 & 8: Running twice for same payment_id updates row with count(payment_id) == 1."""
    db_path = tmp_path / "test_upsert.db"
    payment_id = "pay_upsert_demo"

    # Run 1: Initial decision is RETRY
    state1: RecoveryState = {
        "payment_id": payment_id,
        "arm_probabilities": {"WAIT": 0.2, "RETRY": 0.8},
        "arm_net_values": {"WAIT": 20.0, "RETRY": 75.0},
        "llm_decision": {
            "decision": "RETRY",
            "confidence": 0.80,
            "reasoning": "First attempt",
            "risk_level": "low",
            "expected_incremental_value": 75.0,
            "decision_source": "llm",
        },
        "guardrail_result": {"status": "passed", "reason": "ok"},
        "final_action": "RETRY",
        "error": None,
        "audit_trail": [],
    }

    save_decision_audit(state1, db_path=db_path)

    assert get_audit_row_count(db_path=db_path) == 1
    assert get_audit_records_count_for_payment(payment_id, db_path=db_path) == 1
    rec1 = get_audit_record(payment_id, db_path=db_path)
    assert rec1["final_action"] == "RETRY"
    assert rec1["llm_confidence"] == 0.80

    # Run 2: Re-run with updated action RETRY_NUDGE
    state2: RecoveryState = {
        "payment_id": payment_id,
        "arm_probabilities": {"WAIT": 0.2, "RETRY": 0.8},
        "arm_net_values": {"WAIT": 20.0, "RETRY": 75.0},
        "llm_decision": {
            "decision": "RETRY_NUDGE",
            "confidence": 0.95,
            "reasoning": "Updated evaluation",
            "risk_level": "medium",
            "expected_incremental_value": 90.0,
            "decision_source": "llm",
        },
        "guardrail_result": {"status": "passed", "reason": "ok"},
        "final_action": "RETRY_NUDGE",
        "error": None,
        "audit_trail": [],
    }

    save_decision_audit(state2, db_path=db_path)

    # Must still have exactly 1 row in total, and 1 row for this payment_id
    assert get_audit_row_count(db_path=db_path) == 1
    assert get_audit_records_count_for_payment(payment_id, db_path=db_path) == 1
    rec2 = get_audit_record(payment_id, db_path=db_path)
    assert rec2["final_action"] == "RETRY_NUDGE"
    assert rec2["llm_confidence"] == 0.95


def test_guardrail_pass_and_override_nodes():
    """Test 1 & 2: Verify guardrail_node pass and override behavior."""
    # Pass case
    pass_state: RecoveryState = {
        "llm_decision": {"decision": "RETRY"},
        "payment_context": {"status": "failed", "retry_count_current_cycle": 0},
        "customer_history": {"lifetime_escalations": 0},
        "audit_trail": [],
    }
    pass_res = guardrail_node(pass_state)
    assert pass_res["guardrail_result"]["status"] == "passed"
    assert pass_res["guardrail_result"]["overridden"] is False
    assert pass_res["final_action"] == "RETRY"

    # Override case (Escalation cap hit)
    override_state: RecoveryState = {
        "llm_decision": {"decision": "ESCALATE"},
        "payment_context": {"status": "failed"},
        "customer_history": {"lifetime_escalations": 1},
        "audit_trail": [],
    }
    override_res = guardrail_node(override_state)
    assert override_res["guardrail_result"]["status"] == "overridden"
    assert override_res["guardrail_result"]["overridden"] is True
    assert override_res["guardrail_result"]["proposed_action"] == "ESCALATE"
    assert override_res["guardrail_result"]["final_action"] == "WAIT"
    assert "escalation" in override_res["guardrail_result"]["reason"].lower()
    assert override_res["final_action"] == "WAIT"


def test_execution_node_receives_final_action(tmp_path):
    """Test 4: Verify execution_node receives final_action and records execution event."""
    db_path = tmp_path / "test_exec.db"
    state: RecoveryState = {
        "payment_id": "pay_exec_test",
        "final_action": "RETRY",
        "llm_decision": {"decision": "RETRY", "confidence": 0.8},
        "guardrail_result": {"status": "passed"},
        "audit_trail": [],
    }

    result = execution_node(state, db_path=db_path)

    assert result["final_action"] == "RETRY"
    assert len(result["audit_trail"]) == 1
    assert result["audit_trail"][0]["node"] == "execution_node"
    assert result["audit_trail"][0]["status"] == "executed"
    assert result["audit_trail"][0]["final_action"] == "RETRY"


def test_error_path_bypasses_and_persists(tmp_path):
    """Test 3 & 12: Verify error path record is persisted correctly with final_action=WAIT."""
    db_path = tmp_path / "test_err_persist.db"
    state: RecoveryState = {
        "payment_id": "pay_err_01",
        "error": "Payment ID not found in dataset",
        "final_action": "WAIT",
        "llm_decision": {"decision": "N/A — error path"},
        "guardrail_result": {"status": "N/A — error path"},
        "audit_trail": [],
    }

    result = execution_node(state, db_path=db_path)
    assert result["final_action"] == "WAIT"

    rec = get_audit_record("pay_err_01", db_path=db_path)
    assert rec is not None
    assert rec["payment_id"] == "pay_err_01"
    assert rec["final_action"] == "WAIT"
    assert rec["error_status"] == "Payment ID not found in dataset"
    assert rec["llm_proposed_decision"] == "N/A — error path"
    assert rec["guardrail_verdict"] == "N/A — error path"


def test_full_reasoning_to_execution_chain_preserved(tmp_path):
    """Test 9, 10, 11: End-to-end chain verification in SQLite audit table."""
    db_path = tmp_path / "test_chain.db"
    mock_policy = make_mock_policy()
    mock_llm = make_mock_llm(
        decisions=[
            LLMDecision(
                decision="ESCALATE",
                confidence=0.92,
                reasoning="High balance account requiring immediate agent contact.",
                risk_level="high",
            )
        ]
    )

    graph = create_recovery_graph(policy=mock_policy, llm=mock_llm, db_path=db_path)

    # Trigger guardrail override via pre-set lifetime_escalations = 1
    input_state: RecoveryState = {
        "payment_id": "pay_000001_a1",
        "customer_history": {"lifetime_escalations": 1},
        "audit_trail": [],
    }

    final_state = graph.invoke(input_state)

    # 1. State assertions
    assert final_state["llm_decision"]["decision"] == "ESCALATE"
    assert final_state["guardrail_result"]["overridden"] is True
    assert final_state["final_action"] == "WAIT"

    # 2. SQLite direct query assertions
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM decision_audit WHERE payment_id = ?", ("pay_000001_a1",))
        row = dict(cur.fetchone())

    assert row["payment_id"] == "pay_000001_a1"
    assert row["llm_proposed_decision"] == "ESCALATE"
    assert row["final_action"] == "WAIT"
    assert row["guardrail_verdict"] == "overridden"
    assert "escalation" in row["guardrail_reason"].lower()
    assert row["llm_confidence"] == 0.92
    assert row["decision_source"] == "llm"
    assert "RETRY_NUDGE" in row["raw_arm_probabilities"]
    assert "RETRY_NUDGE" in row["raw_arm_net_values"]
