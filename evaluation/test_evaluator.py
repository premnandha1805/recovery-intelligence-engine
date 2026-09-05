"""
Unit tests for evaluation/evaluator.py.

Includes the hand-worked test case required by Day 3C spec:
  amount = 1000
  hidden probs: WAIT=0.40, RETRY=0.60, RETRY_NUDGE=0.70, ESCALATE=0.90
  costs: WAIT=0, RETRY=5, RETRY_NUDGE=15, ESCALATE=250

Expected net values:
  WAIT        : 1000 * 0.40 - 0   = 400
  RETRY       : 1000 * 0.60 - 5   = 595
  RETRY_NUDGE : 1000 * 0.70 - 15  = 685
  ESCALATE    : 1000 * 0.90 - 250 = 650
"""

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.evaluator import evaluate_policy
from models.schemas import Action
from policy.cost_config import ACTION_COSTS


def test_hand_worked_example():
    """Verify hand-worked test case across all four actions."""
    # 1. Observable dataset (amount = 1000)
    obs_df = pd.DataFrame([{
        "payment_id": "test_payment",
        "amount": 1000.0,
    }])

    # 2. Hidden ground truth probabilities
    gt_df = pd.DataFrame([{
        "payment_id": "test_payment",
        "natural_recovery_probability": 0.40,
        "retry_success_probability": 0.60,
        "nudge_success_probability": 0.70,
        "escalation_success_probability": 0.90,
    }])

    # 3. Policy decisions testing all four actions
    decisions_df = pd.DataFrame([
        {"payment_id": "test_payment", "policy_name": "TestWait", "chosen_action": Action.WAIT.value},
        {"payment_id": "test_payment", "policy_name": "TestRetry", "chosen_action": Action.RETRY.value},
        {"payment_id": "test_payment", "policy_name": "TestNudge", "chosen_action": Action.RETRY_NUDGE.value},
        {"payment_id": "test_payment", "policy_name": "TestEscalate", "chosen_action": Action.ESCALATE.value},
    ])

    # 4. Custom costs (matching ACTION_COSTS)
    cost_cfg = {
        Action.WAIT: 0.0,
        Action.RETRY: 5.0,
        Action.RETRY_NUDGE: 15.0,
        Action.ESCALATE: 250.0,
    }

    # Run evaluator
    result_df = evaluate_policy(decisions_df, gt_df, obs_df, cost_config=cost_cfg)

    print("\n--- Hand-Worked Unit Test Results ---")
    print(result_df[["policy_name", "chosen_action", "amount", "p_success_for_chosen_action", "action_cost", "expected_recovery", "expected_net_value"]].to_string(index=False))

    # Assertions
    wait_row = result_df[result_df["chosen_action"] == Action.WAIT.value].iloc[0]
    retry_row = result_df[result_df["chosen_action"] == Action.RETRY.value].iloc[0]
    nudge_row = result_df[result_df["chosen_action"] == Action.RETRY_NUDGE.value].iloc[0]
    esc_row = result_df[result_df["chosen_action"] == Action.ESCALATE.value].iloc[0]

    # Expected recoveries: 400, 600, 700, 900
    assert np.isclose(wait_row["expected_recovery"], 400.0), f"WAIT recovery failed: {wait_row['expected_recovery']}"
    assert np.isclose(retry_row["expected_recovery"], 600.0), f"RETRY recovery failed: {retry_row['expected_recovery']}"
    assert np.isclose(nudge_row["expected_recovery"], 700.0), f"NUDGE recovery failed: {nudge_row['expected_recovery']}"
    assert np.isclose(esc_row["expected_recovery"], 900.0), f"ESCALATE recovery failed: {esc_row['expected_recovery']}"

    # Expected net values: 400, 595, 685, 650
    assert np.isclose(wait_row["expected_net_value"], 400.0), f"WAIT net value failed: {wait_row['expected_net_value']}"
    assert np.isclose(retry_row["expected_net_value"], 595.0), f"RETRY net value failed: {retry_row['expected_net_value']}"
    assert np.isclose(nudge_row["expected_net_value"], 685.0), f"NUDGE net value failed: {nudge_row['expected_net_value']}"
    assert np.isclose(esc_row["expected_net_value"], 650.0), f"ESCALATE net value failed: {esc_row['expected_net_value']}"

    print("\n[OK] HAND-WORKED UNIT TEST PASSED SUCCESSFULLY!")


def test_fail_fast_validation():
    """Verify that evaluator fails loudly on invalid inputs."""
    obs_df = pd.DataFrame([{"payment_id": "p1", "amount": 500.0}])
    gt_df = pd.DataFrame([{
        "payment_id": "p1",
        "natural_recovery_probability": 0.5,
        "retry_success_probability": 0.6,
        "nudge_success_probability": 0.7,
        "escalation_success_probability": 0.8,
    }])
    bad_decisions = pd.DataFrame([{"payment_id": "p1", "policy_name": "Bad", "chosen_action": "INVALID_ACTION"}])

    try:
        evaluate_policy(bad_decisions, gt_df, obs_df)
        assert False, "Should have raised ValueError for invalid action"
    except ValueError as e:
        assert "Invalid chosen_action values found" in str(e)
        print("[OK] Fail-fast validation test passed!")


if __name__ == "__main__":
    test_hand_worked_example()
    test_fail_fast_validation()
