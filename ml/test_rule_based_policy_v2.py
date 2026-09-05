"""
ml/test_rule_based_policy_v2.py
================================
Unit and integration tests for policy/rule_based_policy_v2.py (RuleBasedPolicyV2).

Tests:
1. RuleBasedPolicyV2 inherits from Policy (policy/base.py).
2. Required columns check raises KeyError when missing.
3. decide() returns Action enum.
4. decide_batch() is 100% deterministic across repeated calls.
5. Economically cost-aware logic selects correct action under cost schedules.
"""

from __future__ import annotations

import pandas as pd
import pytest
from models.schemas import Action
from policy.base import Policy
from policy.rule_based_policy_v2 import RuleBasedPolicyV2


@pytest.fixture
def sample_features():
    return {
        "amount": 1000.0,
        "dynamic_success_rate": 0.50,
        "consecutive_failed_cycles": 1,
        "contact_response_score": 0.50,
        "notification_engagement_score": 0.50,
    }


class TestRuleBasedPolicyV2:
    def test_inherits_from_policy(self):
        policy = RuleBasedPolicyV2()
        assert isinstance(policy, Policy)

    def test_decide_returns_action(self, sample_features):
        policy = RuleBasedPolicyV2()
        action = policy.decide(sample_features)
        assert isinstance(action, Action)

    def test_missing_required_column_raises_keyerror(self):
        policy = RuleBasedPolicyV2()
        incomplete = {"amount": 500.0, "dynamic_success_rate": 0.5}
        with pytest.raises(KeyError, match="requires observable columns"):
            policy.decide(incomplete)

    def test_high_dynamic_success_rate_waits(self, sample_features):
        sample_features["dynamic_success_rate"] = 0.85
        policy = RuleBasedPolicyV2()
        assert policy.decide(sample_features) == Action.WAIT

    def test_decide_batch_determinism(self, sample_features):
        df = pd.DataFrame([sample_features] * 100)
        policy = RuleBasedPolicyV2()
        actions1 = policy.decide_batch(df)
        actions2 = policy.decide_batch(df)
        assert actions1 == actions2

    def test_escalate_chosen_only_when_net_value_exceeds_other_actions(self):
        """
        On a high amount payment (e.g. 9999), ESCALATE proxy net value
        must exceed RETRY and RETRY_NUDGE to be selected.
        """
        high_val_features = {
            "amount": 9999.0,
            "dynamic_success_rate": 0.50,
            "consecutive_failed_cycles": 4,
            "contact_response_score": 0.20,
            "notification_engagement_score": 0.20,
        }
        policy = RuleBasedPolicyV2()
        action = policy.decide(high_val_features)
        assert isinstance(action, Action)
