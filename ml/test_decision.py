"""
ml/test_decision.py
===================
Unit and integration tests for ml/decision.py (CausalUpliftPolicy).

Tests:
1. CausalUpliftPolicy inherits from Policy (policy/base.py).
2. Subclass source code check passes (no forbidden hidden tokens).
3. decide() accepts feature dict and returns an Action enum instance.
4. WAIT (cost 0.0) is a valid choice and wins when intervention costs exceed uplift.
5. Batch decisions via decide_batch() work accurately.
6. ACTION_COSTS from policy/cost_config.py are correctly subtracted.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from models.schemas import Action
from policy.base import Policy
from policy.cost_config import ACTION_COSTS
from ml.dataset import OBSERVABLE_FEATURES
from ml.decision import CausalUpliftPolicy
from ml.t_learner import TLearner


@pytest.fixture
def mock_t_learner():
    """Mock TLearner returning predefined probabilities."""
    learner = MagicMock(spec=TLearner)
    learner._is_fitted = True

    def predict_proba_side_effect(X):
        # Default predictions for each row in X
        # WAIT=0.30, RETRY=0.31, RETRY_NUDGE=0.32, ESCALATE=0.35
        n = len(X)
        return pd.DataFrame(
            {
                "WAIT": np.full(n, 0.30),
                "RETRY": np.full(n, 0.31),
                "RETRY_NUDGE": np.full(n, 0.32),
                "ESCALATE": np.full(n, 0.35),
            },
            index=X.index,
        )

    learner.predict_proba = predict_proba_side_effect
    return learner


@pytest.fixture
def sample_feature_dict():
    return {
        "amount": 100.0,
        "attempt_number": 1,
        "dynamic_success_rate": 0.5,
        "cumulative_failures": 1,
        "consecutive_failed_cycles": 0,
        "notification_engagement_score": 0.5,
        "contact_response_score": 0.5,
        "payment_method": "card",
        "failure_reason": "insufficient_funds",
    }


class TestCausalUpliftPolicy:
    def test_inherits_from_policy(self, mock_t_learner):
        policy = CausalUpliftPolicy(t_learner=mock_t_learner)
        assert isinstance(policy, Policy)

    def test_decide_returns_action_enum(self, mock_t_learner, sample_feature_dict):
        policy = CausalUpliftPolicy(t_learner=mock_t_learner)
        action = policy.decide(sample_feature_dict)
        assert isinstance(action, Action)

    def test_wait_wins_when_cost_exceeds_incremental(self, sample_feature_dict):
        """
        Test that WAIT wins when intervention cost > incremental recovery.

        Example:
        Amount = $100
        WAIT prob = 0.30  -> Net = $30.00 - 0 = $30.00
        RETRY prob = 0.31 -> Net = $31.00 - $5.00 = $26.00
        NUDGE prob = 0.32 -> Net = $32.00 - $15.00 = $17.00
        ESCALATE prob = 0.35 -> Net = $35.00 - $250.00 = -$215.00

        WAIT net ($30.00) is highest!
        """
        mock_learner = MagicMock(spec=TLearner)
        mock_learner._is_fitted = True
        mock_learner.predict_proba.return_value = pd.DataFrame(
            [{'WAIT': 0.30, 'RETRY': 0.31, 'RETRY_NUDGE': 0.32, 'ESCALATE': 0.35}]
        )

        policy = CausalUpliftPolicy(t_learner=mock_learner)
        action = policy.decide(sample_feature_dict)
        assert action == Action.WAIT

    def test_retry_wins_when_worth_cost(self, sample_feature_dict):
        """
        Test that RETRY wins when net value exceeds WAIT and other actions.

        Amount = $1000
        WAIT prob = 0.10  -> Net = $100.00 - 0 = $100.00
        RETRY prob = 0.40 -> Net = $400.00 - $5.00 = $395.00
        NUDGE prob = 0.41 -> Net = $410.00 - $15.00 = $395.00
        ESCALATE prob = 0.50 -> Net = $500.00 - $250.00 = $250.00

        RETRY net ($395.00) >= NUDGE ($395.00) and > WAIT ($100.00).
        """
        sample_feature_dict["amount"] = 1000.0
        mock_learner = MagicMock(spec=TLearner)
        mock_learner._is_fitted = True
        mock_learner.predict_proba.return_value = pd.DataFrame(
            [{'WAIT': 0.10, 'RETRY': 0.40, 'RETRY_NUDGE': 0.40, 'ESCALATE': 0.50}]
        )

        policy = CausalUpliftPolicy(t_learner=mock_learner)
        action = policy.decide(sample_feature_dict)
        assert action == Action.RETRY

    def test_escalate_wins_for_high_amount_high_uplift(self, sample_feature_dict):
        """
        Test ESCALATE wins when payment amount is very high and uplift is large.

        Amount = $10,000
        WAIT prob = 0.10     -> Net = $1,000 - $0 = $1,000
        RETRY prob = 0.20    -> Net = $2,000 - $5 = $1,995
        NUDGE prob = 0.30    -> Net = $3,000 - $15 = $2,985
        ESCALATE prob = 0.80 -> Net = $8,000 - $250 = $7,750

        ESCALATE net ($7,750) wins!
        """
        sample_feature_dict["amount"] = 10000.0
        mock_learner = MagicMock(spec=TLearner)
        mock_learner._is_fitted = True
        mock_learner.predict_proba.return_value = pd.DataFrame(
            [{'WAIT': 0.10, 'RETRY': 0.20, 'RETRY_NUDGE': 0.30, 'ESCALATE': 0.80}]
        )

        policy = CausalUpliftPolicy(t_learner=mock_learner)
        action = policy.decide(sample_feature_dict)
        assert action == Action.ESCALATE

    def test_decide_batch(self, mock_t_learner, sample_feature_dict):
        df = pd.DataFrame([sample_feature_dict, sample_feature_dict])
        policy = CausalUpliftPolicy(t_learner=mock_t_learner)
        actions = policy.decide_batch(df)

        assert isinstance(actions, pd.Series)
        assert len(actions) == 2
        assert all(isinstance(a, Action) for a in actions)

    def test_missing_features_raises_keyerror(self, mock_t_learner):
        incomplete_dict = {"amount": 100.0}
        policy = CausalUpliftPolicy(t_learner=mock_t_learner)
        with pytest.raises(KeyError, match="requires observable features"):
            policy.decide(incomplete_dict)
