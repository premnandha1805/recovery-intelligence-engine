"""
ml/decision.py
==============
Causal Uplift Policy for optimal recovery action selection.

This module implements CausalUpliftPolicy, which inherits from Policy
(defined in policy/base.py). It uses a fitted T-Learner model to predict
potential recovery outcomes across all four treatment arms, subtracts
intervention costs defined in policy/cost_config.py, and selects the
action that maximizes net expected financial value.

Decision Rule
-------------
For each candidate action a in {WAIT, RETRY, RETRY_NUDGE, ESCALATE}:
    predicted_recovery(a) = P(Y=1 | X, T=a) * payment_amount
    net_value(a) = predicted_recovery(a) - ACTION_COSTS[a]

Chosen action:
    argmax_a (net_value(a))

WAIT (cost = 0.0) is a valid choice and will win whenever the expected
incremental recovery from active interventions (RETRY, RETRY_NUDGE, ESCALATE)
does not outweigh their respective costs.

Immutability & Safety
---------------------
- Implements Policy interface without modifying policy/base.py or any
  existing policy file.
- Uses only observable features defined in ml/dataset.py.
- Contains no references to hidden simulator tokens.
"""

from __future__ import annotations

import pandas as pd

from models.schemas import Action
from policy.base import Policy
from policy.cost_config import ACTION_COSTS
from ml.dataset import OBSERVABLE_FEATURES
from ml.t_learner import TLearner, fit_final_models

# Mapping from T-Learner arm names to Action enum values
ARM_TO_ACTION: dict[str, Action] = {
    "WAIT": Action.WAIT,
    "RETRY": Action.RETRY,
    "RETRY_NUDGE": Action.RETRY_NUDGE,
    "ESCALATE": Action.ESCALATE,
}


class CausalUpliftPolicy(Policy):
    """
    Net-value maximizing policy powered by a T-Learner uplift model.
    """

    def __init__(self, t_learner: TLearner | None = None):
        """
        Initialize CausalUpliftPolicy.

        Parameters
        ----------
        t_learner : TLearner, optional
            A fitted TLearner model. If None, fits a final TLearner model
            on the full training dataset via ml.t_learner.fit_final_models().
        """
        if t_learner is None:
            self.t_learner = fit_final_models()
        else:
            self.t_learner = t_learner

    def decide(self, payment_features: dict) -> Action:
        """
        Choose net-value maximizing recovery action for a single payment.

        Parameters
        ----------
        payment_features : dict
            Dictionary of observable payment features. Must include all
            columns defined in OBSERVABLE_FEATURES.

        Returns
        -------
        Action
            Action.WAIT, Action.RETRY, Action.RETRY_NUDGE, or Action.ESCALATE.
        """
        missing = [c for c in OBSERVABLE_FEATURES if c not in payment_features]
        if missing:
            raise KeyError(
                f"CausalUpliftPolicy requires observable features {missing}. "
                f"Keys provided: {sorted(payment_features.keys())}"
            )

        amount = float(payment_features.get("amount", 1.0))
        df_single = pd.DataFrame([payment_features])[OBSERVABLE_FEATURES]

        probas = self.t_learner.predict_proba(df_single).iloc[0]

        best_action = Action.WAIT
        best_net_value = float("-inf")

        for arm_name, action_enum in ARM_TO_ACTION.items():
            prob = float(probas[arm_name])
            cost = float(ACTION_COSTS[action_enum])
            net_val = (prob * amount) - cost

            if net_val > best_net_value:
                best_net_value = net_val
                best_action = action_enum

        return best_action

    def decide_batch(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute decisions for a batch of payment records.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing columns for all OBSERVABLE_FEATURES.

        Returns
        -------
        pd.Series
            Series of Action enum values indexed like df.
        """
        missing = [c for c in OBSERVABLE_FEATURES if c not in df.columns]
        if missing:
            raise KeyError(
                f"CausalUpliftPolicy requires observable feature columns {missing}."
            )

        X = df[OBSERVABLE_FEATURES]
        amounts = df["amount"].astype(float) if "amount" in df.columns else pd.Series(1.0, index=df.index)

        probas = self.t_learner.predict_proba(X)

        net_values = pd.DataFrame(index=df.index)
        for arm_name, action_enum in ARM_TO_ACTION.items():
            cost = float(ACTION_COSTS[action_enum])
            net_values[arm_name] = (probas[arm_name] * amounts) - cost

        best_arms = net_values.idxmax(axis=1)
        return best_arms.map(ARM_TO_ACTION)
