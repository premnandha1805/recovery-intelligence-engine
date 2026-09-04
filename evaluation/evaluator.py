"""
evaluation/evaluator.py — Evaluates policy decisions against hidden ground truth.

Hard leakage rule
-----------------
This is the ONLY module allowed to join observable features + policy decisions
with hidden ground-truth probabilities. Policies must NEVER import or read
ground-truth probabilities.

Functions
---------
evaluate_policy(
    policy_decisions_df,
    hidden_ground_truth_df,
    observable_dataset_df,
    cost_config=None,
)
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from models.schemas import Action
from policy.cost_config import ACTION_COSTS

# Mapping from Action enum / action string to ground_truth.csv column name
ACTION_PROB_COLUMN_MAP: dict[str, str] = {
    Action.WAIT.value:        "natural_recovery_probability",
    Action.RETRY.value:       "retry_success_probability",
    Action.RETRY_NUDGE.value: "nudge_success_probability",
    Action.ESCALATE.value:    "escalation_success_probability",
    Action.STOP.value:        "natural_recovery_probability",  # STOP assumes no intervention
}


def evaluate_policy(
    policy_decisions_df: pd.DataFrame,
    hidden_ground_truth_df: pd.DataFrame,
    observable_dataset_df: pd.DataFrame,
    cost_config: dict | None = None,
) -> pd.DataFrame:
    """
    Evaluate policy decisions against ground-truth success probabilities.

    Parameters
    ----------
    policy_decisions_df : pd.DataFrame
        Must contain [payment_id, policy_name, chosen_action].
    hidden_ground_truth_df : pd.DataFrame
        Must contain [payment_id, natural_recovery_probability,
                      retry_success_probability, nudge_success_probability,
                      escalation_success_probability].
    observable_dataset_df : pd.DataFrame
        Must contain [payment_id, amount].  `amount` MUST come from here.
    cost_config : dict, optional
        Action cost mapping. Defaults to ACTION_COSTS from policy.cost_config.

    Returns
    -------
    pd.DataFrame
        Detailed per-decision evaluation with columns:
        [payment_id, policy_name, chosen_action, amount,
         p_success_for_chosen_action, action_cost, expected_recovery,
         expected_net_value]
    """
    costs = cost_config if cost_config is not None else ACTION_COSTS

    # --- 1. Fail-fast validation of required input columns ---
    dec_req = {"payment_id", "policy_name", "chosen_action"}
    missing_dec = dec_req - set(policy_decisions_df.columns)
    if missing_dec:
        raise KeyError(f"policy_decisions_df missing required columns: {sorted(missing_dec)}")

    gt_req = {
        "payment_id",
        "natural_recovery_probability",
        "retry_success_probability",
        "nudge_success_probability",
        "escalation_success_probability",
    }
    missing_gt = gt_req - set(hidden_ground_truth_df.columns)
    if missing_gt:
        raise KeyError(f"hidden_ground_truth_df missing required columns: {sorted(missing_gt)}")

    obs_req = {"payment_id", "amount"}
    missing_obs = obs_req - set(observable_dataset_df.columns)
    if missing_obs:
        raise KeyError(f"observable_dataset_df missing required columns: {sorted(missing_obs)}")

    # Check for NaN in payment_id or amount
    if observable_dataset_df["payment_id"].isna().any():
        raise ValueError("observable_dataset_df contains NaN payment_id values")
    if observable_dataset_df["amount"].isna().any():
        raise ValueError("observable_dataset_df contains NaN amount values")

    # --- 2. Join amount from observable dataset ---
    if observable_dataset_df["payment_id"].duplicated().any():
        raise ValueError("observable_dataset_df contains duplicate payment_id entries")

    amount_lookup = observable_dataset_df.set_index("payment_id")["amount"]

    # --- 3. Join ground-truth probabilities ---
    if hidden_ground_truth_df["payment_id"].duplicated().any():
        raise ValueError("hidden_ground_truth_df contains duplicate payment_id entries")

    gt_cols = [
        "natural_recovery_probability",
        "retry_success_probability",
        "nudge_success_probability",
        "escalation_success_probability",
    ]
    gt_lookup = hidden_ground_truth_df.set_index("payment_id")[gt_cols]

    # Check duplicate (payment_id, policy_name) in policy_decisions
    if policy_decisions_df.duplicated(subset=["payment_id", "policy_name"]).any():
        raise ValueError("policy_decisions_df contains duplicate (payment_id, policy_name) entries")

    # --- 4. Validate actions and calculate metrics ---
    valid_action_values = {a.value for a in Action}

    merged = policy_decisions_df[["payment_id", "policy_name", "chosen_action"]].copy()

    # Verify every action is valid
    invalid_actions = set(merged["chosen_action"]) - valid_action_values
    if invalid_actions:
        raise ValueError(f"Invalid chosen_action values found: {invalid_actions}. Must be in {valid_action_values}")

    # Join amount from observable dataset
    merged["amount"] = merged["payment_id"].map(amount_lookup)
    if merged["amount"].isna().any():
        missing_pids = merged[merged["amount"].isna()]["payment_id"].unique()
        raise ValueError(f"Could not join amount for payment_ids: {missing_pids[:5]}")

    # Merge ground truth probabilities
    merged = merged.merge(gt_lookup, on="payment_id", how="left")

    # Map chosen_action -> p_success and action_cost
    p_success_vals = []
    action_cost_vals = []

    for _, row in merged.iterrows():
        act_val = row["chosen_action"]
        prob_col = ACTION_PROB_COLUMN_MAP.get(act_val)
        if not prob_col or prob_col not in row or pd.isna(row[prob_col]):
            raise ValueError(f"Missing success probability for action {act_val!r} on payment {row['payment_id']!r}")

        prob = float(row[prob_col])
        p_success_vals.append(prob)

        # Lookup action cost
        cost = None
        if act_val in costs:
            cost = float(costs[act_val])
        else:
            try:
                enum_act = Action(act_val)
                if enum_act in costs:
                    cost = float(costs[enum_act])
            except ValueError:
                pass

        if cost is None:
            raise ValueError(f"Action cost missing from cost_config for action {act_val!r}")
        action_cost_vals.append(cost)

    merged["p_success_for_chosen_action"] = p_success_vals
    merged["action_cost"] = action_cost_vals
    merged["expected_recovery"] = (merged["amount"] * merged["p_success_for_chosen_action"]).round(4)
    merged["expected_net_value"] = (merged["expected_recovery"] - merged["action_cost"]).round(4)

    output_cols = [
        "payment_id",
        "policy_name",
        "chosen_action",
        "amount",
        "p_success_for_chosen_action",
        "action_cost",
        "expected_recovery",
        "expected_net_value",
    ]

    return merged[output_cols]
