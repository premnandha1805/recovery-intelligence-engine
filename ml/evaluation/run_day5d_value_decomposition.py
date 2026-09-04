"""
ml/evaluation/run_day5d_value_decomposition.py
================================================
Day 5D — Value Decomposition Analysis: CausalUpliftPolicy vs. RuleBasedPolicy.

Analyzes decision transitions from RuleBasedPolicy -> CausalUpliftPolicy:
1. For every payment, pairs RuleBased action and CausalUplift action.
2. Computes: expected recovery, incremental uplift over WAIT, action cost, net value.
3. Groups by transition pair (e.g. ESCALATE -> RETRY, RETRY_NUDGE -> RETRY, etc.).
4. For each transition reports:
   - payment count & percentage
   - mean & total expected recovery difference
   - mean & total action cost difference
   - mean & total net-value difference
5. Identifies the TOP 3 REASONS CausalUpliftPolicy produces higher net financial value.

Explanatory analysis ONLY — does NOT modify any evaluator, policy, or core module.
"""

from __future__ import annotations

import pathlib
import sys
import pandas as pd

# Project root setup
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.evaluator import evaluate_policy
from policy.cost_config import ACTION_COSTS
from ml.t_learner import fit_final_models
from ml.decision import CausalUpliftPolicy

ML_EVAL_DIR = _REPO_ROOT / "ml" / "evaluation"

EVAL_POPULATIONS = {
    "Seed 777 (OOS Population 1)": {
        "obs": ML_EVAL_DIR / "seed777_data" / "payment_scenarios.csv",
        "gt": ML_EVAL_DIR / "seed777_data" / "ground_truth.csv",
    },
    "Seed 555 (OOS Population 2)": {
        "obs": ML_EVAL_DIR / "seed555_data" / "payment_scenarios.csv",
        "gt": ML_EVAL_DIR / "seed555_data" / "ground_truth.csv",
    },
    "Seed 42 (In-Sample Fit Check)": {
        "obs": _REPO_ROOT / "data" / "v2" / "payment_scenarios.csv",
        "gt": _REPO_ROOT / "data" / "v2" / "ground_truth.csv",
    },
}


def get_rule_based_actions(obs_df: pd.DataFrame) -> list[str]:
    """Vectorized implementation of RuleBasedPolicy decision logic."""
    dyn_sr = obs_df["dynamic_success_rate"].astype(float).values
    consec_fails = obs_df["consecutive_failed_cycles"].astype(int).values
    contact_score = obs_df["contact_response_score"].astype(float).values
    notif_score = obs_df["notification_engagement_score"].astype(float).values

    actions = []
    for i in range(len(obs_df)):
        if dyn_sr[i] >= 0.80:
            actions.append("WAIT")
        elif consec_fails[i] >= 3 and dyn_sr[i] < 0.30:
            actions.append("ESCALATE")
        elif contact_score[i] > notif_score[i]:
            actions.append("RETRY")
        else:
            actions.append("RETRY_NUDGE")
    return actions


def run_value_decomposition_for_population(
    pop_name: str,
    obs_path: pathlib.Path,
    gt_path: pathlib.Path,
    causal_policy: CausalUpliftPolicy,
) -> pd.DataFrame:
    """Run full payment-level value decomposition for a specific population."""
    obs_df = pd.read_csv(obs_path)
    gt_df = pd.read_csv(gt_path)

    pids = obs_df["payment_id"].tolist()
    n_total = len(obs_df)

    # Decisions
    rb_actions = get_rule_based_actions(obs_df)
    causal_actions = causal_policy.decide_batch(obs_df)

    rb_decs = pd.DataFrame([
        {"payment_id": pid, "policy_name": "RuleBasedPolicy", "chosen_action": act}
        for pid, act in zip(pids, rb_actions)
    ])
    causal_decs = pd.DataFrame([
        {"payment_id": pid, "policy_name": "CausalUpliftPolicy", "chosen_action": act.value}
        for pid, act in zip(pids, causal_actions)
    ])

    # Evaluate both using exact Day 3 evaluator
    rb_eval = evaluate_policy(rb_decs, gt_df, obs_df, ACTION_COSTS)
    causal_eval = evaluate_policy(causal_decs, gt_df, obs_df, ACTION_COSTS)

    # Join WAIT probability for incremental uplift calculation
    gt_wait_map = gt_df.set_index("payment_id")["natural_recovery_probability"].to_dict()

    # Merge evaluated datasets by payment_id
    rb_eval = rb_eval.rename(columns={
        "chosen_action": "action_rule",
        "p_success_for_chosen_action": "p_success_rule",
        "action_cost": "cost_rule",
        "expected_recovery": "recovery_rule",
        "expected_net_value": "net_val_rule",
    })
    causal_eval = causal_eval.rename(columns={
        "chosen_action": "action_causal",
        "p_success_for_chosen_action": "p_success_causal",
        "action_cost": "cost_causal",
        "expected_recovery": "recovery_causal",
        "expected_net_value": "net_val_causal",
    })

    merged = causal_eval[[
        "payment_id", "amount", "action_causal", "p_success_causal",
        "cost_causal", "recovery_causal", "net_val_causal"
    ]].merge(
        rb_eval[[
            "payment_id", "action_rule", "p_success_rule",
            "cost_rule", "recovery_rule", "net_val_rule"
        ]],
        on="payment_id",
        how="inner",
    )

    merged["p_success_wait"] = merged["payment_id"].map(gt_wait_map)
    merged["incremental_uplift_causal"] = merged["p_success_causal"] - merged["p_success_wait"]
    merged["incremental_uplift_rule"] = merged["p_success_rule"] - merged["p_success_wait"]

    merged["transition"] = merged["action_rule"] + " -> " + merged["action_causal"]

    # Differences per payment (Causal - RuleBased)
    merged["recovery_diff"] = merged["recovery_causal"] - merged["recovery_rule"]
    merged["cost_diff"] = merged["cost_causal"] - merged["cost_rule"]
    merged["net_val_diff"] = merged["net_val_causal"] - merged["net_val_rule"]

    # Aggregate by transition
    transition_groups = []
    for trans, group in merged.groupby("transition"):
        n_p = len(group)
        pct_p = (n_p / n_total) * 100

        total_rec_diff = group["recovery_diff"].sum()
        total_cost_diff = group["cost_diff"].sum()
        total_net_diff = group["net_val_diff"].sum()

        mean_rec_diff = group["recovery_diff"].mean()
        mean_cost_diff = group["cost_diff"].mean()
        mean_net_diff = group["net_val_diff"].mean()

        mean_uplift_causal = group["incremental_uplift_causal"].mean() * 100
        mean_uplift_rule = group["incremental_uplift_rule"].mean() * 100

        transition_groups.append({
            "Population": pop_name,
            "Transition (Rule -> Causal)": trans,
            "Payment Count": n_p,
            "% of Total": round(pct_p, 2),
            "Total Net Val Diff (INR)": round(total_net_diff, 2),
            "Total Recov Diff (INR)": round(total_rec_diff, 2),
            "Total Cost Diff (INR)": round(total_cost_diff, 2),
            "Mean Net Val Diff (INR)": round(mean_net_diff, 2),
            "Mean Recov Diff (INR)": round(mean_rec_diff, 2),
            "Mean Cost Diff (INR)": round(mean_cost_diff, 2),
            "Mean Causal Uplift": f"{mean_uplift_causal:+.2f} pp",
            "Mean Rule Uplift": f"{mean_uplift_rule:+.2f} pp",
        })

    trans_df = pd.DataFrame(transition_groups)
    trans_df = trans_df.sort_values(by="Total Net Val Diff (INR)", ascending=False).reset_index(drop=True)
    return trans_df


def main():
    print("=" * 80)
    print("  DAY 5D — VALUE DECOMPOSITION ANALYSIS: CAUSAL VS RULEBASED")
    print("=" * 80)

    print("\nLoading canonical fitted T-Learner model...")
    causal_policy = CausalUpliftPolicy()

    all_trans_dfs = []

    for pop_name, pop_paths in EVAL_POPULATIONS.items():
        print(f"\n{'=' * 75}")
        print(f"  ANALYZING VALUE DECOMPOSITION ON {pop_name.upper()}")
        print(f"{'=' * 75}")

        trans_df = run_value_decomposition_for_population(
            pop_name, pop_paths["obs"], pop_paths["gt"], causal_policy
        )

        print(trans_df.to_string(index=False))
        all_trans_dfs.append(trans_df)

    # Save summary artifact
    combined_df = pd.concat(all_trans_dfs, ignore_index=True)
    out_csv = ML_EVAL_DIR / "day5d_value_decomposition.csv"
    combined_df.to_csv(out_csv, index=False)
    print(f"\nSaved Day 5D value decomposition artifact: {out_csv}")


if __name__ == "__main__":
    main()
