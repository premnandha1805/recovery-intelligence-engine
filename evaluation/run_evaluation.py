"""
evaluation/run_evaluation.py — Evaluates policy_decisions.csv against V2 ground truth.

Pipeline
--------
1. Load policy decisions (simulator/output/policy_decisions.csv).
2. Load observable payment dataset (data/v2/payment_scenarios.csv).
3. Load hidden ground truth (data/v2/ground_truth.csv).
4. Run evaluate_policy() from evaluation.evaluator.
5. Perform integrity checks.
6. Write evaluation/output/policy_evaluation.csv.
7. Print summary metrics per policy.

Usage
-----
    python evaluation/run_evaluation.py
"""

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.evaluator import evaluate_policy
from models.schemas import Action
from policy.cost_config import ACTION_COSTS

# Default paths
DEFAULT_DECISIONS_PATH = os.path.join(PROJECT_ROOT, "simulator", "output", "policy_decisions.csv")
DEFAULT_OBS_PATH       = os.path.join(PROJECT_ROOT, "data", "v2", "payment_scenarios.csv")
DEFAULT_GT_PATH        = os.path.join(PROJECT_ROOT, "data", "v2", "ground_truth.csv")
DEFAULT_OUT_DIR        = os.path.join(PROJECT_ROOT, "evaluation", "output")
DEFAULT_OUT_PATH       = os.path.join(DEFAULT_OUT_DIR, "policy_evaluation.csv")


def run(
    decisions_path: str = DEFAULT_DECISIONS_PATH,
    obs_path: str = DEFAULT_OBS_PATH,
    gt_path: str = DEFAULT_GT_PATH,
    out_path: str = DEFAULT_OUT_PATH,
) -> pd.DataFrame:
    """Run full dataset evaluation and save results."""
    print("=" * 65)
    print("  RUNNING FULL DATASET EVALUATION (v2)")
    print("=" * 65)

    print(f"Loading policy decisions:  {decisions_path}")
    dec_df = pd.read_csv(decisions_path)
    print(f"  {len(dec_df):,} decision rows")

    print(f"Loading observable data:  {obs_path}")
    obs_df = pd.read_csv(obs_path)
    print(f"  {len(obs_df):,} observable rows")

    print(f"Loading ground truth:     {gt_path}")
    gt_df = pd.read_csv(gt_path)
    print(f"  {len(gt_df):,} ground-truth rows")

    # Run evaluation
    print("\nExecuting evaluate_policy()...")
    eval_df = evaluate_policy(
        policy_decisions_df=dec_df,
        hidden_ground_truth_df=gt_df,
        observable_dataset_df=obs_df,
        cost_config=ACTION_COSTS,
    )

    # --- Step 7 Integrity Checks ---
    print("\n" + "=" * 65)
    print("  INTEGRITY CHECKS")
    print("=" * 65)

    # 1. Total rows check (121,888)
    n_total = len(eval_df)
    n_expected = len(obs_df) * len(dec_df["policy_name"].unique())
    print(f"  [PASS] Output row count: {n_total:,} (expected {n_expected:,})")
    assert n_total == n_expected, f"Row count mismatch: {n_total} vs {n_expected}"

    # 2. Per-policy row count check (30,472)
    per_pol_counts = eval_df.groupby("policy_name").size()
    print("  Per-policy row counts:")
    for pol_name, cnt in per_pol_counts.items():
        print(f"    - {pol_name:20s}: {cnt:,}")
        assert cnt == len(obs_df), f"{pol_name} row count {cnt} != {len(obs_df)}"

    # 3. Duplicate check (payment_id, policy_name)
    dups = eval_df.duplicated(subset=["payment_id", "policy_name"]).sum()
    print(f"  [PASS] Duplicate (payment_id, policy_name) pairs: {dups}")
    assert dups == 0, "Duplicate entries found in evaluation output"

    # 4. Check all payment_ids exist in observable data and ground truth
    obs_pids = set(obs_df["payment_id"])
    gt_pids  = set(gt_df["payment_id"])
    eval_pids = set(eval_df["payment_id"])
    print(f"  [PASS] All payment_ids in observable data: {eval_pids.issubset(obs_pids)}")
    print(f"  [PASS] All payment_ids in ground truth:    {eval_pids.issubset(gt_pids)}")
    assert eval_pids.issubset(obs_pids), "Payment IDs missing from observable dataset"
    assert eval_pids.issubset(gt_pids), "Payment IDs missing from ground truth dataset"

    # 5. Missing value check
    null_counts = eval_df.isna().sum()
    print(f"  [PASS] Missing / NaN values across all columns: {null_counts.sum()}")
    assert null_counts.sum() == 0, f"NaN values found: {null_counts.to_dict()}"

    # 6. Valid action enum check
    valid_actions = {a.value for a in Action}
    eval_actions = set(eval_df["chosen_action"])
    print(f"  [PASS] All actions in Action enum: {eval_actions.issubset(valid_actions)}")
    assert eval_actions.issubset(valid_actions), f"Invalid actions: {eval_actions - valid_actions}"

    # 7. Action costs match cost_config
    cost_mismatches = 0
    for _, r in eval_df.iterrows():
        expected_c = ACTION_COSTS[Action(r["chosen_action"])]
        if r["action_cost"] != expected_c:
            cost_mismatches += 1
    print(f"  [PASS] Action cost mismatches against cost_config: {cost_mismatches}")
    assert cost_mismatches == 0, "Action cost mismatch against policy/cost_config.py"

    # Save output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    eval_df.to_csv(out_path, index=False)
    print(f"\nSaved evaluation output: {out_path}")

    # --- Summary Table ---
    print("\n" + "=" * 65)
    print("  SUMMARY EVALUATION TABLE")
    print("=" * 65)

    summary_rows = []
    for pol_name, group in eval_df.groupby("policy_name"):
        n_p = len(group)
        rec_rate = group["p_success_for_chosen_action"].mean()
        gross_rev = group["expected_recovery"].sum()
        tot_cost  = group["action_cost"].sum()
        net_val   = group["expected_net_value"].sum()

        summary_rows.append({
            "Policy": pol_name,
            "Payments": f"{n_p:,}",
            "Recovery Rate": f"{rec_rate*100:.2f}%",
            "Gross Revenue (INR)": f"INR {gross_rev:,.2f}",
            "Action Cost (INR)": f"INR {tot_cost:,.2f}",
            "Net Value (INR)": f"INR {net_val:,.2f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    return eval_df


if __name__ == "__main__":
    run()
