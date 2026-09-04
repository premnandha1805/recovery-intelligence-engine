"""
ml/evaluation/run_day4_evaluation.py
=====================================
Day 4J — Final Causal Policy Evaluation Against Day 3 Baselines (Fast Vectorized).

Pipeline
--------
1. Load evaluation datasets for Seed 42 (data/v2/) and Seed 777 (ml/evaluation/seed777_data/).
2. Run 5 policies over each seed:
     - WaitPolicy
     - AlwaysRetryPolicy
     - AlwaysNudgePolicy
     - RuleBasedPolicy
     - CausalUpliftPolicy (ml/decision.py)
3. Evaluate using the EXACT Day 3 evaluator (evaluation/evaluator.py).
4. Build reports using the EXACT Day 3 reporter (evaluation/report.py).
5. Output results to ml/evaluation/ (preserving all Day 3 files).
"""

from __future__ import annotations

import pathlib
import sys
import numpy as np
import pandas as pd

# Project root setup
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.evaluator import evaluate_policy
from evaluation.report import generate_aggregate_report
from policy.cost_config import ACTION_COSTS
from ml.decision import CausalUpliftPolicy

# Output directory for Day 4 evaluation artifacts
ML_EVAL_DIR = _REPO_ROOT / "ml" / "evaluation"


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


def run_evaluation_for_seed(
    seed_name: str,
    obs_path: pathlib.Path,
    gt_path: pathlib.Path,
    causal_policy: CausalUpliftPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run 5-policy evaluation on a specific seed dataset."""
    print(f"\n{'=' * 70}")
    print(f"  EVALUATING 5 POLICIES ON {seed_name.upper()}")
    print(f"{'=' * 70}")

    obs_df = pd.read_csv(obs_path)
    gt_df = pd.read_csv(gt_path)

    print(f"Loaded observable dataset: {obs_path} ({len(obs_df):,} rows)")
    print(f"Loaded ground-truth dataset: {gt_path} ({len(gt_df):,} rows)")

    n_obs = len(obs_df)
    pids = obs_df["payment_id"].tolist()

    records = []

    # 1. WaitPolicy
    records.extend([{"payment_id": pid, "policy_name": "WaitPolicy", "chosen_action": "WAIT"} for pid in pids])

    # 2. AlwaysRetryPolicy
    records.extend([{"payment_id": pid, "policy_name": "AlwaysRetryPolicy", "chosen_action": "RETRY"} for pid in pids])

    # 3. AlwaysNudgePolicy
    records.extend([{"payment_id": pid, "policy_name": "AlwaysNudgePolicy", "chosen_action": "RETRY_NUDGE"} for pid in pids])

    # 4. RuleBasedPolicy
    rb_actions = get_rule_based_actions(obs_df)
    records.extend([{"payment_id": pid, "policy_name": "RuleBasedPolicy", "chosen_action": act} for pid, act in zip(pids, rb_actions)])

    # 5. CausalUpliftPolicy
    causal_actions = causal_policy.decide_batch(obs_df)
    records.extend([{"payment_id": pid, "policy_name": "CausalUpliftPolicy", "chosen_action": act.value} for pid, act in zip(pids, causal_actions)])

    decisions_df = pd.DataFrame(records)
    print(f"Total decision records generated: {len(decisions_df):,} (5 × {n_obs:,})")

    # Run Day 3 evaluator
    detailed_eval = evaluate_policy(
        policy_decisions_df=decisions_df,
        hidden_ground_truth_df=gt_df,
        observable_dataset_df=obs_df,
        cost_config=ACTION_COSTS,
    )

    out_eval_path = ML_EVAL_DIR / f"{seed_name}_detailed_evaluation.csv"
    detailed_eval.to_csv(out_eval_path, index=False)
    print(f"Saved detailed evaluation: {out_eval_path}")

    # Run Day 3 reporter
    out_report_path = ML_EVAL_DIR / f"{seed_name}_baseline_report.csv"
    report_df = generate_aggregate_report(detailed_eval, out_path=str(out_report_path))

    return detailed_eval, report_df


def main():
    print("=" * 70)
    print("  Day 4J — Final Causal Policy Evaluation")
    print("=" * 70)

    ML_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Initializing CausalUpliftPolicy (using Day 4 T-Learner)...")
    causal_policy = CausalUpliftPolicy()

    # Seed 42 evaluation
    seed42_obs = _REPO_ROOT / "data" / "v2" / "payment_scenarios.csv"
    seed42_gt = _REPO_ROOT / "data" / "v2" / "ground_truth.csv"
    eval42_df, report42_df = run_evaluation_for_seed(
        "seed42", seed42_obs, seed42_gt, causal_policy
    )

    # Seed 777 evaluation
    seed777_obs = ML_EVAL_DIR / "seed777_data" / "payment_scenarios.csv"
    seed777_gt = ML_EVAL_DIR / "seed777_data" / "ground_truth.csv"
    eval777_df, report777_df = run_evaluation_for_seed(
        "seed777", seed777_obs, seed777_gt, causal_policy
    )

    # Print final tables
    print("\n" + "=" * 70)
    print("  SEED 42 COMPARISON TABLE")
    print("=" * 70)
    print(report42_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("  SEED 777 COMPARISON TABLE")
    print("=" * 70)
    print(report777_df.to_string(index=False))


if __name__ == "__main__":
    main()
