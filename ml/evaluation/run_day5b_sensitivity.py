"""
ml/evaluation/run_day5b_sensitivity.py
========================================
Day 5B — Randomized Training-Seed Sensitivity Analysis.

Tests four training dataset random seeds:
  - 2024 (Canonical Baseline)
  - 1111 (New)
  - 7777 (New)
  - 9999 (New)

Evaluates:
  1. Per-arm GroupKFold ROC AUC & Brier scores.
  2. OOF Treatment Effect (CATE) distributions (mean/std for tau_retry, tau_nudge, tau_escalate).
  3. CausalUpliftPolicy performance across Seed 42 (In-Sample Fit Check) and Seed 777 (Out-of-Sample).
  4. Comparison against RuleBasedPolicy.
"""

from __future__ import annotations

import pathlib
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

# Project root setup
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.evaluator import evaluate_policy
from evaluation.report import generate_aggregate_report
from policy.cost_config import ACTION_COSTS
from ml.dataset import OBSERVABLE_FEATURES, load_causal_dataset
from ml.splits import create_group_kfold_splits
from ml.t_learner import ARMS, TLearner
from ml.decision import CausalUpliftPolicy
from ml.treatment_effects import estimate_potential_outcomes

ML_EVAL_DIR = _REPO_ROOT / "ml" / "evaluation"

TRAINING_SEEDS = {
    2024: _REPO_ROOT / "ml" / "data" / "causal_training_data.csv",
    1111: ML_EVAL_DIR / "training_seed1111" / "causal_training_data.csv",
    7777: ML_EVAL_DIR / "training_seed7777" / "causal_training_data.csv",
    9999: ML_EVAL_DIR / "training_seed9999" / "causal_training_data.csv",
}

EVAL_POPULATIONS = {
    "Seed 42 (In-Sample Fit Check)": {
        "obs": _REPO_ROOT / "data" / "v2" / "payment_scenarios.csv",
        "gt": _REPO_ROOT / "data" / "v2" / "ground_truth.csv",
    },
    "Seed 777 (OOS Population 1)": {
        "obs": ML_EVAL_DIR / "seed777_data" / "payment_scenarios.csv",
        "gt": ML_EVAL_DIR / "seed777_data" / "ground_truth.csv",
    },
    "Seed 555 (OOS Population 2)": {
        "obs": ML_EVAL_DIR / "seed555_data" / "payment_scenarios.csv",
        "gt": ML_EVAL_DIR / "seed555_data" / "ground_truth.csv",
    },
}


def evaluate_training_seed_cv(data_path: pathlib.Path) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Run GroupKFold cross-validation for a specific training dataset."""
    X, T, Y = load_causal_dataset(data_path)

    arm_auc_scores: dict[str, list[float]] = {arm: [] for arm in ARMS}
    arm_brier_scores: dict[str, list[float]] = {arm: [] for arm in ARMS}

    # Out-of-fold potential outcome storage
    oof_po = pd.DataFrame(
        np.nan,
        index=X.index,
        columns=["y_hat_wait", "y_hat_retry", "y_hat_retry_nudge", "y_hat_escalate"],
    )

    for train_idx, val_idx, _groups in create_group_kfold_splits(training_data_path=data_path):
        X_train, T_train, Y_train = X.iloc[train_idx], T.iloc[train_idx], Y.iloc[train_idx]
        X_val, T_val, Y_val = X.iloc[val_idx], T.iloc[val_idx], Y.iloc[val_idx]

        learner = TLearner()
        learner.fit(X_train, T_train, Y_train)

        # OOF prediction for treatment effects
        po_val = estimate_potential_outcomes(learner, X_val)
        oof_po.iloc[val_idx] = po_val.values

        # Per-arm AUC / Brier metrics
        for arm in ARMS:
            val_arm_mask = T_val == arm
            if val_arm_mask.sum() > 0:
                X_val_arm = X_val.loc[val_arm_mask]
                Y_val_arm = Y_val.loc[val_arm_mask]
                proba = learner.predict_arm_proba(X_val_arm, arm)

                if len(Y_val_arm.unique()) >= 2:
                    auc = roc_auc_score(Y_val_arm, proba)
                    arm_auc_scores[arm].append(auc)

                brier = brier_score_loss(Y_val_arm, proba)
                arm_brier_scores[arm].append(brier)

    # Summarize per-arm model quality
    arm_summary: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        aucs = arm_auc_scores[arm]
        briers = arm_brier_scores[arm]
        arm_summary[arm] = {
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "mean_brier": float(np.mean(briers)),
        }

    # OOF Treatment effects
    oof_te = pd.DataFrame(
        {
            "tau_retry": oof_po["y_hat_retry"] - oof_po["y_hat_wait"],
            "tau_nudge": oof_po["y_hat_retry_nudge"] - oof_po["y_hat_wait"],
            "tau_escalate": oof_po["y_hat_escalate"] - oof_po["y_hat_wait"],
        },
        index=oof_po.index,
    )

    return arm_summary, oof_te


def main():
    print("=" * 80)
    print("  DAY 5B — RANDOMIZED TRAINING-SEED SENSITIVITY ANALYSIS")
    print("=" * 80)

    # 1. Model Quality across training seeds
    cv_arm_results = []
    te_results = []
    fitted_policies: dict[int, CausalUpliftPolicy] = {}

    for seed, path in TRAINING_SEEDS.items():
        print(f"\nEvaluating Training Seed {seed} from {path.name}...")
        arm_summary, oof_te = evaluate_training_seed_cv(path)

        cv_arm_results.append({
            "Training Seed": seed,
            "WAIT AUC": f"{arm_summary['WAIT']['mean_auc']:.4f} ± {arm_summary['WAIT']['std_auc']:.4f}",
            "RETRY AUC": f"{arm_summary['RETRY']['mean_auc']:.4f} ± {arm_summary['RETRY']['std_auc']:.4f}",
            "NUDGE AUC": f"{arm_summary['RETRY_NUDGE']['mean_auc']:.4f} ± {arm_summary['RETRY_NUDGE']['std_auc']:.4f}",
            "ESCALATE AUC": f"{arm_summary['ESCALATE']['mean_auc']:.4f} ± {arm_summary['ESCALATE']['std_auc']:.4f}",
        })

        te_results.append({
            "Training Seed": seed,
            "tau_retry (mean ± std)": f"{oof_te['tau_retry'].mean()*100:+.2f} pp ± {oof_te['tau_retry'].std()*100:.2f} pp",
            "tau_nudge (mean ± std)": f"{oof_te['tau_nudge'].mean()*100:+.2f} pp ± {oof_te['tau_nudge'].std()*100:.2f} pp",
            "tau_escalate (mean ± std)": f"{oof_te['tau_escalate'].mean()*100:+.2f} pp ± {oof_te['tau_escalate'].std()*100:.2f} pp",
        })

        # Fit final model on 100% of this training seed's data
        X, T, Y = load_causal_dataset(path)
        learner = TLearner().fit(X, T, Y)
        fitted_policies[seed] = CausalUpliftPolicy(t_learner=learner)

    # Print Table 1: Model Quality (AUC)
    print("\n" + "=" * 80)
    print("  TABLE 1: PER-ARM MODEL QUALITY (GROUPKFOLD AUC)")
    print("=" * 80)
    df_auc = pd.DataFrame(cv_arm_results)
    print(df_auc.to_string(index=False))

    # Print Table 2: Treatment Effect Distributions
    print("\n" + "=" * 80)
    print("  TABLE 2: OUT-OF-FOLD TREATMENT EFFECT ESTIMATES")
    print("=" * 80)
    df_te = pd.DataFrame(te_results)
    print(df_te.to_string(index=False))

    # 2. Policy Evaluation & Robustness across Evaluation Populations
    policy_eval_rows = []
    comparison_rows = []

    for pop_label, pop_paths in EVAL_POPULATIONS.items():
        obs_df = pd.read_csv(pop_paths["obs"])
        gt_df = pd.read_csv(pop_paths["gt"])
        pids = obs_df["payment_id"].tolist()

        # Evaluate RuleBasedPolicy baseline for this population
        from ml.evaluation.run_day5_evaluation import get_rule_based_actions
        rb_actions = get_rule_based_actions(obs_df)
        rb_decisions = pd.DataFrame([
            {"payment_id": pid, "policy_name": "RuleBasedPolicy", "chosen_action": act}
            for pid, act in zip(pids, rb_actions)
        ])
        rb_eval = evaluate_policy(rb_decisions, gt_df, obs_df, ACTION_COSTS)
        rb_report = generate_aggregate_report(rb_eval)
        rb_net_val = float(rb_report[rb_report["policy_name"] == "RuleBasedPolicy"]["net_recovered_value"].iloc[0])

        for seed, policy in fitted_policies.items():
            causal_actions = policy.decide_batch(obs_df)
            causal_decisions = pd.DataFrame([
                {"payment_id": pid, "policy_name": f"CausalUplift_Seed{seed}", "chosen_action": act.value}
                for pid, act in zip(pids, causal_actions)
            ])

            # Concatenate with RuleBasedPolicy & WaitPolicy to run full report ranking
            combined_decisions = pd.concat([
                causal_decisions,
                rb_decisions,
                pd.DataFrame([{"payment_id": pid, "policy_name": "WaitPolicy", "chosen_action": "WAIT"} for pid in pids]),
                pd.DataFrame([{"payment_id": pid, "policy_name": "AlwaysRetryPolicy", "chosen_action": "RETRY"} for pid in pids]),
                pd.DataFrame([{"payment_id": pid, "policy_name": "AlwaysNudgePolicy", "chosen_action": "RETRY_NUDGE"} for pid in pids]),
            ])

            eval_df = evaluate_policy(combined_decisions, gt_df, obs_df, ACTION_COSTS)
            report_df = generate_aggregate_report(eval_df)

            c_row = report_df[report_df["policy_name"] == f"CausalUplift_Seed{seed}"].iloc[0]
            causal_net_val = float(c_row["net_recovered_value"])
            causal_rank = int(c_row["rank"])
            diff = causal_net_val - rb_net_val
            pct_diff = (diff / rb_net_val) * 100

            policy_eval_rows.append({
                "Training Seed": seed,
                "Evaluation Population": pop_label,
                "Recovery Rate": f"{c_row['mean_p_success']*100:.2f}%",
                "Gross Revenue (INR)": f"INR {c_row['gross_recovered_revenue']:,.2f}",
                "Total Cost (INR)": f"INR {c_row['total_action_cost']:,.2f}",
                "Net Value (INR)": f"INR {causal_net_val:,.2f}",
                "WAIT %": f"{c_row['wait_pct']:.2f}%",
                "RETRY %": f"{c_row['retry_pct']:.2f}%",
                "NUDGE %": f"{c_row['retry_nudge_pct']:.2f}%",
                "ESCALATE %": f"{c_row['escalate_pct']:.2f}%",
            })

            comparison_rows.append({
                "Training Seed": seed,
                "Evaluation Population": pop_label,
                "Causal Net Value": f"INR {causal_net_val:,.2f}",
                "RuleBased Net Value": f"INR {rb_net_val:,.2f}",
                "Difference (INR)": f"{diff:+,.2f}",
                "% Difference": f"{pct_diff:+.2f}%",
                "Causal Rank": f"#{causal_rank}",
            })

    # Print Table 3: Detailed Policy Performance
    print("\n" + "=" * 80)
    print("  TABLE 3: CAUSAL UPLIFT POLICY EVALUATION ACROSS TRAINING SEEDS")
    print("=" * 80)
    df_policy = pd.DataFrame(policy_eval_rows)
    print(df_policy.to_string(index=False))

    # Print Table 4: Robustness Comparison
    print("\n" + "=" * 80)
    print("  TABLE 4: COMPACT CAUSAL VS RULEBASED ROBUSTNESS COMPARISON")
    print("=" * 80)
    df_comp = pd.DataFrame(comparison_rows)
    print(df_comp.to_string(index=False))

    # Save artifact tables
    out_csv = ML_EVAL_DIR / "day5b_sensitivity_report.csv"
    df_comp.to_csv(out_csv, index=False)
    print(f"\nSaved Day 5B sensitivity report artifact: {out_csv}")


if __name__ == "__main__":
    main()
