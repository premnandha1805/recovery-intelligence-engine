"""
ml/evaluation/run_day5c_cost_sensitivity.py
=============================================
Day 5C — Cost Sensitivity Analysis for CausalUpliftPolicy.

Evaluates how sensitive CausalUpliftPolicy's decisions and performance are
to assumed intervention costs across 4 hypothetical cost scenarios:

1. Baseline:                WAIT=0, RETRY=5, RETRY_NUDGE=15, ESCALATE=250
2. Lower Escalation Cost:   WAIT=0, RETRY=5, RETRY_NUDGE=15, ESCALATE=100
3. Higher Escalation Cost:  WAIT=0, RETRY=5, RETRY_NUDGE=15, ESCALATE=500
4. Zero Intervention Costs: WAIT=0, RETRY=0, RETRY_NUDGE=0,   ESCALATE=0

Evaluates on:
  - Seed 42 (In-Sample Fit-Quality Check)
  - Seed 777 (Out-of-Sample Population 1)
  - Seed 555 (Out-of-Sample Population 2)

Does NOT modify policy/cost_config.py, evaluation/evaluator.py, or policy files.
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
from evaluation.report import generate_aggregate_report
from models.schemas import Action
from ml.dataset import OBSERVABLE_FEATURES
from ml.t_learner import TLearner, fit_final_models

ML_EVAL_DIR = _REPO_ROOT / "ml" / "evaluation"

ARM_TO_ACTION: dict[str, Action] = {
    "WAIT": Action.WAIT,
    "RETRY": Action.RETRY,
    "RETRY_NUDGE": Action.RETRY_NUDGE,
    "ESCALATE": Action.ESCALATE,
}

COST_SCENARIOS = {
    "Baseline (ESCALATE=250)": {
        Action.WAIT: 0.0, Action.RETRY: 5.0, Action.RETRY_NUDGE: 15.0, Action.ESCALATE: 250.0, Action.STOP: 0.0,
    },
    "Lower Escalation Cost (ESCALATE=100)": {
        Action.WAIT: 0.0, Action.RETRY: 5.0, Action.RETRY_NUDGE: 15.0, Action.ESCALATE: 100.0, Action.STOP: 0.0,
    },
    "Higher Escalation Cost (ESCALATE=500)": {
        Action.WAIT: 0.0, Action.RETRY: 5.0, Action.RETRY_NUDGE: 15.0, Action.ESCALATE: 500.0, Action.STOP: 0.0,
    },
    "Zero Intervention Costs (all=0)": {
        Action.WAIT: 0.0, Action.RETRY: 0.0, Action.RETRY_NUDGE: 0.0, Action.ESCALATE: 0.0, Action.STOP: 0.0,
    },
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


def decide_batch_custom_costs(
    t_learner: TLearner,
    df: pd.DataFrame,
    costs: dict[Action, float],
) -> pd.Series:
    """Compute decisions for a batch of payments given custom action costs."""
    X = df[OBSERVABLE_FEATURES]
    amounts = df["amount"].astype(float) if "amount" in df.columns else pd.Series(1.0, index=df.index)

    probas = t_learner.predict_proba(X)

    net_values = pd.DataFrame(index=df.index)
    for arm_name, action_enum in ARM_TO_ACTION.items():
        cost = float(costs[action_enum])
        net_values[arm_name] = (probas[arm_name] * amounts) - cost

    best_arms = net_values.idxmax(axis=1)
    return best_arms.map(ARM_TO_ACTION)


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


def main():
    print("=" * 80)
    print("  DAY 5C — COST SENSITIVITY ANALYSIS FOR CAUSAL UPLIFT POLICY")
    print("=" * 80)

    print("\nLoading canonical fitted T-Learner model...")
    t_learner = fit_final_models()

    detailed_results = []
    comparison_results = []

    for scenario_name, scenario_costs in COST_SCENARIOS.items():
        print(f"\n{'=' * 75}")
        print(f"  SCENARIO: {scenario_name}")
        print(f"  Costs: {scenario_costs}")
        print(f"{'=' * 75}")

        for pop_name, pop_paths in EVAL_POPULATIONS.items():
            obs_df = pd.read_csv(pop_paths["obs"])
            gt_df = pd.read_csv(pop_paths["gt"])
            pids = obs_df["payment_id"].tolist()

            # 1. Generate CausalUpliftPolicy decisions using scenario costs
            causal_actions = decide_batch_custom_costs(t_learner, obs_df, scenario_costs)

            # 2. RuleBasedPolicy actions (fixed risk rules)
            rb_actions = get_rule_based_actions(obs_df)

            # 3. Concatenate all 5 policies for complete ranking report
            decisions_records = []
            decisions_records.extend([{"payment_id": pid, "policy_name": "WaitPolicy", "chosen_action": "WAIT"} for pid in pids])
            decisions_records.extend([{"payment_id": pid, "policy_name": "AlwaysRetryPolicy", "chosen_action": "RETRY"} for pid in pids])
            decisions_records.extend([{"payment_id": pid, "policy_name": "AlwaysNudgePolicy", "chosen_action": "RETRY_NUDGE"} for pid in pids])
            decisions_records.extend([{"payment_id": pid, "policy_name": "RuleBasedPolicy", "chosen_action": act} for pid, act in zip(pids, rb_actions)])
            decisions_records.extend([{"payment_id": pid, "policy_name": "CausalUpliftPolicy", "chosen_action": act.value} for pid, act in zip(pids, causal_actions)])

            decisions_df = pd.DataFrame(decisions_records)

            # 4. Evaluate using exact evaluator and reporter
            eval_df = evaluate_policy(
                policy_decisions_df=decisions_df,
                hidden_ground_truth_df=gt_df,
                observable_dataset_df=obs_df,
                cost_config=scenario_costs,
            )

            report_df = generate_aggregate_report(eval_df)

            # Extract Causal and RuleBased metrics
            causal_row = report_df[report_df["policy_name"] == "CausalUpliftPolicy"].iloc[0]
            rb_row = report_df[report_df["policy_name"] == "RuleBasedPolicy"].iloc[0]

            causal_net = float(causal_row["net_recovered_value"])
            rb_net = float(rb_row["net_recovered_value"])
            diff_net = causal_net - rb_net
            pct_diff = (diff_net / rb_net) * 100 if rb_net > 0 else 0.0

            detailed_results.append({
                "Scenario": scenario_name,
                "Population": pop_name,
                "Policy": "CausalUpliftPolicy",
                "Rank": int(causal_row["rank"]),
                "Recovery Rate": f"{causal_row['mean_p_success']*100:.2f}%",
                "Gross Revenue (INR)": f"INR {causal_row['gross_recovered_revenue']:,.2f}",
                "Total Cost (INR)": f"INR {causal_row['total_action_cost']:,.2f}",
                "Net Value (INR)": f"INR {causal_net:,.2f}",
                "WAIT %": f"{causal_row['wait_pct']:.2f}%",
                "RETRY %": f"{causal_row['retry_pct']:.2f}%",
                "NUDGE %": f"{causal_row['retry_nudge_pct']:.2f}%",
                "ESCALATE %": f"{causal_row['escalate_pct']:.2f}%",
            })

            detailed_results.append({
                "Scenario": scenario_name,
                "Population": pop_name,
                "Policy": "RuleBasedPolicy",
                "Rank": int(rb_row["rank"]),
                "Recovery Rate": f"{rb_row['mean_p_success']*100:.2f}%",
                "Gross Revenue (INR)": f"INR {rb_row['gross_recovered_revenue']:,.2f}",
                "Total Cost (INR)": f"INR {rb_row['total_action_cost']:,.2f}",
                "Net Value (INR)": f"INR {rb_net:,.2f}",
                "WAIT %": f"{rb_row['wait_pct']:.2f}%",
                "RETRY %": f"{rb_row['retry_pct']:.2f}%",
                "NUDGE %": f"{rb_row['retry_nudge_pct']:.2f}%",
                "ESCALATE %": f"{rb_row['escalate_pct']:.2f}%",
            })

            comparison_results.append({
                "Scenario": scenario_name,
                "Population": pop_name,
                "Causal Net Value": f"INR {causal_net:,.2f}",
                "RuleBased Net Value": f"INR {rb_net:,.2f}",
                "Difference (INR)": f"{diff_net:+,.2f}",
                "% Difference": f"{pct_diff:+.2f}%",
                "Causal Rank": f"#{int(causal_row['rank'])}",
                "Causal ESCALATE %": f"{causal_row['escalate_pct']:.2f}%",
            })

    # Print Full Scenario Reports
    df_det = pd.DataFrame(detailed_results)
    df_comp = pd.DataFrame(comparison_results)

    print("\n" + "=" * 80)
    print("  TABLE 1: DETAILED POLICY METRICS ACROSS COST SCENARIOS")
    print("=" * 80)
    print(df_det.to_string(index=False))

    print("\n" + "=" * 80)
    print("  TABLE 2: SUMMARY CAUSAL VS RULEBASED PERFORMANCE ACROSS COST SCENARIOS")
    print("=" * 80)
    print(df_comp.to_string(index=False))

    # Save artifact tables
    out_csv = ML_EVAL_DIR / "day5c_cost_sensitivity_summary.csv"
    out_det_csv = ML_EVAL_DIR / "day5c_cost_sensitivity_detailed.csv"
    df_comp.to_csv(out_csv, index=False)
    df_det.to_csv(out_det_csv, index=False)
    print(f"\nSaved Day 5C sensitivity report artifacts:")
    print(f"  - {out_csv}")
    print(f"  - {out_det_csv}")


if __name__ == "__main__":
    main()
