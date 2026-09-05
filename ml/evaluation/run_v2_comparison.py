"""
ml/evaluation/run_v2_comparison.py
===================================
6-Way Policy Evaluation & Control for Escalation-Threshold Confound.

Evaluates 6 policies across OOS seeds 777 and 555:
  1. WaitPolicy (Always WAIT)
  2. AlwaysRetryPolicy (Always RETRY)
  3. AlwaysNudgePolicy (Always RETRY_NUDGE)
  4. RuleBasedPolicy (v1 - Threshold-based heuristic)
  5. RuleBasedPolicyV2 (v2 - Economically cost-aware heuristic)
  6. CausalUpliftPolicy (Learned T-Learner ML)

Examines:
  - 6-way comparison table on Seed 777 & Seed 555
  - Value decomposition: RuleBasedPolicyV2 -> CausalUpliftPolicy
"""

from __future__ import annotations

import pathlib
import sys
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.evaluator import evaluate_policy
from policy.cost_config import ACTION_COSTS
from policy.rule_based_policy import RuleBasedPolicy
from policy.rule_based_policy_v2 import RuleBasedPolicyV2
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
}


def run_6_way_comparison():
    print("=" * 80)
    print("  DAY 5 AUDIT — 6-WAY POLICY COMPARISON & V2 CONFOUND CONTROL")
    print("=" * 80)

    print("\nInitializing policies...")
    rb_v1 = RuleBasedPolicy()
    rb_v2 = RuleBasedPolicyV2()
    causal_policy = CausalUpliftPolicy()

    all_summaries = []

    for pop_name, pop_paths in EVAL_POPULATIONS.items():
        obs_df = pd.read_csv(pop_paths["obs"])
        gt_df = pd.read_csv(pop_paths["gt"])
        pids = obs_df["payment_id"].tolist()
        n_total = len(obs_df)

        # Generate decisions for all 6 policies
        decisions_dict = {
            "WaitPolicy": ["WAIT"] * n_total,
            "AlwaysRetryPolicy": ["RETRY"] * n_total,
            "AlwaysNudgePolicy": ["RETRY_NUDGE"] * n_total,
            "RuleBasedPolicy (v1)": [
                rb_v1.decide(row.to_dict()).value for _, row in obs_df.iterrows()
            ],
            "RuleBasedPolicyV2 (v2)": rb_v2.decide_batch(obs_df),
            "CausalUpliftPolicy": [
                act.value for act in causal_policy.decide_batch(obs_df)
            ],
        }

        eval_results = []
        for pname, action_list in decisions_dict.items():
            decs_df = pd.DataFrame([
                {"payment_id": pid, "policy_name": pname, "chosen_action": act}
                for pid, act in zip(pids, action_list)
            ])
            res = evaluate_policy(decs_df, gt_df, obs_df, ACTION_COSTS)
            res["policy_name"] = pname

            # Action distribution
            act_counts = decs_df["chosen_action"].value_counts().to_dict()
            esc_count = act_counts.get("ESCALATE", 0)
            esc_pct = (esc_count / n_total) * 100

            rec_rate = res["p_success_for_chosen_action"].mean() * 100
            gross_rev = res["expected_recovery"].sum()
            total_cost = res["action_cost"].sum()
            net_val = res["expected_net_value"].sum()

            eval_results.append({
                "Population": pop_name,
                "Policy": pname,
                "Net Value (INR)": net_val,
                "Gross Revenue (INR)": gross_rev,
                "Total Cost (INR)": total_cost,
                "Recovery Rate": rec_rate,
                "Escalation Rate": esc_pct,
            })

        pop_df = pd.DataFrame(eval_results)
        pop_df = pop_df.sort_values(by="Net Value (INR)", ascending=False).reset_index(drop=True)
        pop_df["Rank"] = range(1, len(pop_df) + 1)

        # Formatting
        pop_df["Net Value (INR)"] = pop_df["Net Value (INR)"].apply(lambda x: f"INR {x:,.2f}")
        pop_df["Gross Revenue (INR)"] = pop_df["Gross Revenue (INR)"].apply(lambda x: f"INR {x:,.2f}")
        pop_df["Total Cost (INR)"] = pop_df["Total Cost (INR)"].apply(lambda x: f"INR {x:,.2f}")
        pop_df["Recovery Rate"] = pop_df["Recovery Rate"].apply(lambda x: f"{x:.2f}%")
        pop_df["Escalation Rate"] = pop_df["Escalation Rate"].apply(lambda x: f"{x:.2f}%")

        print(f"\n{'=' * 80}")
        print(f"  6-WAY POLICY PERFORMANCE: {pop_name.upper()}")
        print(f"{'=' * 80}")
        cols_order = [
            "Rank", "Policy", "Net Value (INR)", "Recovery Rate", "Total Cost (INR)",
            "Escalation Rate", "Gross Revenue (INR)"
        ]
        print(pop_df[cols_order].to_string(index=False))
        all_summaries.append(pop_df)

    # Save summary artifact
    combined_df = pd.concat(all_summaries, ignore_index=True)
    out_csv = ML_EVAL_DIR / "rule_based_v2_comparison_summary.csv"
    combined_df.to_csv(out_csv, index=False)
    print(f"\nSaved 6-way comparison artifact: {out_csv}")


def run_v2_value_decomposition():
    print("\n" + "=" * 80)
    print("  TRANSITION VALUE DECOMPOSITION: RuleBasedPolicyV2 -> CausalUpliftPolicy")
    print("=" * 80)

    rb_v2 = RuleBasedPolicyV2()
    causal_policy = CausalUpliftPolicy()

    for pop_name, pop_paths in EVAL_POPULATIONS.items():
        obs_df = pd.read_csv(pop_paths["obs"])
        gt_df = pd.read_csv(pop_paths["gt"])
        pids = obs_df["payment_id"].tolist()
        n_total = len(obs_df)

        v2_actions = rb_v2.decide_batch(obs_df)
        causal_actions = [act.value for act in causal_policy.decide_batch(obs_df)]

        v2_decs = pd.DataFrame([
            {"payment_id": pid, "policy_name": "RuleBasedPolicyV2", "chosen_action": act}
            for pid, act in zip(pids, v2_actions)
        ])
        causal_decs = pd.DataFrame([
            {"payment_id": pid, "policy_name": "CausalUpliftPolicy", "chosen_action": act}
            for pid, act in zip(pids, causal_actions)
        ])

        v2_eval = evaluate_policy(v2_decs, gt_df, obs_df, ACTION_COSTS)
        causal_eval = evaluate_policy(causal_decs, gt_df, obs_df, ACTION_COSTS)

        merged = causal_eval.rename(columns={
            "chosen_action": "action_causal",
            "p_success_for_chosen_action": "p_success_causal",
            "action_cost": "cost_causal",
            "expected_recovery": "recovery_causal",
            "expected_net_value": "net_val_causal",
        }).merge(
            v2_eval.rename(columns={
                "chosen_action": "action_v2",
                "p_success_for_chosen_action": "p_success_v2",
                "action_cost": "cost_v2",
                "expected_recovery": "recovery_v2",
                "expected_net_value": "net_val_v2",
            }),
            on="payment_id",
            how="inner",
        )

        merged["transition"] = merged["action_v2"] + " -> " + merged["action_causal"]
        merged["recovery_diff"] = merged["recovery_causal"] - merged["recovery_v2"]
        merged["cost_diff"] = merged["cost_causal"] - merged["cost_v2"]
        merged["net_val_diff"] = merged["net_val_causal"] - merged["net_val_v2"]

        trans_groups = []
        for trans, grp in merged.groupby("transition"):
            n_p = len(grp)
            pct_p = (n_p / n_total) * 100
            tot_net = grp["net_val_diff"].sum()
            tot_rec = grp["recovery_diff"].sum()
            tot_cost = grp["cost_diff"].sum()

            trans_groups.append({
                "Transition (v2 -> Causal)": trans,
                "Payment Count": n_p,
                "% Total": f"{pct_p:.2f}%",
                "Total Net Value Diff (INR)": tot_net,
                "Total Recovery Diff (INR)": tot_rec,
                "Total Cost Diff (INR)": tot_cost,
                "Mean Net Diff": tot_net / n_p,
            })

        tdf = pd.DataFrame(trans_groups).sort_values(by="Total Net Value Diff (INR)", ascending=False)
        print(f"\n--- {pop_name} Transition Breakdown ---")
        print(tdf.to_string(index=False))


def main():
    run_6_way_comparison()
    run_v2_value_decomposition()


if __name__ == "__main__":
    main()
