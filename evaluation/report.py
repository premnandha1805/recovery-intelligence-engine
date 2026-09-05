"""
evaluation/report.py — Aggregates policy evaluation metrics and builds baseline reports.

Features
--------
1. Aggregates per-payment policy evaluation data (policy_evaluation.csv).
2. Computes:
   - Payments count
   - Mean expected success probability (informative recovery metric)
   - Trivial recovery rate (fraction with expected_recovery > 0)
   - Gross recovered revenue (INR)
   - Total action cost (INR)
   - Net recovered value (INR)
   - Action distribution (WAIT %, RETRY %, RETRY_NUDGE %, ESCALATE %)
3. Ranks policies strictly by net_recovered_value.
4. Generates structured aggregate reports for seed 42, seed 777, or custom runs.

Usage
-----
    python evaluation/report.py --input evaluation/output/policy_evaluation.csv --out evaluation/output/baseline_report_seed42.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.schemas import Action

DEFAULT_EVAL_INPUT = os.path.join(PROJECT_ROOT, "evaluation", "output", "policy_evaluation.csv")
DEFAULT_REPORT_OUT = os.path.join(PROJECT_ROOT, "evaluation", "output", "baseline_report_seed42.csv")


def generate_aggregate_report(
    eval_df: pd.DataFrame,
    out_path: str | None = None,
) -> pd.DataFrame:
    """
    Build per-policy aggregate report dataframe from detailed evaluation rows.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Detailed policy evaluation dataframe.
    out_path : str, optional
        Path to save CSV output.

    Returns
    -------
    pd.DataFrame
        Aggregated report table.
    """
    required_cols = {
        "payment_id", "policy_name", "chosen_action", "amount",
        "p_success_for_chosen_action", "action_cost", "expected_recovery",
        "expected_net_value"
    }
    missing = required_cols - set(eval_df.columns)
    if missing:
        raise KeyError(f"eval_df missing required columns: {sorted(missing)}")

    rows = []
    for pol_name, group in eval_df.groupby("policy_name"):
        n_p = len(group)
        mean_p_success = group["p_success_for_chosen_action"].mean()
        rec_gt0_rate = (group["expected_recovery"] > 0).mean()
        gross_rev = group["expected_recovery"].sum()
        tot_cost = group["action_cost"].sum()
        net_val = group["expected_net_value"].sum()

        counts = group["chosen_action"].value_counts().to_dict()
        p_wait    = (counts.get(Action.WAIT.value, 0) / n_p) * 100
        p_retry   = (counts.get(Action.RETRY.value, 0) / n_p) * 100
        p_nudge   = (counts.get(Action.RETRY_NUDGE.value, 0) / n_p) * 100
        p_escalate = (counts.get(Action.ESCALATE.value, 0) / n_p) * 100

        rows.append({
            "policy_name": pol_name,
            "n_payments": n_p,
            "mean_p_success": round(mean_p_success, 6),
            "recovery_rate_gt0": round(rec_gt0_rate, 4),
            "gross_recovered_revenue": round(gross_rev, 2),
            "total_action_cost": round(tot_cost, 2),
            "net_recovered_value": round(net_val, 2),
            "wait_pct": round(p_wait, 2),
            "retry_pct": round(p_retry, 2),
            "retry_nudge_pct": round(p_nudge, 2),
            "escalate_pct": round(p_escalate, 2),
        })

    report_df = pd.DataFrame(rows)

    # Rank strictly by net_recovered_value descending
    report_df = report_df.sort_values(by="net_recovered_value", ascending=False).reset_index(drop=True)
    report_df["rank"] = range(1, len(report_df) + 1)

    # Reorder columns
    cols = [
        "rank", "policy_name", "n_payments", "mean_p_success", "recovery_rate_gt0",
        "gross_recovered_revenue", "total_action_cost", "net_recovered_value",
        "wait_pct", "retry_pct", "retry_nudge_pct", "escalate_pct"
    ]
    report_df = report_df[cols]

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        report_df.to_csv(out_path, index=False)
        print(f"Saved aggregate baseline report: {out_path}")

    return report_df


def main():
    parser = argparse.ArgumentParser(description="Aggregate policy evaluation into baseline report.")
    parser.add_argument("--input", default=DEFAULT_EVAL_INPUT, help="Path to policy_evaluation.csv")
    parser.add_argument("--out", default=DEFAULT_REPORT_OUT, help="Path to save aggregate report CSV")
    args = parser.parse_args()

    eval_df = pd.read_csv(args.input)
    report_df = generate_aggregate_report(eval_df, args.out)

    print("\n--- AGGREGATE BASELINE REPORT ---")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
