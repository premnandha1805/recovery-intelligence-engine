"""
ml/policy_overlap_diagnostic.py
================================
Day 4H — RuleBasedPolicy Action Distribution / Overlap Diagnostic.

This script performs a read-only analysis of the actual treatment/action
assignments produced by Day 3's RuleBasedPolicy in
`simulator/output/policy_decisions.csv`.

Purpose
-------
Diagnose action coverage and overlap in historical baseline decisions.
This is strictly an overlap diagnostic.

Immutability & Safety
---------------------
- Read-only analysis of simulator/output/policy_decisions.csv.
- Does NOT use ml/data/causal_training_data.csv (which is randomized).
- Does NOT modify any policy, evaluator, model, or dataset.
- Does NOT restrict or filter any treatment arm.
"""

from __future__ import annotations

import pathlib
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
POLICY_DECISIONS_PATH = _REPO_ROOT / "simulator" / "output" / "policy_decisions.csv"

EXPECTED_ACTIONS = ["WAIT", "RETRY", "RETRY_NUDGE", "ESCALATE"]


def analyze_rule_based_policy_overlap(
    csv_path: pathlib.Path = POLICY_DECISIONS_PATH,
) -> dict:
    """Analyze action distribution for RuleBasedPolicy from policy_decisions.csv.

    Returns
    -------
    dict containing counts, percentages, checks, and metadata.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Policy decisions file not found at: {csv_path}")

    full_df = pd.read_csv(csv_path)

    if "policy_name" not in full_df.columns:
        raise ValueError("Missing 'policy_name' column in policy_decisions.csv")
    if "chosen_action" not in full_df.columns:
        raise ValueError("Missing 'chosen_action' column in policy_decisions.csv")

    # Filter strictly for RuleBasedPolicy
    rb_df = full_df[full_df["policy_name"] == "RuleBasedPolicy"].copy()

    total_rows = len(rb_df)
    n_unique_payments = rb_df["payment_id"].nunique() if "payment_id" in rb_df.columns else total_rows
    n_dupe_payments = total_rows - n_unique_payments

    # Action counts
    action_counts = rb_df["chosen_action"].value_counts().to_dict()

    # Detect unknown/unexpected actions
    unknown_actions = {
        act: count for act, count in action_counts.items() if act not in EXPECTED_ACTIONS
    }
    n_unknown = sum(unknown_actions.values())
    n_missing = int(rb_df["chosen_action"].isna().sum())

    # Build exact table for expected actions
    table_data = {}
    for act in EXPECTED_ACTIONS:
        cnt = action_counts.get(act, 0)
        pct = (cnt / total_rows * 100.0) if total_rows > 0 else 0.0
        table_data[act] = {"count": cnt, "percentage": pct}

    return {
        "source_file": str(csv_path.relative_to(_REPO_ROOT)),
        "total_file_rows": len(full_df),
        "rule_based_rows": total_rows,
        "unique_payments": n_unique_payments,
        "duplicate_payments": n_dupe_payments,
        "missing_actions": n_missing,
        "unknown_actions": unknown_actions,
        "n_unknown": n_unknown,
        "table_data": table_data,
    }


def main() -> None:
    sep = "=" * 70
    print(sep)
    print("  Day 4H — RuleBasedPolicy Action Distribution / Overlap Diagnostic")
    print(sep)

    results = analyze_rule_based_policy_overlap()

    print(f"\n1. SOURCE FILE & SCHEMA IDENTIFICATION")
    print(f"   Source file         : {results['source_file']}")
    print(f"   Total file rows     : {results['total_file_rows']:,} (across all policies)")
    print(f"   RuleBasedPolicy rows: {results['rule_based_rows']:,}")
    print(f"   Unique payment IDs  : {results['unique_payments']:,}")
    print(f"   Duplicate payment IDs: {results['duplicate_payments']}")
    print(f"   Action column       : 'chosen_action'")
    print(f"   Missing/null actions: {results['missing_actions']}")
    print(f"   Unknown action types: {results['n_unknown']}")

    print(f"\n2. ACTUAL RuleBasedPolicy ACTION DISTRIBUTION")
    print("-" * 55)
    print(f"{'Action':<15} | {'Count':>12} | {'Percentage':>12}")
    print("-" * 55)

    sum_count = 0
    sum_pct = 0.0

    for act in EXPECTED_ACTIONS:
        data = results["table_data"][act]
        cnt = data["count"]
        pct = data["percentage"]
        sum_count += cnt
        sum_pct += pct
        print(f"{act:<15} | {cnt:>12,} | {pct:>11.2f}%")

    print("-" * 55)
    print(f"{'TOTAL':<15} | {sum_count:>12,} | {sum_pct:>11.2f}%")
    print("-" * 55)

    # Sanity checks
    assert sum_count == results["rule_based_rows"], "Action count sum mismatch!"
    assert abs(sum_pct - 100.0) < 1e-4, "Percentage sum mismatch!"

    print(f"\n3. OVERLAP DIAGNOSIS & INTERPRETATION")
    print("-" * 70)

    zero_overlap_arms = []
    low_overlap_arms = []

    for act in EXPECTED_ACTIONS:
        cnt = results["table_data"][act]["count"]
        pct = results["table_data"][act]["percentage"]
        if cnt == 0:
            zero_overlap_arms.append(act)
            print(f"  [!] ZERO OVERLAP under RuleBasedPolicy for {act}")
        elif pct < 5.0:
            low_overlap_arms.append((act, cnt, pct))
            print(f"  [*] LIMITED OVERLAP under RuleBasedPolicy for {act}: {cnt:,} ({pct:.2f}%)")
        else:
            print(f"  [OK] Substantial representation for {act}: {cnt:,} ({pct:.2f}%)")

    print("\n  Summary Interpretation:")
    print("  - Day 4B Randomized Training Data: ~25% uniform assignment per arm by construction.")
    print("  - Day 3 RuleBasedPolicy: Actual heuristic policy assignment distribution.")

    if zero_overlap_arms:
        for arm in zero_overlap_arms:
            print(f"  -> ZERO OVERLAP under RuleBasedPolicy for {arm}")
    else:
        print("  - All four treatment arms receive non-zero allocations under RuleBasedPolicy.")

    print("\n4. DATA GOVERNANCE & SAFEGUARDS")
    print("  - Read-only analysis; Day 4B randomized data was NOT used.")
    print("  - No protected files were modified.")
    print("  - No treatment arm filtering or restriction applied.")
    print(sep)


if __name__ == "__main__":
    main()
