"""
policy/run_policies.py — Run every policy over the observable dataset.

Pipeline
--------
1. Load the observable dataset (payment_scenarios.csv — no hidden columns).
2. Run each policy row-by-row; record (payment_id, policy_name, chosen_action).
3. Assert that every policy was evaluated against the identical set of
   payment_ids.
4. Write simulator/output/policy_decisions.csv.
5. Print a preview (first 10 rows) and per-policy row counts.

Usage
-----
    python policy/run_policies.py
    python policy/run_policies.py --obs path/to/observable.csv --out path/to/decisions.csv

Input
-----
    data/raw/payment_scenarios.csv   (observable features — Day 1 simulator output)

Output
------
    simulator/output/policy_decisions.csv
        Columns: payment_id | policy_name | chosen_action
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# ── Project root on path ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from policy import (
    AlwaysNudgePolicy,
    AlwaysRetryPolicy,
    RuleBasedPolicy,
    WaitPolicy,
)

# ── Path constants ────────────────────────────────────────────────────────
DEFAULT_OBS_PATH = os.path.join(PROJECT_ROOT, "data", "v2", "payment_scenarios.csv")
DEFAULT_OUT_DIR  = os.path.join(PROJECT_ROOT, "simulator", "output")
DEFAULT_OUT_PATH = os.path.join(DEFAULT_OUT_DIR, "policy_decisions.csv")

# ── Policies to evaluate (name → instance) ───────────────────────────────
POLICIES: dict[str, object] = {
    "WaitPolicy":        WaitPolicy(),
    "AlwaysRetryPolicy": AlwaysRetryPolicy(),
    "AlwaysNudgePolicy": AlwaysNudgePolicy(),
    "RuleBasedPolicy":   RuleBasedPolicy(),
}


def run(obs_path: str, out_path: str) -> pd.DataFrame:
    """
    Load the observable dataset, run every policy, write decisions CSV.

    Parameters
    ----------
    obs_path : str
        Path to the observable payment scenarios CSV.
    out_path : str
        Destination path for policy_decisions.csv.

    Returns
    -------
    pd.DataFrame
        The full decisions table (payment_id, policy_name, chosen_action).
    """
    # ── 1. Load observable dataset ────────────────────────────────────────
    print(f"Loading observable dataset: {obs_path}")
    obs_df = pd.read_csv(obs_path)
    print(f"  {len(obs_df):,} rows × {len(obs_df.columns)} columns")

    # Hard guard: observable dataset must NOT contain hidden columns
    hidden_cols = [c for c in obs_df.columns if "p_success" in c.lower() or "hidden" in c.lower()]
    assert not hidden_cols, (
        f"Observable dataset contains hidden columns: {hidden_cols}. "
        f"run_policies.py must not be given the ground-truth file."
    )

    all_payment_ids = obs_df["payment_id"].tolist()

    # ── 2. Run every policy over every row ────────────────────────────────
    records: list[dict] = []
    policy_id_sets: dict[str, set] = {}

    for policy_name, policy in POLICIES.items():
        print(f"  Running {policy_name}...", end=" ", flush=True)
        seen_ids: set[str] = set()

        for _, row in obs_df.iterrows():
            features = row.to_dict()
            action = policy.decide(features)  # type: ignore[attr-defined]
            pid = features["payment_id"]
            records.append({
                "payment_id":    pid,
                "policy_name":   policy_name,
                "chosen_action": action.value,
            })
            seen_ids.add(pid)

        policy_id_sets[policy_name] = seen_ids
        print(f"done ({len(seen_ids):,} payments)")

    # ── 3. Assert identical payment_id coverage across all policies ───────
    reference_name = next(iter(policy_id_sets))
    reference_ids  = policy_id_sets[reference_name]
    for pname, pids in policy_id_sets.items():
        assert pids == reference_ids, (
            f"Payment ID mismatch: {reference_name!r} has {len(reference_ids):,} IDs "
            f"but {pname!r} has {len(pids):,}. "
            f"Symmetric difference: {pids.symmetric_difference(reference_ids)}"
        )
    print(f"\n[OK] All {len(POLICIES)} policies evaluated over identical "
          f"{len(reference_ids):,} payment_ids.")

    # ── 4. Build DataFrame and write CSV ──────────────────────────────────
    decisions_df = pd.DataFrame(records, columns=["payment_id", "policy_name", "chosen_action"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    decisions_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(decisions_df):,} rows -> {out_path}")

    # -- 5. Preview --------------------------------------------------------
    print("\n-- First 10 rows of policy_decisions.csv ------------------")
    print(decisions_df.head(10).to_string(index=False))

    print("\n-- Row count per policy -----------------------------------")
    counts = decisions_df.groupby("policy_name").size().reset_index(name="n_rows")
    print(counts.to_string(index=False))

    print("\n-- Action distribution per policy -------------------------")
    dist = pd.crosstab(decisions_df["policy_name"], decisions_df["chosen_action"])
    print(dist.to_string())

    return decisions_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all policies over the observable payment dataset."
    )
    parser.add_argument(
        "--obs",
        default=DEFAULT_OBS_PATH,
        help=f"Path to observable CSV (default: {DEFAULT_OBS_PATH})",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUT_PATH})",
    )
    args = parser.parse_args()
    run(args.obs, args.out)


if __name__ == "__main__":
    main()
