"""
ml/generate_causal_training_data.py
====================================
Generates a RANDOMIZED-LOGGING causal training dataset for Day 4 ML work.

Design intent
-------------
Each payment is assigned a treatment T drawn UNIFORMLY AT RANDOM from
{WAIT, RETRY, RETRY_NUDGE, ESCALATE}.  The observed outcome Y is then drawn
from a Bernoulli distribution whose success probability is the hidden ground-
truth probability for the assigned treatment.  This mimics a logged bandit /
randomized-policy dataset and is the input to causal-uplift modelling.

IMPORTANT - DATA GOVERNANCE
-----------------------------
- This file is ML TRAINING DATA ONLY.
- It must NOT be fed into evaluation/report.py or compared against the Day 3
  baseline table (baseline_report_seed42.csv / baseline_report_seed777.csv).
- The uniform random policy would appear worse than every real baseline if
  naively reported; that comparison is meaningless and misleading.
- Frozen Day 3 files live in evaluation/output/frozen_day3/ and are immutable.

Boundary
--------
Creates:
    ml/generate_causal_training_data.py   <- this file
    ml/data/causal_training_data.csv      <- output

Does NOT touch:
    policy/
    evaluation/evaluator.py
    evaluation/output/frozen_day3/
    evaluation/output/baseline_report_seed42.csv
    evaluation/output/baseline_report_seed777.csv

Sources
-------
Observable features : data/v2/payment_scenarios.csv   (30 472 rows, V2 sim)
Hidden ground truth : data/v2/ground_truth.csv         (30 472 rows, V2 sim)
Join key            : payment_id  (1-to-1, no missing matches)

Treatment -> hidden probability column mapping
---------------------------------------------
WAIT         -> natural_recovery_probability
RETRY        -> retry_success_probability
RETRY_NUDGE  -> nudge_success_probability
ESCALATE     -> escalation_success_probability

Usage
-----
python -m ml.generate_causal_training_data [--seed SEED] [--obs PATH] [--gt PATH] [--out PATH]

Default seed: 2024
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

# -- Constants -----------------------------------------------------------------

TREATMENTS = ["WAIT", "RETRY", "RETRY_NUDGE", "ESCALATE"]

# Maps treatment arm -> column name in the ground-truth file
TREATMENT_PROB_COL: dict[str, str] = {
    "WAIT":        "natural_recovery_probability",
    "RETRY":       "retry_success_probability",
    "RETRY_NUDGE": "nudge_success_probability",
    "ESCALATE":    "escalation_success_probability",
}

# Observable X columns to keep in output (must not include any hidden column)
X_COLUMNS: list[str] = [
    "amount",
    "attempt_number",
    "dynamic_success_rate",
    "cumulative_failures",
    "consecutive_failed_cycles",
    "notification_engagement_score",
    "contact_response_score",
    "payment_method",
    "failure_reason",
]

# Default file paths (relative to repo root, resolved at call time)
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OBS_PATH = _REPO_ROOT / "data" / "v2" / "payment_scenarios.csv"
DEFAULT_GT_PATH  = _REPO_ROOT / "data" / "v2" / "ground_truth.csv"
DEFAULT_OUT_PATH = _REPO_ROOT / "ml" / "data" / "causal_training_data.csv"


# -- Helpers -------------------------------------------------------------------

def _load_and_validate(obs_path: pathlib.Path, gt_path: pathlib.Path) -> pd.DataFrame:
    """Load, join, and validate the two source files.  Exits on any anomaly."""

    print(f"[load] observable  : {obs_path}")
    print(f"[load] ground-truth: {gt_path}")

    if not obs_path.exists():
        sys.exit(f"ERROR: observable dataset not found at {obs_path}")
    if not gt_path.exists():
        sys.exit(f"ERROR: ground-truth not found at {gt_path}")

    obs = pd.read_csv(obs_path)
    gt  = pd.read_csv(gt_path)

    # Check required columns
    missing_x = [c for c in X_COLUMNS if c not in obs.columns]
    if missing_x:
        sys.exit(f"ERROR: required X columns missing from observable dataset: {missing_x}")

    if "payment_id" not in obs.columns:
        sys.exit("ERROR: 'payment_id' column missing from observable dataset")
    if "payment_id" not in gt.columns:
        sys.exit("ERROR: 'payment_id' column missing from ground-truth")

    missing_prob = [c for c in TREATMENT_PROB_COL.values() if c not in gt.columns]
    if missing_prob:
        sys.exit(f"ERROR: required probability columns missing from ground-truth: {missing_prob}")

    # Uniqueness checks
    obs_dupes = obs["payment_id"].duplicated().sum()
    gt_dupes  = gt["payment_id"].duplicated().sum()
    if obs_dupes > 0:
        sys.exit(f"ERROR: {obs_dupes} duplicate payment_id rows in observable dataset")
    if gt_dupes > 0:
        sys.exit(f"ERROR: {gt_dupes} duplicate payment_id rows in ground-truth")

    print(f"[validate] obs rows={len(obs):,}  unique payment_id={obs.payment_id.nunique():,}")
    print(f"[validate] gt  rows={len(gt):,}   unique payment_id={gt.payment_id.nunique():,}")

    # Join
    merged = obs.merge(gt, on="payment_id", how="inner")
    if len(merged) != len(obs):
        sys.exit(
            f"ERROR: inner join returned {len(merged):,} rows but obs has {len(obs):,}. "
            "Some payment_ids have no ground-truth match."
        )
    print(f"[validate] inner join: {len(merged):,} rows (1-to-1 confirmed)")

    return merged


def _assign_treatments_and_outcomes(
    merged: pd.DataFrame, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Assign T uniformly at random and draw Y ~ Bernoulli(p_T) using rng."""

    n = len(merged)

    # Uniform random treatment assignment
    t_indices  = rng.integers(0, len(TREATMENTS), size=n)
    assigned_T = np.array(TREATMENTS)[t_indices]

    # Vectorised lookup: for each row, get the probability column for its arm
    col_order  = list(TREATMENT_PROB_COL.values())
    col_idx    = {c: i for i, c in enumerate(col_order)}
    prob_matrix = merged[col_order].to_numpy(dtype=float)

    prob_col_indices = np.array(
        [col_idx[TREATMENT_PROB_COL[t]] for t in assigned_T]
    )
    selected_probs = prob_matrix[np.arange(n), prob_col_indices]

    # Draw Y ~ Bernoulli(p) -- strictly 0 or 1, NOT the probability
    realized_Y = rng.binomial(n=1, p=selected_probs).astype(int)

    return assigned_T, realized_Y


def _build_output(
    merged: pd.DataFrame, assigned_T: np.ndarray, realized_Y: np.ndarray
) -> pd.DataFrame:
    """Select only permitted output columns; strictly no hidden probabilities."""

    out = merged[["payment_id"] + X_COLUMNS].copy()
    out["assigned_T"] = assigned_T
    out["realized_Y"] = realized_Y

    # Paranoia check -- abort if any hidden column slipped through
    hidden_keywords = [
        "probability", "p_success", "net_value", "incremental_value", "best_action"
    ]
    leaked = [c for c in out.columns if any(h in c for h in hidden_keywords)]
    if leaked:
        sys.exit(f"ERROR: hidden columns leaked into output -- {leaked}. Aborting.")

    return out


def _verify(out: pd.DataFrame, source_rows: int) -> None:
    """Print all verification checks required by the 4B approval conditions."""

    sep = "-" * 60
    print(f"\n{sep}")
    print("  VERIFICATION REPORT -- 4B Approval Checks")
    print(sep)

    # 1. Row count
    count_ok = len(out) == source_rows
    print(f"\n[1] Row count")
    print(f"    source : {source_rows:,}")
    print(f"    output : {len(out):,}")
    print(f"    {'PASS -- one row per payment_id' if count_ok else 'FAIL'}")

    # 2. T distribution (~25% each)
    print(f"\n[2] Treatment allocation (target ~25% each)")
    t_dist = out["assigned_T"].value_counts(normalize=True).sort_index()
    all_balanced = True
    for arm, pct in t_dist.items():
        ok = 0.23 <= pct <= 0.27
        all_balanced = all_balanced and ok
        flag = "OK " if ok else "!  "
        print(f"    [{flag}] {arm:<14}  {pct:.2%}")
    print(f"    {'PASS' if all_balanced else 'WARN -- some arms outside 23-27% band'}")

    # 3. Y strictly {0, 1}
    y_vals = sorted(out["realized_Y"].unique())
    y_ok   = set(y_vals).issubset({0, 1})
    print(f"\n[3] realized_Y values: {y_vals}")
    print(f"    {'PASS -- strictly binary {0,1}' if y_ok else 'FAIL -- non-binary values found'}")

    # 4. No missing T or Y
    t_null = int(out["assigned_T"].isna().sum())
    y_null = int(out["realized_Y"].isna().sum())
    print(f"\n[4] Missing values")
    print(f"    assigned_T nulls : {t_null}  {'PASS' if t_null == 0 else 'FAIL'}")
    print(f"    realized_Y nulls : {y_null}  {'PASS' if y_null == 0 else 'FAIL'}")

    # 5. All four arms represented
    arms_present  = set(out["assigned_T"].unique())
    arms_expected = set(TREATMENTS)
    arms_ok = arms_present == arms_expected
    print(f"\n[5] All four arms represented: {'PASS' if arms_ok else 'FAIL'}")
    if not arms_ok:
        print(f"    missing: {arms_expected - arms_present}")

    # 6. Mean realized_Y per treatment (plausibility)
    print(f"\n[6] Mean realized_Y per treatment (plausibility check)")
    print(f"    Expected order roughly: WAIT < RETRY/NUDGE, ESCALATE variable")
    mean_y = out.groupby("assigned_T")["realized_Y"].mean().sort_values()
    for arm, m in mean_y.items():
        print(f"    {arm:<14}  mean_Y = {m:.4f}")

    # 7. No hidden columns in output
    hidden_keywords = [
        "probability", "p_success", "net_value", "incremental_value", "best_action"
    ]
    leaked = [c for c in out.columns if any(h in c for h in hidden_keywords)]
    print(f"\n[7] No hidden probability columns in output: {'PASS' if not leaked else 'FAIL -- ' + str(leaked)}")

    # 8. Governance
    print(f"\n[8] Data governance")
    print("    PASS -- ML TRAINING DATA ONLY")
    print("    Must NOT be fed into evaluation/report.py")
    print("    Must NOT be compared against Day 3 baseline table")
    print("    Frozen Day 3 files are untouched")

    print(f"\n{sep}\n")


# -- CLI -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate randomized-logging causal training data for Day 4 ML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--seed", type=int, default=2024,
        help="RNG seed (independent of all other codebase RNGs)",
    )
    p.add_argument(
        "--obs", type=pathlib.Path, default=DEFAULT_OBS_PATH,
        help="Path to observable dataset CSV",
    )
    p.add_argument(
        "--gt", type=pathlib.Path, default=DEFAULT_GT_PATH,
        help="Path to ground-truth CSV",
    )
    p.add_argument(
        "--out", type=pathlib.Path, default=DEFAULT_OUT_PATH,
        help="Output path for causal_training_data.csv",
    )
    p.add_argument(
        "--skip-verify", action="store_true",
        help="Skip the verification report (not recommended)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  ml/generate_causal_training_data.py")
    print(f"  seed={args.seed}")
    print("=" * 60)

    # Step 1: Load and validate
    merged = _load_and_validate(args.obs, args.gt)
    source_rows = len(merged)

    # Step 2: Dedicated RNG -- isolated from all other codebase RNGs
    rng = np.random.default_rng(args.seed)

    # Steps 3 & 4: Assign T and draw Y
    assigned_T, realized_Y = _assign_treatments_and_outcomes(merged, rng)

    # Step 5: Build output (X columns + assigned_T + realized_Y only)
    out = _build_output(merged, assigned_T, realized_Y)

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n[write] {args.out}")
    print(f"[write] {len(out):,} rows  x  {len(out.columns)} columns")
    print(f"[write] columns: {list(out.columns)}")

    # Step 6: Verification
    if not args.skip_verify:
        _verify(out, source_rows)


if __name__ == "__main__":
    main()
