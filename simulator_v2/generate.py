"""
Recovery Intelligence Engine v2 — main generator.

Pipeline:
    Generate customers
        ↓
    Generate subscriptions (1:1)
        ↓
    Step through billing cycles — stateful, chronological
        ↓
    Attach hidden ground truth (counterfactuals + net values)
        ↓
    Save datasets:
            data/v2/payment_scenarios.csv   → observable features only
            data/v2/ground_truth.csv        → hidden truth (NEVER feed to model)
            data/v2/cv_groups.csv           → customer_id → fold mapping for GroupKFold

Usage:
    python simulator_v2/generate.py
    python simulator_v2/generate.py --customers 3000 --cycles 12 --seed 42

Differences from v1
--------------------
- Simulation drives through n_customers × n_billing_cycles instead of
  directly targeting a fixed number of failed payments.
- Observable records reflect stateful cumulative_failures / dynamic_success_rate.
- Ground truth includes net recovery values and incremental values vs WAIT.
- cv_groups.csv is emitted for downstream GroupKFold cross-validation.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# Ensure the project root is on the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulator_v2.config import (
    DEFAULT_NUM_CUSTOMERS,
    DEFAULT_NUM_BILLING_CYCLES,
    DEFAULT_SEED,
)
from simulator_v2.customer_generator import generate_customers
from simulator_v2.subscription_generator import generate_subscriptions
from simulator_v2.payment_generator import generate_payments
from simulator_v2.ground_truth import attach_ground_truth


def _make_cv_groups(payments_df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """
    Assign each customer_id to a cross-validation fold.

    The mapping must be applied at training time using GroupKFold so that
    a single customer's payments never appear in both train and validation
    sets simultaneously.

    Returns a DataFrame with columns: customer_id, cv_fold (0 to n_folds-1).
    """
    unique_customers = payments_df["customer_id"].unique()
    folds = np.arange(len(unique_customers)) % n_folds
    return pd.DataFrame({
        "customer_id": unique_customers,
        "cv_fold": folds,
    })


def run(
    n_customers: int,
    n_billing_cycles: int,
    seed: int,
    out_dir: str,
) -> None:
    """Run the full v2 generation pipeline."""
    rng = np.random.default_rng(seed)
    start = time.time()

    print(f"[v2] Generating {n_customers:,} customers × {n_billing_cycles} billing cycles …")

    # --- 1. Customers ---
    customers_df = generate_customers(n_customers, rng)
    print(f"  Customers:          {len(customers_df):,}")

    # --- 2. Subscriptions (1:1 with customers) ---
    subscriptions_df = generate_subscriptions(
        customers_df["customer_id"].tolist(), rng
    )
    print(f"  Subscriptions:      {len(subscriptions_df):,}")

    # --- 3. Sequential payment attempts ---
    payments_df = generate_payments(customers_df, subscriptions_df, n_billing_cycles, rng)
    print(f"  Payment attempts:   {len(payments_df):,}")

    # --- 4. Hidden ground truth ---
    ground_truth_df = attach_ground_truth(payments_df, customers_df, rng)

    # --- 5. CV group assignments ---
    cv_groups_df = _make_cv_groups(payments_df, n_folds=5)

    # --- 6. Strip hidden columns before saving observable dataset ---
    hidden_cols = [c for c in payments_df.columns if c.startswith("_hidden_")]
    obs_df = payments_df.drop(columns=hidden_cols)

    # --- 7. Save ---
    os.makedirs(out_dir, exist_ok=True)

    obs_path = os.path.join(out_dir, "payment_scenarios.csv")
    gt_path  = os.path.join(out_dir, "ground_truth.csv")
    cv_path  = os.path.join(out_dir, "cv_groups.csv")

    obs_df.to_csv(obs_path, index=False)
    ground_truth_df.to_csv(gt_path, index=False)
    cv_groups_df.to_csv(cv_path, index=False)

    elapsed = time.time() - start

    # --- 8. Summary ---
    print(f"\n[v2] Dataset summary:")
    print(f"  Total payment attempts:  {len(obs_df):,}")
    print(f"  Unique customers:        {obs_df['customer_id'].nunique():,}")
    print(f"  Unique billing cycles:   {obs_df['billing_cycle_id'].nunique():,}")
    print(f"  Attempt distribution:\n{obs_df['attempt_number'].value_counts().sort_index().to_string()}")

    print(f"\n  Ground truth best_action distribution:")
    print(ground_truth_df["best_action"].value_counts().to_string())

    print(f"\nSaved:")
    print(f"  {obs_path}")
    print(f"  {gt_path}")
    print(f"  {cv_path}")
    print(f"\n({elapsed:.1f}s, seed={seed})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate v2 synthetic payment scenarios (stateful, sequential)."
    )
    parser.add_argument(
        "--customers", type=int, default=DEFAULT_NUM_CUSTOMERS,
        help=f"Number of customers (default: {DEFAULT_NUM_CUSTOMERS:,})",
    )
    parser.add_argument(
        "--cycles", type=int, default=DEFAULT_NUM_BILLING_CYCLES,
        help=f"Billing cycles per customer (default: {DEFAULT_NUM_BILLING_CYCLES})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--out", type=str,
        default=os.path.join(PROJECT_ROOT, "data", "v2"),
        help="Output directory (default: data/v2/)",
    )
    args = parser.parse_args()
    run(args.customers, args.cycles, args.seed, args.out)


if __name__ == "__main__":
    main()
