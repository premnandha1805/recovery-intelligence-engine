"""
Recovery Intelligence Engine — main generator (Day 1).

Pipeline:
    Generate customers
        ↓
    Generate subscriptions
        ↓
    Generate failed payments
        ↓
    Assign hidden ground truth
        ↓
    Save dataset

Usage:
    python simulator/generate.py
    python simulator/generate.py --n 10000 --seed 42

Output:
    data/raw/payment_scenarios.csv   → observable features (model input)
    data/raw/ground_truth.csv        → hidden truth (evaluation only, NEVER feed to model)
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# Ensure the project root is on the path so `simulator.*` imports work
# when running as `python simulator/generate.py`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulator.config import DEFAULT_NUM_PAYMENTS, DEFAULT_CUSTOMER_RATIO, DEFAULT_SEED
from simulator.customer_generator import generate_customers
from simulator.subscription_generator import generate_subscriptions
from simulator.payment_generator import generate_payments
from simulator.ground_truth import attach_ground_truth


def run(n_payments: int, seed: int, out_dir: str) -> None:
    """Run the full Day-1 generation pipeline."""
    rng = np.random.default_rng(seed)
    start = time.time()

    # --- 1. Customers ---
    n_customers = max(int(n_payments * DEFAULT_CUSTOMER_RATIO), 100)
    customers_df = generate_customers(n_customers, rng)
    print(f"  Customers: {len(customers_df):,}")

    # --- 2. Subscriptions (1:1 with customers for Day 1) ---
    subscriptions_df = generate_subscriptions(
        customers_df["customer_id"].tolist(), rng
    )
    print(f"  Subscriptions: {len(subscriptions_df):,}")

    # --- 3. Failed payments ---
    payments_df = generate_payments(customers_df, subscriptions_df, n_payments, rng)
    print(f"  Failed payments: {len(payments_df):,}")

    # --- 4. Hidden ground truth ---
    ground_truth_df = attach_ground_truth(payments_df, customers_df, rng)

    # --- 5. Save ---
    os.makedirs(out_dir, exist_ok=True)

    obs_path = os.path.join(out_dir, "payment_scenarios.csv")
    gt_path = os.path.join(out_dir, "ground_truth.csv")

    payments_df.to_csv(obs_path, index=False)
    ground_truth_df.to_csv(gt_path, index=False)

    elapsed = time.time() - start

    print(f"\nGenerated {n_payments:,} payment scenarios\n")
    print(f"Customers: {n_customers:,}")
    print(f"Subscriptions: {len(subscriptions_df):,}")
    print(f"Failed payments: {n_payments:,}")
    print(f"\nSaved:")
    print(f"  {obs_path}")
    print(f"  {gt_path}")
    print(f"\n({elapsed:.1f}s, seed={seed})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic payment scenarios (Day 1)."
    )
    parser.add_argument(
        "--n", type=int, default=DEFAULT_NUM_PAYMENTS,
        help=f"Number of failed payments to generate (default: {DEFAULT_NUM_PAYMENTS:,})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--out", type=str,
        default=os.path.join(PROJECT_ROOT, "data", "raw"),
        help="Output directory (default: data/raw/)",
    )
    args = parser.parse_args()
    run(args.n, args.seed, args.out)


if __name__ == "__main__":
    main()
