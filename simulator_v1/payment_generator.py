"""
Payment generator — creates failed payment records.

Each failed payment carries only OBSERVABLE features:
    payment_id, customer_id, subscription_id, amount, payment_method,
    failure_reason, attempt_number, days_since_last_success,
    historical_success_rate, previous_failure_count

Hidden ground truth is attached separately by ground_truth.py.
"""

import numpy as np
import pandas as pd

from simulator.config import (
    FailureReason,
    FAILURE_REASON_WEIGHTS,
)


def generate_payments(
    customers_df: pd.DataFrame,
    subscriptions_df: pd.DataFrame,
    n_payments: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate *n_payments* failed-payment records.

    Customers may appear more than once (multiple billing cycles can fail).
    The customer's observable traits are merged in so each payment row is
    self-contained.

    Parameters
    ----------
    customers_df : DataFrame
        Must include customer_id and observable columns.
    subscriptions_df : DataFrame
        Must include subscription_id, customer_id, amount.
    n_payments : int
        Total number of failed payments to generate.
    rng : np.random.Generator
        Seeded random generator.
    """
    # Merge customers ↔ subscriptions
    merged = subscriptions_df.merge(customers_df, on="customer_id")

    # Weight customer selection by failure propensity — customers with
    # lower historical_success_rate are more likely to generate failures
    failure_propensity = 1.0 - merged["historical_success_rate"].values + 0.05
    failure_propensity = failure_propensity / failure_propensity.sum()

    # Failure-reason distribution
    reasons = list(FAILURE_REASON_WEIGHTS.keys())
    reason_p = np.array([FAILURE_REASON_WEIGHTS[r] for r in reasons])
    reason_p = reason_p / reason_p.sum()

    rows = []
    for i in range(n_payments):
        # Pick a customer/subscription pair (weighted toward failure-prone)
        idx = rng.choice(len(merged), p=failure_propensity)
        record = merged.iloc[idx]

        payment_id = f"pay_{i + 1:06d}"
        failure_reason = reasons[rng.choice(len(reasons), p=reason_p)]

        # Attempt number: most failures are attempt 1, some are retries
        attempt_number = int(rng.choice([1, 2, 3, 4], p=[0.55, 0.25, 0.13, 0.07]))

        # Days since last successful payment (correlated with history)
        base_days = rng.exponential(scale=15)
        days_since_last_success = int(
            np.clip(base_days * (1 + record["previous_failure_count"] * 0.3), 1, 365)
        )

        rows.append({
            "payment_id": payment_id,
            "customer_id": record["customer_id"],
            "subscription_id": record["subscription_id"],
            "amount": record["amount"],
            "payment_method": record["payment_method"],
            "failure_reason": failure_reason.value,
            "attempt_number": attempt_number,
            "days_since_last_success": days_since_last_success,
            "historical_success_rate": record["historical_success_rate"],
            "previous_failure_count": record["previous_failure_count"],
        })

    return pd.DataFrame(rows)
