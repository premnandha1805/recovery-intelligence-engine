"""
Subscription generator — creates one subscription per customer.

Each subscription has:
    subscription_id, customer_id, plan_type, amount,
    billing_frequency, subscription_age_days
"""

import numpy as np
import pandas as pd

from simulator.config import (
    PLAN_AMOUNTS,
    PLAN_AMOUNT_WEIGHTS,
    PLAN_TYPES,
    PLAN_TYPE_WEIGHTS,
    BILLING_FREQUENCIES,
    BILLING_FREQUENCY_WEIGHTS,
)


def generate_subscriptions(
    customer_ids: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate one subscription per customer.

    Parameters
    ----------
    customer_ids : list[str]
        List of customer IDs to attach subscriptions to.
    rng : np.random.Generator
        Seeded random generator for reproducibility.
    """
    # Normalise weights
    amount_p = np.array(PLAN_AMOUNT_WEIGHTS, dtype=float)
    amount_p = amount_p / amount_p.sum()

    type_p = np.array(PLAN_TYPE_WEIGHTS, dtype=float)
    type_p = type_p / type_p.sum()

    freq_p = np.array(BILLING_FREQUENCY_WEIGHTS, dtype=float)
    freq_p = freq_p / freq_p.sum()

    rows = []
    for i, cust_id in enumerate(customer_ids):
        subscription_id = f"sub_{i + 1:06d}"

        amount = PLAN_AMOUNTS[rng.choice(len(PLAN_AMOUNTS), p=amount_p)]
        plan_type = PLAN_TYPES[rng.choice(len(PLAN_TYPES), p=type_p)]
        billing_frequency = BILLING_FREQUENCIES[
            rng.choice(len(BILLING_FREQUENCIES), p=freq_p)
        ]
        subscription_age_days = int(rng.integers(15, 900))

        rows.append({
            "subscription_id": subscription_id,
            "customer_id": cust_id,
            "plan_type": plan_type,
            "amount": amount,
            "billing_frequency": billing_frequency,
            "subscription_age_days": subscription_age_days,
        })

    return pd.DataFrame(rows)
