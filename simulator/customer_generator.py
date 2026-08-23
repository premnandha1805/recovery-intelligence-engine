"""
Customer generator — creates synthetic customers with observable AND hidden traits.

Observable traits (what the model sees):
    customer_id, customer_tier, historical_success_rate,
    previous_failure_count, average_payment_delay, payment_method

Hidden traits (simulator-only ground truth):
    intrinsic_recovery_probability, retry_responsiveness, nudge_responsiveness

The hidden values are generated from the customer's archetype (CustomerType)
but are NEVER exposed to the model.
"""

import numpy as np
import pandas as pd

from simulator.config import (
    CustomerType,
    PaymentMethod,
    CUSTOMER_TYPE_WEIGHTS,
    TYPE_BASE_PROBS,
    PAYMENT_METHOD_WEIGHTS,
)


def _sample_hidden_traits(customer_type: CustomerType, rng: np.random.Generator) -> dict:
    """
    Generate hidden (simulator-only) traits from the customer archetype.

    Returns per-customer probabilities with noise so no two customers of
    the same type are identical.
    """
    base = TYPE_BASE_PROBS[customer_type]
    noise = rng.normal(0, 0.05, size=4)
    probs = np.clip(np.array(base) + noise, 0.01, 0.99)

    return {
        "customer_type": customer_type.value,
        "intrinsic_recovery_probability": round(float(probs[0]), 4),
        "retry_responsiveness": round(float(probs[1] - probs[0]), 4),
        "nudge_responsiveness": round(float(probs[2] - probs[0]), 4),
    }


def _sample_observable_traits(
    customer_type: CustomerType,
    hidden: dict,
    rng: np.random.Generator,
) -> dict:
    """
    Generate observable features that CORRELATE with hidden state but are
    noisy enough that a trivial regression cannot reconstruct them.
    """
    p_natural = hidden["intrinsic_recovery_probability"]

    # Historical success rate — simulate a finite history window
    history_len = rng.integers(5, 50)
    historical_success_rate = np.clip(
        rng.binomial(history_len, p_natural) / history_len, 0.0, 1.0
    )

    # Previous failure count (Poisson, higher for weaker customers)
    previous_failure_count = int(rng.poisson(lam=(1 - p_natural) * 6))

    # Average payment delay in hours (exponential, longer for weaker customers)
    average_payment_delay = round(
        float(rng.exponential(scale=(1 - p_natural) * 48 + 2)), 1
    )

    # Payment method — weighted random
    methods = list(PAYMENT_METHOD_WEIGHTS.keys())
    method_p = np.array([PAYMENT_METHOD_WEIGHTS[m] for m in methods])
    method_p = method_p / method_p.sum()
    payment_method = methods[rng.choice(len(methods), p=method_p)]

    # Customer tier — correlated with customer type
    if customer_type in (CustomerType.RELIABLE, CustomerType.RETRY_RESPONSIVE):
        tier = rng.choice(
            ["basic", "standard", "premium", "enterprise"],
            p=[0.10, 0.30, 0.35, 0.25],
        )
    elif customer_type == CustomerType.CHRONIC_FAILURE:
        tier = rng.choice(
            ["basic", "standard", "premium", "enterprise"],
            p=[0.45, 0.30, 0.15, 0.10],
        )
    else:
        tier = rng.choice(
            ["basic", "standard", "premium", "enterprise"],
            p=[0.25, 0.35, 0.25, 0.15],
        )

    return {
        "customer_tier": tier,
        "historical_success_rate": round(float(historical_success_rate), 3),
        "previous_failure_count": previous_failure_count,
        "average_payment_delay": average_payment_delay,
        "payment_method": payment_method.value,
    }


def generate_customers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate *n* customers with observable + hidden columns.

    Returns a single DataFrame. Hidden columns are prefixed with `_hidden_`
    so they can easily be stripped before model training.
    """
    types = list(CUSTOMER_TYPE_WEIGHTS.keys())
    type_p = np.array([CUSTOMER_TYPE_WEIGHTS[t] for t in types])
    type_p = type_p / type_p.sum()

    rows = []
    for i in range(n):
        customer_id = f"cust_{i + 1:06d}"
        customer_type = types[rng.choice(len(types), p=type_p)]

        hidden = _sample_hidden_traits(customer_type, rng)
        observable = _sample_observable_traits(customer_type, hidden, rng)

        rows.append({
            "customer_id": customer_id,
            **observable,
            # Hidden columns (simulator-only)
            "_hidden_customer_type": hidden["customer_type"],
            "_hidden_intrinsic_recovery_prob": hidden["intrinsic_recovery_probability"],
            "_hidden_retry_responsiveness": hidden["retry_responsiveness"],
            "_hidden_nudge_responsiveness": hidden["nudge_responsiveness"],
        })

    return pd.DataFrame(rows)
