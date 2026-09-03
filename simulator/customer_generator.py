"""
Customer generator (v2).

Customer archetypes, hidden traits, and base observable features use the
same logic and distributions as v1.

New in this version:
    Two observable proxy columns are appended after all customers are
    generated, using noise RNGs derived from the main seed so that
    single-argument reproducibility holds across the full pipeline:

        notification_engagement_score   — noisy proxy for nudge responsiveness
        contact_response_score          — noisy proxy for retry responsiveness

    Both use coeff=1.0, noise_std=0.22, and are clipped to [0, 1].
    Noise seeds are  (seed + 1001)  and  (seed + 2002)  respectively.

v2 sequencing changes live in payment_generator.py and ground_truth.py.
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
        # escalate: responsiveness computed relative to max(retry, nudge)
        # Note: escalate_responsiveness can be negative (penalised customers)
        "escalate_responsiveness": round(float(probs[3] - max(probs[1], probs[2])), 4),
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


def generate_customers(n: int, rng: np.random.Generator, seed: int = 0) -> pd.DataFrame:
    """
    Generate *n* customers with observable + hidden columns.

    Parameters
    ----------
    n : int
        Number of customers to generate.
    rng : np.random.Generator
        Primary seeded generator — shared across the full pipeline.
        All archetype, trait, and observable draws consume from this.
    seed : int
        The same seed passed to ``np.random.default_rng`` for the primary
        ``rng``.  Used ONLY to derive two secondary generators for the
        proxy-column noise so they are reproducible but independent of the
        primary draw sequence::

            notification_engagement_score noise  ->  default_rng(seed + 1001)
            contact_response_score noise         ->  default_rng(seed + 2002)

    Returns
    -------
    pd.DataFrame
        Hidden columns are prefixed with ``_hidden_`` and stripped before
        writing ``payment_scenarios.csv``.
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
            # Hidden columns (simulator-only — never serialized to payment_scenarios)
            "_hidden_customer_type": hidden["customer_type"],
            "_hidden_intrinsic_recovery_prob": hidden["intrinsic_recovery_probability"],
            "_hidden_retry_responsiveness": hidden["retry_responsiveness"],
            "_hidden_nudge_responsiveness": hidden["nudge_responsiveness"],
            "_hidden_escalate_responsiveness": hidden["escalate_responsiveness"],
        })

    df = pd.DataFrame(rows)

    # ── Observable proxy columns for treatment responsiveness ─────────────────
    # Generated AFTER the full population exists so population means are exact.
    # Noise RNGs are derived from `seed` (not from `rng`) to keep them
    # independent of the primary draw sequence while remaining reproducible.

    nudge_resp  = df["_hidden_nudge_responsiveness"].values
    retry_resp  = df["_hidden_retry_responsiveness"].values
    nudge_pop_mean = float(nudge_resp.mean())
    retry_pop_mean = float(retry_resp.mean())

    # notification_engagement_score  — noisy proxy for nudge responsiveness
    rng_nudge = np.random.default_rng(seed + 1001)
    noise_nudge = rng_nudge.normal(0.0, 0.22, size=n)
    df["notification_engagement_score"] = np.clip(
        0.5 + 1.0 * (nudge_resp - nudge_pop_mean) + noise_nudge,
        0.0, 1.0,
    ).round(4)

    # contact_response_score  — noisy proxy for retry responsiveness
    rng_retry = np.random.default_rng(seed + 2002)
    noise_retry = rng_retry.normal(0.0, 0.22, size=n)
    df["contact_response_score"] = np.clip(
        0.5 + 1.0 * (retry_resp - retry_pop_mean) + noise_retry,
        0.0, 1.0,
    ).round(4)

    return df
