"""
Ground truth generator — attaches hidden counterfactual probabilities
to each failed payment.

For every failed payment the simulator knows:
    natural_recovery_probability
    retry_success_probability
    nudge_success_probability
    escalation_success_probability

The AI/model will NEVER see these columns. They exist so the evaluation
framework can compute true uplift later.

Example:
    payment_001
        natural   = 0.72
        retry     = 0.81
        nudge     = 0.76
        escalate  = 0.79
        → retry uplift = 0.81 − 0.72 = 0.09
"""

import numpy as np
import pandas as pd


def attach_ground_truth(
    payments_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Build a ground-truth table with one row per payment.

    Hidden probabilities come from the customer's hidden traits (which
    themselves derive from the customer archetype). Per-payment noise is
    added on top, plus modifiers from failure_reason and attempt_number
    so the truth isn't *only* a function of customer identity.

    Parameters
    ----------
    payments_df : DataFrame
        Must include payment_id, customer_id, failure_reason, attempt_number.
    customers_df : DataFrame
        Must include customer_id and _hidden_* columns.
    rng : np.random.Generator
        Seeded random generator.

    Returns
    -------
    DataFrame with columns:
        payment_id, natural_recovery_probability, retry_success_probability,
        nudge_success_probability, escalation_success_probability
    """
    # Failure-reason modifiers (some reasons are structurally harder to recover)
    REASON_MODIFIER = {
        "insufficient_funds":      -0.10,
        "bank_decline":            -0.05,
        "network_error":            0.05,
        "expired_card":            -0.15,
        "authentication_failure":  -0.08,
        "temporary_bank_issue":     0.08,
    }

    # Attempt-number modifier (later attempts are harder)
    ATTEMPT_MODIFIER = {1: 0.0, 2: -0.05, 3: -0.10, 4: -0.18}

    # Build a lookup of customer hidden traits
    hidden_cols = [
        "customer_id",
        "_hidden_intrinsic_recovery_prob",
        "_hidden_retry_responsiveness",
        "_hidden_nudge_responsiveness",
    ]
    customer_hidden = customers_df[hidden_cols].set_index("customer_id")

    rows = []
    for _, pay in payments_df.iterrows():
        cust = customer_hidden.loc[pay["customer_id"]]

        base_natural = cust["_hidden_intrinsic_recovery_prob"]
        retry_resp = cust["_hidden_retry_responsiveness"]
        nudge_resp = cust["_hidden_nudge_responsiveness"]

        # Context modifiers
        reason_mod = REASON_MODIFIER.get(pay["failure_reason"], 0.0)
        attempt_mod = ATTEMPT_MODIFIER.get(pay["attempt_number"], -0.20)

        # Per-payment noise
        noise = rng.normal(0, 0.03, size=4)

        natural = base_natural + reason_mod + attempt_mod + noise[0]
        retry = natural + retry_resp + noise[1]
        nudge = natural + nudge_resp + noise[2]
        escalate = max(retry, nudge) + abs(rng.normal(0, 0.03)) + noise[3]

        # Clip to valid probability range
        natural = float(np.clip(natural, 0.01, 0.99))
        retry = float(np.clip(retry, 0.01, 0.99))
        nudge = float(np.clip(nudge, 0.01, 0.99))
        escalate = float(np.clip(escalate, 0.01, 0.99))

        rows.append({
            "payment_id": pay["payment_id"],
            "natural_recovery_probability": round(natural, 4),
            "retry_success_probability": round(retry, 4),
            "nudge_success_probability": round(nudge, 4),
            "escalation_success_probability": round(escalate, 4),
        })

    return pd.DataFrame(rows)
