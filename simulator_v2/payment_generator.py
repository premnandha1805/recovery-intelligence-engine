"""
Payment generator (v2) — stateful, sequential, chronological.

Key changes from v1
-------------------
1.  Chronological billing cycles:
        For each customer × subscription pair, the simulator steps through
        `n_billing_cycles` consecutive billing periods. The customer's state
        (cumulative_failures, dynamic_success_rate) is updated AFTER each
        cycle, so later payments reflect the customer's running history.

2.  Sequential attempt progression within a failed cycle:
        When a payment fails on attempt 1, the simulator may generate attempt
        2, 3, … up to MAX_ATTEMPTS_PER_CYCLE. Each attempt row is a separate
        observable record. Attempt N only exists if attempt N-1 failed.

3.  Observable features reflect state AT DECISION TIME:
        cumulative_failures and dynamic_success_rate in each row show the
        customer's state *before* that particular attempt, not a static
        snapshot taken at customer creation.

4.  Hidden columns are NOT included here:
        ground_truth.py reads _hidden_* from customers_df internally and
        outputs a separate table. The DataFrame returned here contains only
        fields a decision engine would realistically observe.

Output columns (observable only)
---------------------------------
    payment_id              Unique attempt identifier  pay_XXXXXX_aY
    billing_cycle_id        Which billing period       cyc_XXXXXX_CCC
    customer_id
    subscription_id
    amount
    payment_method
    failure_reason
    attempt_number          1–4, sequential within cycle
    cumulative_failures     # prior failed cycles for this customer (updated)
    dynamic_success_rate    Running success rate at decision time
    days_active             Customer age in simulated days at this attempt
"""

import numpy as np
import pandas as pd

from simulator_v2.config import (
    FailureReason,
    FAILURE_REASON_WEIGHTS,
    MAX_ATTEMPTS_PER_CYCLE,
)


# Probability that a billing cycle fails at all, given the customer's
# current dynamic_success_rate. We model the cycle-level failure probability
# directly from the inverse of the running success rate.
_MIN_CYCLE_FAILURE_PROB = 0.05   # even reliable customers can have a bad cycle


def _cycle_fails(dynamic_success_rate: float, rng: np.random.Generator) -> bool:
    """Return True if this billing cycle results in a failed payment."""
    p_failure = max(1.0 - dynamic_success_rate, _MIN_CYCLE_FAILURE_PROB)
    return bool(rng.random() < p_failure)


def generate_payments(
    customers_df: pd.DataFrame,
    subscriptions_df: pd.DataFrame,
    n_billing_cycles: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate payment attempt records by stepping through billing cycles.

    Parameters
    ----------
    customers_df : DataFrame
        Must include customer_id, historical_success_rate, previous_failure_count,
        and _hidden_* columns. _hidden_* are used ONLY to pass to ground_truth.py;
        they are NOT written to the output DataFrame.
    subscriptions_df : DataFrame
        Must include subscription_id, customer_id, amount.
    n_billing_cycles : int
        Number of monthly billing cycles to simulate per customer.
    rng : np.random.Generator
        Seeded random generator — same instance propagated throughout.

    Returns
    -------
    pd.DataFrame — observable payment attempt records only.
    """
    # Merge customers ↔ subscriptions (1:1 in v2)
    merged = subscriptions_df.merge(customers_df, on="customer_id")

    # Failure-reason distribution arrays (built once)
    reasons = list(FAILURE_REASON_WEIGHTS.keys())
    reason_p = np.array([FAILURE_REASON_WEIGHTS[r] for r in reasons])
    reason_p = reason_p / reason_p.sum()

    rows = []
    payment_counter = 0

    for _, record in merged.iterrows():
        cust_id = record["customer_id"]
        sub_id = record["subscription_id"]
        amount = record["amount"]
        payment_method = record["payment_method"]

        # --- Mutable per-customer state ---
        # These are updated after each cycle and used as observable features.
        cumulative_failures = int(record["previous_failure_count"])
        cumulative_successes = max(
            int(round(record["historical_success_rate"] * 10)), 1
        )
        days_active = int(record["subscription_age_days"])

        for cycle_idx in range(n_billing_cycles):
            cycle_id = f"cyc_{cust_id[-6:]}_{cycle_idx + 1:03d}"
            days_active += 30   # advance by one billing month

            # Dynamic success rate at the start of this cycle (observable)
            total_cycles_so_far = cumulative_failures + cumulative_successes
            dynamic_success_rate = round(
                cumulative_successes / max(total_cycles_so_far, 1), 3
            )

            if not _cycle_fails(dynamic_success_rate, rng):
                # Successful cycle — update state and move on
                cumulative_successes += 1
                continue

            # --- Billing cycle fails — generate sequential attempts ---
            failure_reason = reasons[rng.choice(len(reasons), p=reason_p)]

            for attempt in range(1, MAX_ATTEMPTS_PER_CYCLE + 1):
                payment_counter += 1
                payment_id = f"pay_{payment_counter:06d}_a{attempt}"

                rows.append({
                    "payment_id":           payment_id,
                    "billing_cycle_id":     cycle_id,
                    "customer_id":          cust_id,
                    "subscription_id":      sub_id,
                    "amount":               amount,
                    "payment_method":       payment_method,
                    "failure_reason":       failure_reason.value,
                    "attempt_number":       attempt,
                    # State AT decision time — reflects all prior cycles
                    "cumulative_failures":  cumulative_failures,
                    "dynamic_success_rate": dynamic_success_rate,
                    "days_active":          days_active,
                })

                # Stop generating further attempts with increasing probability
                # (models whether the cycle eventually self-recovers or stays failed)
                # Actual outcome is determined in ground_truth.py — here we only
                # model whether the system would *attempt* another retry.
                # Simple heuristic: ~40% chance the cycle is abandoned after each attempt.
                if attempt < MAX_ATTEMPTS_PER_CYCLE and rng.random() < 0.40:
                    break

            # Cycle failed — update state
            cumulative_failures += 1

    return pd.DataFrame(rows)
