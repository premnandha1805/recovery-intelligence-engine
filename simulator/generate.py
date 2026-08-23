"""
Recovery Intelligence Engine — synthetic environment generator.

Day 1 target: generate 10,000+ payment scenarios with hidden ground truth
and observable features, using RANDOMIZED action assignment (Mode A).

Mode A exists specifically to avoid confounded uplift estimation: if actions
were assigned by history-dependent rules at generation time, a model trained
on this data would learn "who historical rules chose to intervene on," not
"who actually benefits from intervention." Mode B (policy-driven generation,
for the final baseline comparison) is a separate script — do not build it
until the estimator trained on Mode A data is validated.

Usage:
    python simulator/generate.py --n 10000 --seed 42

Outputs (in simulator/output/):
    observable_dataset.csv   -> what the estimator/LLM/backend are allowed to see
    hidden_ground_truth.csv  -> simulator-only, joinable by payment_id, for
                                 evaluation code ONLY. Never feed this to a model.
"""

import argparse
import os
import sys
import uuid
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.schemas import CustomerType, Action, FailureReason, PaymentMethod  # noqa: E402


# ---------------------------------------------------------------------------
# Hidden type definitions: base P(success) under each action, per type.
# These numbers are the simulator's private truth. Noise is added per-customer
# so the estimator can't reverse-engineer them exactly from features.
# ---------------------------------------------------------------------------

TYPE_BASE_PROBS = {
    # type:                (no_action, retry, nudge,  escalate)
    CustomerType.TRANSIENT_FAILURE:  (0.90, 0.92, 0.91, 0.92),
    CustomerType.DETERIORATING:      (0.45, 0.70, 0.55, 0.72),
    CustomerType.CHRONIC_FAILURE:    (0.05, 0.07, 0.08, 0.10),
    CustomerType.RESPONSIVE:         (0.35, 0.55, 0.75, 0.78),
    CustomerType.NON_RESPONSIVE:     (0.40, 0.42, 0.41, 0.43),
    CustomerType.HIGH_VALUE:         (0.60, 0.75, 0.72, 0.85),
    CustomerType.LOW_VALUE:          (0.50, 0.58, 0.56, 0.60),
    CustomerType.TEMPORARY_FAILURE:  (0.80, 0.93, 0.90, 0.93),
}

TYPE_WEIGHTS = {
    CustomerType.TRANSIENT_FAILURE:  0.20,
    CustomerType.DETERIORATING:      0.15,
    CustomerType.CHRONIC_FAILURE:    0.10,
    CustomerType.RESPONSIVE:         0.12,
    CustomerType.NON_RESPONSIVE:     0.10,
    CustomerType.HIGH_VALUE:         0.13,
    CustomerType.LOW_VALUE:          0.10,
    CustomerType.TEMPORARY_FAILURE:  0.10,
}

FAILURE_REASON_WEIGHTS = {
    FailureReason.INSUFFICIENT_FUNDS: 0.35,
    FailureReason.CARD_EXPIRED:       0.15,
    FailureReason.BANK_DECLINE:       0.20,
    FailureReason.NETWORK_ERROR:      0.15,
    FailureReason.ISSUER_TIMEOUT:     0.10,
    FailureReason.FRAUD_FLAG:         0.05,
}

PAYMENT_METHOD_WEIGHTS = {
    PaymentMethod.CARD: 0.45,
    PaymentMethod.UPI: 0.35,
    PaymentMethod.NETBANKING: 0.15,
    PaymentMethod.WALLET: 0.05,
}

ACTIONS_FOR_RANDOM_ASSIGNMENT = [Action.WAIT, Action.RETRY, Action.RETRY_NUDGE, Action.ESCALATE]

ACTION_TO_PROB_INDEX = {
    Action.WAIT: 0,
    Action.RETRY: 1,
    Action.RETRY_NUDGE: 2,
    Action.ESCALATE: 3,
}


def sample_hidden_probs(customer_type: CustomerType, rng: np.random.Generator):
    """Per-customer hidden probabilities = type base + noise, clipped to [0,1]."""
    base = TYPE_BASE_PROBS[customer_type]
    noise = rng.normal(0, 0.04, size=4)
    probs = np.clip(np.array(base) + noise, 0.01, 0.99)
    # enforce monotonic-ish sanity: escalate should not be much worse than nudge
    return probs  # [no_action, retry, nudge, escalate]


def sample_observable_features(customer_type: CustomerType, hidden_probs, rng: np.random.Generator):
    """
    Observable features CORRELATE with hidden state but are NOT deterministic
    functions of it — real noise is added so a trivial regression can't
    reconstruct the hidden probabilities exactly (see docs/leakage_check.md,
    to be added on Day 2).
    """
    p_no_action = hidden_probs[0]

    # historical_success_rate correlates with p_no_action but is noisy and
    # based on a finite history window (more realistic than a direct copy)
    history_len = rng.integers(3, 40)
    historical_success_rate = np.clip(
        rng.binomial(history_len, p_no_action) / history_len, 0.0, 1.0
    )

    previous_failure_count = rng.poisson(lam=(1 - p_no_action) * 5)
    previous_retry_success_rate = np.clip(
        hidden_probs[1] - hidden_probs[0] + rng.normal(0, 0.15), 0.0, 1.0
    )
    previous_nudge_response = np.clip(
        hidden_probs[2] - hidden_probs[0] + rng.normal(0, 0.15), 0.0, 1.0
    )
    recent_consecutive_failures = rng.integers(1, 5) if p_no_action < 0.5 else rng.integers(1, 2)
    avg_payment_delay_hours = float(rng.exponential(scale=(1 - p_no_action) * 48 + 2))
    recent_payment_trend = rng.choice(["improving", "stable", "declining"],
                                       p=[0.25, 0.45, 0.30] if customer_type != CustomerType.DETERIORATING
                                       else [0.05, 0.25, 0.70])

    return dict(
        historical_success_rate=round(float(historical_success_rate), 3),
        previous_failure_count=int(previous_failure_count),
        previous_retry_success_rate=round(float(previous_retry_success_rate), 3),
        previous_nudge_response=round(float(previous_nudge_response), 3),
        recent_consecutive_failures=int(recent_consecutive_failures),
        avg_payment_delay_hours=round(avg_payment_delay_hours, 1),
        recent_payment_trend=recent_payment_trend,
    )


def generate(n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    # np.random.Generator.choice on a list of Enum members coerces them to
    # numpy strings and loses the Enum type -> select by index instead.
    types = list(TYPE_WEIGHTS.keys())
    type_p = np.array([TYPE_WEIGHTS[t] for t in types])
    type_p = type_p / type_p.sum()

    failure_reasons = list(FAILURE_REASON_WEIGHTS.keys())
    fr_p = np.array([FAILURE_REASON_WEIGHTS[r] for r in failure_reasons])
    fr_p = fr_p / fr_p.sum()

    methods = list(PAYMENT_METHOD_WEIGHTS.keys())
    method_p = np.array([PAYMENT_METHOD_WEIGHTS[m] for m in methods])
    method_p = method_p / method_p.sum()

    observable_rows = []
    hidden_rows = []

    for _ in range(n):
        payment_id = str(uuid.uuid4())
        customer_id = str(uuid.uuid4())

        customer_type = types[rng.choice(len(types), p=type_p)]
        hidden_probs = sample_hidden_probs(customer_type, rng)

        tier = rng.choice(["standard", "premium", "enterprise"], p=[0.6, 0.3, 0.1])
        subscription_amount = float(rng.choice(
            [199, 499, 999, 1999, 4999, 9999],
            p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05]
        ))
        billing_frequency = rng.choice(["monthly", "quarterly", "annual"], p=[0.7, 0.2, 0.1])
        subscription_age_days = int(rng.integers(1, 900))
        failure_reason = failure_reasons[rng.choice(len(failure_reasons), p=fr_p)]
        payment_method = methods[rng.choice(len(methods), p=method_p)]
        attempt_number = int(rng.integers(1, 4))

        obs_features = sample_observable_features(customer_type, hidden_probs, rng)

        # ---- Mode A: RANDOM action assignment (decoupled from hidden type) ----
        action = ACTIONS_FOR_RANDOM_ASSIGNMENT[rng.integers(0, len(ACTIONS_FOR_RANDOM_ASSIGNMENT))]
        prob_for_action = hidden_probs[ACTION_TO_PROB_INDEX[action]]
        success = bool(rng.random() < prob_for_action)

        observable_rows.append(dict(
            payment_id=payment_id,
            customer_id=customer_id,
            tier=tier,
            subscription_amount=subscription_amount,
            billing_frequency=billing_frequency,
            subscription_age_days=subscription_age_days,
            failure_reason=failure_reason.value,
            payment_method=payment_method.value,
            attempt_number=attempt_number,
            **obs_features,
            assigned_action=action.value,   # kept ONLY because Mode A is randomized
            outcome_success=success,        # label, safe to keep (it's the observed outcome)
        ))

        hidden_rows.append(dict(
            payment_id=payment_id,
            customer_type=customer_type.value,
            p_success_no_action=round(float(hidden_probs[0]), 4),
            p_success_retry=round(float(hidden_probs[1]), 4),
            p_success_nudge=round(float(hidden_probs[2]), 4),
            p_success_escalate=round(float(hidden_probs[3]), 4),
        ))

    return pd.DataFrame(observable_rows), pd.DataFrame(hidden_rows)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic recovery scenarios (Mode A: randomized).")
    parser.add_argument("--n", type=int, default=10000, help="Number of payment scenarios to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "output"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    obs_df, hidden_df = generate(args.n, args.seed)

    obs_path = os.path.join(args.out, "observable_dataset.csv")
    hidden_path = os.path.join(args.out, "hidden_ground_truth.csv")
    obs_df.to_csv(obs_path, index=False)
    hidden_df.to_csv(hidden_path, index=False)

    print(f"Generated {len(obs_df)} scenarios (seed={args.seed})")
    print(f"  observable dataset -> {obs_path}  ({obs_df.shape[1]} columns)")
    print(f"  hidden ground truth -> {hidden_path}  (evaluation-only, never feed to a model)")
    print("\nAction distribution (Mode A, should be ~uniform by design):")
    print(obs_df["assigned_action"].value_counts(normalize=True).round(3))
    print("\nObserved success rate by assigned action:")
    print(obs_df.groupby("assigned_action")["outcome_success"].mean().round(3))


if __name__ == "__main__":
    main()
