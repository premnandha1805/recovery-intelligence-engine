"""
Ground truth generator (v2) — counterfactual probabilities + net recovery values.

Key changes from v1
-------------------
1.  Cumulative failure decay:
        p_natural is penalised by ATTEMPT_DECAY_PER_CYCLE × cumulative_failures
        so that a customer who has failed many times is genuinely harder to
        recover — not just noisier.

2.  Escalate is NOT guaranteed to win:
        escalate_responsiveness in the customer archetype can be NEGATIVE
        (sampled from TYPE_BASE_PROBS[3] − max(probs[1], probs[2])). For
        some customers escalation is less effective than retry or nudge.
        The signed value makes ESCALATE the worst action for ~20 % of cases.

3.  Expected net recovery value per action:
        net_value(action) = amount × p_success(action) − cost(action)

4.  Incremental value relative to WAIT:
        incremental_value(action) = net_value(action) − net_value(WAIT)
        WAIT has incremental_value = 0 by definition.
        Negative incremental_value means the action destroys value vs. doing nothing.

5.  Optimal action recorded:
        best_action is the action with the highest net_value.
        If WAIT has the highest net_value, best_action = "WAIT".

Hidden ground-truth output columns
------------------------------------
    payment_id
    natural_recovery_probability
    retry_success_probability
    nudge_success_probability
    escalation_success_probability
    net_value_wait
    net_value_retry
    net_value_nudge
    net_value_escalate
    incremental_value_retry    (relative to WAIT)
    incremental_value_nudge
    incremental_value_escalate
    best_action                (action maximising net_value)
"""

import numpy as np
import pandas as pd

from simulator_v2.config import (
    ACTION_COSTS,
    Action,
    REASON_MODIFIER,
    ATTEMPT_MODIFIER,
    ATTEMPT_DECAY_PER_CYCLE,
)


def attach_ground_truth(
    payments_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Build the ground-truth table with one row per payment attempt.

    Parameters
    ----------
    payments_df : DataFrame
        Observable payment records. Must include payment_id, customer_id,
        failure_reason, attempt_number, cumulative_failures, amount.
    customers_df : DataFrame
        Must include customer_id and _hidden_* columns.
    rng : np.random.Generator
        Seeded random generator.

    Returns
    -------
    DataFrame — hidden ground truth, one row per payment_id.
    """
    # Build a lookup from customer_id → hidden traits
    hidden_cols = [
        "customer_id",
        "_hidden_intrinsic_recovery_prob",
        "_hidden_retry_responsiveness",
        "_hidden_nudge_responsiveness",
        "_hidden_escalate_responsiveness",
    ]
    customer_hidden = customers_df[hidden_cols].set_index("customer_id")

    rows = []
    for _, pay in payments_df.iterrows():
        cust = customer_hidden.loc[pay["customer_id"]]

        base_natural = cust["_hidden_intrinsic_recovery_prob"]
        retry_resp   = cust["_hidden_retry_responsiveness"]
        nudge_resp   = cust["_hidden_nudge_responsiveness"]
        esc_resp     = cust["_hidden_escalate_responsiveness"]   # can be negative

        # --- Context modifiers ---
        reason_mod  = REASON_MODIFIER.get(pay["failure_reason"], 0.0)
        attempt_mod = ATTEMPT_MODIFIER.get(pay["attempt_number"], -0.20)

        # Cumulative failure decay — the more times this customer has failed
        # historically, the lower their baseline recovery probability.
        decay = min(pay["cumulative_failures"] * ATTEMPT_DECAY_PER_CYCLE, 0.40)

        # Per-payment noise (smaller than v1 to preserve decay signal)
        noise = rng.normal(0, 0.02, size=4)

        # --- Compute the four counterfactual probabilities ---
        natural  = base_natural + reason_mod + attempt_mod - decay + noise[0]
        retry    = natural + retry_resp + noise[1]
        nudge    = natural + nudge_resp + noise[2]
        # escalate is built relative to natural + escalate_responsiveness
        # esc_resp can be negative → escalate can be WORSE than retry/nudge
        escalate = natural + esc_resp + noise[3]

        # Clip to valid probability range
        natural  = float(np.clip(natural,  0.01, 0.99))
        retry    = float(np.clip(retry,    0.01, 0.99))
        nudge    = float(np.clip(nudge,    0.01, 0.99))
        escalate = float(np.clip(escalate, 0.01, 0.99))

        # --- Net recovery values (INR) ---
        amount = float(pay["amount"])
        nv_wait     = amount * natural  - ACTION_COSTS[Action.WAIT]
        nv_retry    = amount * retry    - ACTION_COSTS[Action.RETRY]
        nv_nudge    = amount * nudge    - ACTION_COSTS[Action.NUDGE]
        nv_escalate = amount * escalate - ACTION_COSTS[Action.ESCALATE]

        # --- Incremental values relative to WAIT ---
        iv_retry    = nv_retry    - nv_wait
        iv_nudge    = nv_nudge    - nv_wait
        iv_escalate = nv_escalate - nv_wait

        # --- Best action (maximises net_value) ---
        action_values = {
            Action.WAIT.value:     nv_wait,
            Action.RETRY.value:    nv_retry,
            Action.NUDGE.value:    nv_nudge,
            Action.ESCALATE.value: nv_escalate,
        }
        best_action = max(action_values, key=action_values.get)

        rows.append({
            "payment_id":                      pay["payment_id"],
            # Counterfactual probabilities
            "natural_recovery_probability":     round(natural,  4),
            "retry_success_probability":        round(retry,    4),
            "nudge_success_probability":        round(nudge,    4),
            "escalation_success_probability":   round(escalate, 4),
            # Net recovery values
            "net_value_wait":                   round(nv_wait,     2),
            "net_value_retry":                  round(nv_retry,    2),
            "net_value_nudge":                  round(nv_nudge,    2),
            "net_value_escalate":               round(nv_escalate, 2),
            # Incremental values (reference: WAIT)
            "incremental_value_retry":          round(iv_retry,    2),
            "incremental_value_nudge":          round(iv_nudge,    2),
            "incremental_value_escalate":       round(iv_escalate, 2),
            # Optimal action given ground truth
            "best_action":                      best_action,
        })

    return pd.DataFrame(rows)
