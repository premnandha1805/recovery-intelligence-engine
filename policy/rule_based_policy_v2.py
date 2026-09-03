"""
policy/rule_based_policy_v2.py — Canonical Economically Cost-Aware Rule-Based Policy.

Observable features used:
-------------------------------------------------------------------------
amount                          : float  > 0     — transaction amount
dynamic_success_rate            : float  [0, 1]  — running success rate
consecutive_failed_cycles       : int    >= 0    — streak of failed cycles
contact_response_score          : float  [0, 1]  — proxy for retry responsiveness
notification_engagement_score   : float  [0, 1]  — proxy for nudge responsiveness
"""

from __future__ import annotations

import pandas as pd
from models.schemas import Action
from policy.base import Policy
from policy.cost_config import ACTION_COSTS


class RuleBasedPolicyV2(Policy):
    """
    Economically cost-aware deterministic rule-based policy.

    Replaces the arbitrary escalation threshold (consec_fails >= 3 and dyn_sr < 0.30)
    with a cost-aware expected-value comparison using observable proxies:
        net_val(action) = (proxy_p_success * amount) - cost(action)

    Uses NO hidden probabilities and NO learned ML models.
    """

    _REQUIRED_COLS = (
        "dynamic_success_rate",
        "consecutive_failed_cycles",
        "contact_response_score",
        "notification_engagement_score",
        "amount",
    )

    def decide(self, payment_features: dict) -> Action:
        missing = [c for c in self._REQUIRED_COLS if c not in payment_features]
        if missing:
            raise KeyError(
                f"RuleBasedPolicyV2 requires observable columns {missing}. "
                f"Keys provided: {sorted(payment_features.keys())}"
            )

        dyn_sr = float(payment_features["dynamic_success_rate"])
        contact_score = float(payment_features["contact_response_score"])
        notif_score = float(payment_features["notification_engagement_score"])
        amount = float(payment_features["amount"])

        # Rule 1 — High dynamic success rate: self-recovers
        if dyn_sr >= 0.80:
            return Action.WAIT

        # Observable proxy estimates for recovery success
        p_wait = dyn_sr
        p_retry = min(0.95, dyn_sr * (1.0 + 0.30 * contact_score))
        p_nudge = min(0.95, dyn_sr * (1.0 + 0.35 * notif_score))
        p_escalate = min(0.95, dyn_sr * 1.20 + 0.08)

        # Expected net values = (proxy_p * amount) - ACTION_COSTS
        net_wait = p_wait * amount - float(ACTION_COSTS[Action.WAIT])
        net_retry = p_retry * amount - float(ACTION_COSTS[Action.RETRY])
        net_nudge = p_nudge * amount - float(ACTION_COSTS[Action.RETRY_NUDGE])
        net_escalate = p_escalate * amount - float(ACTION_COSTS[Action.ESCALATE])

        best_action = Action.WAIT
        best_net = net_wait

        if net_retry > best_net:
            best_action = Action.RETRY
            best_net = net_retry

        if net_nudge > best_net:
            best_action = Action.RETRY_NUDGE
            best_net = net_nudge

        if net_escalate > best_net:
            best_action = Action.ESCALATE
            best_net = net_escalate

        return best_action

    def decide_batch(self, df: pd.DataFrame) -> list[str]:
        """Vectorized execution for fast batch processing."""
        dyn_sr = df["dynamic_success_rate"].astype(float).values
        contact_score = df["contact_response_score"].astype(float).values
        notif_score = df["notification_engagement_score"].astype(float).values
        amount = df["amount"].astype(float).values

        cost_wait = float(ACTION_COSTS[Action.WAIT])
        cost_retry = float(ACTION_COSTS[Action.RETRY])
        cost_nudge = float(ACTION_COSTS[Action.RETRY_NUDGE])
        cost_escalate = float(ACTION_COSTS[Action.ESCALATE])

        actions = []
        for i in range(len(df)):
            dsr = dyn_sr[i]
            if dsr >= 0.80:
                actions.append("WAIT")
                continue

            amt = amount[i]
            pw = dsr
            pr = min(0.95, dsr * (1.0 + 0.30 * contact_score[i]))
            pn = min(0.95, dsr * (1.0 + 0.35 * notif_score[i]))
            pe = min(0.95, dsr * 1.20 + 0.08)

            nw = pw * amt - cost_wait
            nr = pr * amt - cost_retry
            nn = pn * amt - cost_nudge
            ne = pe * amt - cost_escalate

            best_a = "WAIT"
            best_n = nw
            if nr > best_n:
                best_a = "RETRY"
                best_n = nr
            if nn > best_n:
                best_a = "RETRY_NUDGE"
                best_n = nn
            if ne > best_n:
                best_a = "ESCALATE"
                best_n = ne
            actions.append(best_a)

        return actions
