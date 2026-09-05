"""
policy/rule_based_policy.py — Threshold-driven heuristic policy (V2 schema).

Observable features used (strictly required — absent keys raise KeyError)
-------------------------------------------------------------------------
dynamic_success_rate           : float  [0, 1]  — running success rate at decision time
consecutive_failed_cycles      : int    >= 0    — ongoing streak of failed billing cycles
contact_response_score         : float  [0, 1]  — observable proxy for retry responsiveness
notification_engagement_score  : float  [0, 1]  — observable proxy for nudge responsiveness
"""

from __future__ import annotations

from models.schemas import Action
from policy.base import Policy

# ── Tunable thresholds (V2 observable schema) ────────────────────────────
HIGH_SUCCESS_THRESHOLD: float = 0.80
CONSEC_FAIL_THRESHOLD: int = 3
LOW_SUCCESS_THRESHOLD: float = 0.30


class RuleBasedPolicy(Policy):
    """
    Deterministic heuristic policy using real V2 observable features.
    Fails loudly with KeyError if any required V2 column is missing.
    """

    _REQUIRED_COLS = (
        "dynamic_success_rate",
        "consecutive_failed_cycles",
        "contact_response_score",
        "notification_engagement_score",
    )

    def decide(self, payment_features: dict) -> Action:
        missing = [c for c in self._REQUIRED_COLS if c not in payment_features]
        if missing:
            raise KeyError(
                f"RuleBasedPolicy requires V2 observable columns {missing}. "
                f"Keys provided: {sorted(payment_features.keys())}"
            )

        dyn_sr: float = float(payment_features["dynamic_success_rate"])
        consec_fails: int = int(payment_features["consecutive_failed_cycles"])
        contact_score: float = float(payment_features["contact_response_score"])
        notif_score: float = float(payment_features["notification_engagement_score"])

        # Rule 1 — High dynamic success rate: customer likely self-recovers
        if dyn_sr >= HIGH_SUCCESS_THRESHOLD:
            return Action.WAIT

        # Rule 2 — Persistent failed-cycle streak + low success rate: escalate
        if consec_fails >= CONSEC_FAIL_THRESHOLD and dyn_sr < LOW_SUCCESS_THRESHOLD:
            return Action.ESCALATE

        # Rule 3 — Retry-responsiveness proxy exceeds nudge-responsiveness proxy
        if contact_score > notif_score:
            return Action.RETRY

        # Rule 4 — Default: Nudge alongside retry
        return Action.RETRY_NUDGE

