"""
policy/wait_policy.py — WaitPolicy: always returns Action.WAIT.

Baseline policy that never intervenes.  Useful as the "do-nothing" arm
in offline counterfactual evaluation.
"""

from models.schemas import Action
from policy.base import Policy


class WaitPolicy(Policy):
    """Always recommends waiting — no active intervention."""

    def decide(self, payment_features: dict) -> Action:
        return Action.WAIT
