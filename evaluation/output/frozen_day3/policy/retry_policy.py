"""
policy/retry_policy.py — AlwaysRetryPolicy: always returns Action.RETRY.

Aggressive baseline that unconditionally retries every failed payment.
"""

from models.schemas import Action
from policy.base import Policy


class AlwaysRetryPolicy(Policy):
    """Always recommends an immediate retry — no communication sent."""

    def decide(self, payment_features: dict) -> Action:
        return Action.RETRY
