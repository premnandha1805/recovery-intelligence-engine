"""
policy/nudge_policy.py — AlwaysNudgePolicy: always returns Action.RETRY_NUDGE.

Baseline that always pairs the retry with a customer nudge (notification /
dunning message).  Uses Action.RETRY_NUDGE from the existing enum — no new
action value is introduced.
"""

from models.schemas import Action
from policy.base import Policy


class AlwaysNudgePolicy(Policy):
    """Always recommends a retry accompanied by a customer nudge."""

    def decide(self, payment_features: dict) -> Action:
        return Action.RETRY_NUDGE
