"""
policy — collection of recovery action policies.

Public API
----------
Policy          : abstract base class
WaitPolicy      : always returns Action.WAIT
AlwaysRetryPolicy  : always returns Action.RETRY
AlwaysNudgePolicy  : always returns Action.RETRY_NUDGE
RuleBasedPolicy : threshold-driven heuristic using observable features only
"""

from policy.base import Policy
from policy.wait_policy import WaitPolicy
from policy.retry_policy import AlwaysRetryPolicy
from policy.nudge_policy import AlwaysNudgePolicy
from policy.rule_based_policy import RuleBasedPolicy

__all__ = [
    "Policy",
    "WaitPolicy",
    "AlwaysRetryPolicy",
    "AlwaysNudgePolicy",
    "RuleBasedPolicy",
]
