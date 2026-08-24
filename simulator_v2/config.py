"""
Simulator v2 configuration — all constants, enums, and probabilities live here.

Changes from v1:
  - Added ACTION_COSTS dict for configurable cost-per-action (in INR).
  - Added ATTEMPT_DECAY_PER_CYCLE: penalty applied to p_natural for each
    cumulative prior failure the customer has accumulated.
  - Added NUDGE_FATIGUE_PER_NUDGE: small penalty when a customer has already
    received a nudge in the current billing cycle attempt sequence.
  - All other constants are identical to v1.
"""

from enum import Enum


# ---------------------------------------------------------------------------
# Enums (internal to the simulator — the model never sees CustomerType)
# ---------------------------------------------------------------------------

class CustomerType(str, Enum):
    """Hidden behavioral archetype. NEVER exposed as a feature."""
    RELIABLE = "reliable"
    NORMAL = "normal"
    DETERIORATING = "deteriorating"
    CHRONIC_FAILURE = "chronic_failure"
    RETRY_RESPONSIVE = "retry_responsive"
    NUDGE_RESPONSIVE = "nudge_responsive"


class FailureReason(str, Enum):
    INSUFFICIENT_FUNDS = "insufficient_funds"
    BANK_DECLINE = "bank_decline"
    NETWORK_ERROR = "network_error"
    EXPIRED_CARD = "expired_card"
    AUTHENTICATION_FAILURE = "authentication_failure"
    TEMPORARY_BANK_ISSUE = "temporary_bank_issue"


class PaymentMethod(str, Enum):
    CARD = "card"
    UPI = "upi"
    NETBANKING = "netbanking"
    WALLET = "wallet"


class Action(str, Enum):
    WAIT = "WAIT"
    RETRY = "RETRY"
    NUDGE = "NUDGE"
    ESCALATE = "ESCALATE"


# ---------------------------------------------------------------------------
# Action costs (INR) — the "price" of each intervention
#
# These are used to compute expected NET recovery value:
#   net_value(action) = amount × p_success(action) − cost(action)
#
# Incremental value is always computed relative to WAIT:
#   incremental_value(action) = net_value(action) − net_value(WAIT)
# ---------------------------------------------------------------------------

ACTION_COSTS: dict[Action, float] = {
    Action.WAIT:     0.00,    # Passive — no intervention
    Action.RETRY:    5.00,    # Gateway transaction fee
    Action.NUDGE:   15.00,    # SMS / email / push notification
    Action.ESCALATE: 250.00,  # Manual agent call / support ticket
}

# ---------------------------------------------------------------------------
# Customer-type distribution — how common each archetype is
# ---------------------------------------------------------------------------

CUSTOMER_TYPE_WEIGHTS = {
    CustomerType.RELIABLE:          0.25,
    CustomerType.NORMAL:            0.25,
    CustomerType.DETERIORATING:     0.15,
    CustomerType.CHRONIC_FAILURE:   0.10,
    CustomerType.RETRY_RESPONSIVE:  0.13,
    CustomerType.NUDGE_RESPONSIVE:  0.12,
}

# ---------------------------------------------------------------------------
# Hidden base probabilities per customer type
# (natural_recovery, retry_success, nudge_success, escalation_success)
#
# These are the simulator's private truth. Per-customer noise is added
# so the model can't reverse-engineer them from observable features.
# ---------------------------------------------------------------------------

TYPE_BASE_PROBS = {
    #                                   natural  retry   nudge   escalate
    CustomerType.RELIABLE:             (0.88,    0.92,   0.90,   0.93),
    CustomerType.NORMAL:               (0.55,    0.65,   0.62,   0.70),
    CustomerType.DETERIORATING:        (0.30,    0.45,   0.38,   0.50),
    CustomerType.CHRONIC_FAILURE:      (0.08,    0.12,   0.10,   0.15),
    CustomerType.RETRY_RESPONSIVE:     (0.40,    0.78,   0.48,   0.72),
    CustomerType.NUDGE_RESPONSIVE:     (0.38,    0.50,   0.80,   0.75),
}

# ---------------------------------------------------------------------------
# Failure-reason distribution
# ---------------------------------------------------------------------------

FAILURE_REASON_WEIGHTS = {
    FailureReason.INSUFFICIENT_FUNDS:      0.30,
    FailureReason.BANK_DECLINE:            0.22,
    FailureReason.NETWORK_ERROR:           0.18,
    FailureReason.EXPIRED_CARD:            0.12,
    FailureReason.AUTHENTICATION_FAILURE:  0.10,
    FailureReason.TEMPORARY_BANK_ISSUE:    0.08,
}

# ---------------------------------------------------------------------------
# Failure-reason modifiers: applied to all four action probabilities
# ---------------------------------------------------------------------------

REASON_MODIFIER = {
    "insufficient_funds":      -0.10,
    "bank_decline":            -0.05,
    "network_error":            0.05,
    "expired_card":            -0.15,
    "authentication_failure":  -0.08,
    "temporary_bank_issue":     0.08,
}

# ---------------------------------------------------------------------------
# Attempt-number modifier (per-attempt within a billing cycle)
# Later attempts are harder; this is applied on top of the failure-reason mod.
# ---------------------------------------------------------------------------

ATTEMPT_MODIFIER = {1: 0.00, 2: -0.06, 3: -0.12, 4: -0.20}

# ---------------------------------------------------------------------------
# Cumulative-failure decay: per prior lifetime failure of the customer.
# Captures the idea that a chronic failer is progressively harder to recover.
# Capped internally in ground_truth.py to avoid driving probabilities to 0.
# ---------------------------------------------------------------------------

ATTEMPT_DECAY_PER_CYCLE: float = 0.03   # subtracted from p_natural per prior failure

# ---------------------------------------------------------------------------
# ESCALATE is not always best: penalty applied when amount is small enough
# that the cost of escalation exceeds expected gross recovery uplift.
# This penalty is NOT applied to the probability — it is already handled by
# the net-value formula (cost=250). The design intentionally makes ESCALATE
# unprofitable on low-value subscriptions.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Payment-method distribution
# ---------------------------------------------------------------------------

PAYMENT_METHOD_WEIGHTS = {
    PaymentMethod.CARD:       0.40,
    PaymentMethod.UPI:        0.35,
    PaymentMethod.NETBANKING: 0.15,
    PaymentMethod.WALLET:     0.10,
}

# ---------------------------------------------------------------------------
# Subscription plan amounts (INR)
# ---------------------------------------------------------------------------

PLAN_AMOUNTS = [499, 799, 999, 1499, 1999, 2999, 4999, 9999]

PLAN_AMOUNT_WEIGHTS = [0.15, 0.12, 0.20, 0.15, 0.15, 0.10, 0.08, 0.05]

# ---------------------------------------------------------------------------
# Subscription tiers and billing frequencies
# ---------------------------------------------------------------------------

PLAN_TYPES = ["basic", "standard", "premium", "enterprise"]
PLAN_TYPE_WEIGHTS = [0.25, 0.35, 0.25, 0.15]

BILLING_FREQUENCIES = ["monthly", "quarterly", "annual"]
BILLING_FREQUENCY_WEIGHTS = [0.65, 0.25, 0.10]

# ---------------------------------------------------------------------------
# Default generation settings
# ---------------------------------------------------------------------------

DEFAULT_NUM_CUSTOMERS = 3_000
DEFAULT_NUM_BILLING_CYCLES = 12   # simulate 12 billing cycles per customer
MAX_ATTEMPTS_PER_CYCLE = 4        # max retry attempts within one failed cycle
DEFAULT_SEED = 42
