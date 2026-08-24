"""
Simulator configuration — all constants, enums, and probabilities live here.

This keeps the generator modules clean and makes it easy to tune the
synthetic world in one place.
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

DEFAULT_NUM_PAYMENTS = 10_000
DEFAULT_CUSTOMER_RATIO = 0.30    # ~3,000 customers for 10,000 payments
DEFAULT_SEED = 42
