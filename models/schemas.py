"""
Core schemas for the Recovery Intelligence Engine.

These are the tables described in the project spec:
Customer, Subscription, Payment, PaymentAttempt, Intervention, Outcome, Decision.

Kept as plain dataclasses (not pydantic) so simulator/generate.py has zero
extra dependencies beyond numpy/pandas. Swap for pydantic models later if
the NestJS/FastAPI boundary needs request validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CustomerType(str, Enum):
    """Hidden behavioral type. NEVER exposed as a feature to the estimator."""
    TRANSIENT_FAILURE = "transient_failure"
    DETERIORATING = "deteriorating"
    CHRONIC_FAILURE = "chronic_failure"
    RESPONSIVE = "responsive"
    NON_RESPONSIVE = "non_responsive"
    HIGH_VALUE = "high_value"
    LOW_VALUE = "low_value"
    TEMPORARY_FAILURE = "temporary_failure"


class Action(str, Enum):
    WAIT = "WAIT"
    RETRY = "RETRY"
    RETRY_NUDGE = "RETRY_NUDGE"
    ESCALATE = "ESCALATE"
    STOP = "STOP"


class FailureReason(str, Enum):
    INSUFFICIENT_FUNDS = "insufficient_funds"
    CARD_EXPIRED = "card_expired"
    BANK_DECLINE = "bank_decline"
    NETWORK_ERROR = "network_error"
    ISSUER_TIMEOUT = "issuer_timeout"
    FRAUD_FLAG = "fraud_flag"


class PaymentMethod(str, Enum):
    CARD = "card"
    UPI = "upi"
    NETBANKING = "netbanking"
    WALLET = "wallet"


class GenerationMode(str, Enum):
    """
    Mode A: RANDOM   -> action assigned independent of hidden type. Used to
                         train/calibrate the uplift estimator without
                         confounding.
    Mode B: POLICY    -> action assigned by a named policy (no_action,
                         aggressive, rule_based, engine). Used only for the
                         final head-to-head baseline comparison, never for
                         training the estimator.
    """
    RANDOM = "random"
    POLICY = "policy"


# ---------------------------------------------------------------------------
# Hidden ground truth (simulator-only — never serialized to the estimator)
# ---------------------------------------------------------------------------

@dataclass
class HiddenGroundTruth:
    customer_type: CustomerType
    p_success_no_action: float
    p_success_retry: float
    p_success_nudge: float
    p_success_escalate: float
    responsiveness: float  # 0-1, how much this customer reacts to contact


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

@dataclass
class Customer:
    customer_id: str
    tier: str                      # "standard" | "premium" | "enterprise"
    signup_days_ago: int
    hidden: HiddenGroundTruth = field(repr=False)  # excluded from any export


@dataclass
class Subscription:
    subscription_id: str
    customer_id: str
    amount: float
    billing_frequency: str         # "monthly" | "quarterly" | "annual"
    subscription_age_days: int


@dataclass
class Payment:
    payment_id: str
    subscription_id: str
    amount: float
    status: str                    # "failed" | "success" | "pending"
    failure_reason: Optional[FailureReason]
    attempt_number: int
    payment_method: PaymentMethod


@dataclass
class PaymentAttempt:
    attempt_id: str
    payment_id: str
    attempt_number: int
    outcome: str                   # "success" | "failure"
    time_since_previous_attempt_hours: float


@dataclass
class Intervention:
    intervention_id: str
    payment_id: str
    action: Action
    mode: GenerationMode
    timestamp: str


@dataclass
class Outcome:
    payment_id: str
    success: bool
    recovered_amount: float
    recovery_latency_hours: float


@dataclass
class Decision:
    decision_id: str
    payment_id: str
    action: Action
    confidence: float
    expected_incremental_value: float
    numeric_source: str            # e.g. "uplift_model_v1" — audit provenance
    reasoning: str                 # LLM-generated rationale, not the number
    risk_level: str
    model_version: str
