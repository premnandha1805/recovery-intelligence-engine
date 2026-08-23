# Recovery Intelligence Engine

## Problem
A failed recurring payment is not automatically a lost payment, and a payment
that succeeds after an intervention was not necessarily *caused* to succeed
by that intervention. Conventional recovery systems (including Razorpay's
existing Subscription Recovery) execute retries/nudges/escalations well, but
don't answer a prior question: **is intervention actually likely to change
this specific outcome, or would the payment have recovered — or failed —
regardless?**

## Solution
A decision-and-evaluation layer sitting between failure detection and
recovery execution. For each failed payment it estimates the *incremental*
probability that intervention changes the outcome (uplift / CATE — not raw
success probability), picks the minimum action justified by that estimate
under deterministic guardrails, and logs an auditable decision. The policy
is evaluated against baseline policies (no-action, aggressive, rule-based)
on a shared synthetic population with known hidden counterfactual truth.

This is a **decision + evaluation layer**, not a recovery executor. It is
explicitly positioned as complementary to, not a replacement for, Razorpay's
existing recovery execution capabilities.

## Scope
- One vertical only: failed recurring/subscription payments.
- Action space: WAIT / RETRY / RETRY+NUDGE / ESCALATE / STOP.
- Fully synthetic environment — no real Razorpay data or API calls.
- Estimation is done by a trained/calibrated statistical model
  (logistic regression / gradient boosting, T-learner CATE estimation).
  The LLM layer consumes that estimate and produces structured judgment,
  tie-breaking, and human-readable audit rationale — it does not compute
  the probability itself.

## Assumptions
- Real-world treatment assignment is never randomized; this project uses a
  **randomized data-generation pass** (Mode A) specifically to avoid
  confounded/circular uplift estimation, separate from the **policy
  evaluation pass** (Mode B) used for the final baseline comparison.
- All results are **simulated recovery improvement against defined
  baselines**, not proven production causal uplift.
- Hidden ground-truth probabilities exist only in the simulator and are
  never exposed to the estimator, the LLM, or the decision engine.

## What we do NOT build
- Voice/WhatsApp/email integrations (mocked only)
- Multi-channel orchestration, B2B collections, loans, fraud detection
- RAG / vector databases
- Multi-agent swarms, reinforcement learning, contextual bandits
- Real Razorpay API integration
- Kubernetes, Kafka, unnecessary microservices
- An LLM that outputs the success-probability number directly (that's the
  estimator's job, not the LLM's)

## Repository layout
```
recovery-intelligence-engine/
  backend/     NestJS service (decision API, guardrails, audit trail)
  simulator/   Synthetic payment environment (Mode A + Mode B generators)
  models/      Schemas + the uplift/CATE estimator
  evaluation/  Baseline policies + Qini/AUUC evaluation framework
  frontend/    Dashboard
  docs/        Architecture notes, simulator assumptions, limitations
```

## Limitations (stated up front, not buried)
Synthetic data is not real Razorpay production data. Counterfactual ground
truth comes from the simulator, not from a real experiment. LLM reasoning
does not establish causality. This is a prototype decision layer, not a
production payment-recovery system.
