# Recovery Intelligence Engine

AI-assisted, causal-uplift recovery decision engine for recurring/subscription payment failures.

---

## 1. Problem & Core Objective

A failed recurring payment does not necessarily mean lost revenue, and a payment that recovers after an intervention does not imply the intervention caused the recovery. Conventional dunning and retry policies suffer from:
- **Treatment-Selection Bias**: Aggressive interventions are confounded with customer risk profiles.
- **Excessive Intervention Cost**: Over-escalating to expensive channels (e.g. manual support tickets or SMS nudges) erodes net margins.
- **Negative Treatment Effects**: Unnecessary outreach can alienate responsive customers and induce churn.

**Goal**: Build an end-to-end intelligence system that:
1. Simulates stateful recurring payment lifecycles with hidden counterfactual ground truth.
2. Estimates Conditional Average Treatment Effects (CATE) across candidate recovery actions (`WAIT`, `RETRY`, `RETRY_NUDGE`, `ESCALATE`) using a meta-learner (`T-Learner`) with strict leakage protection.
3. Deploys an expected-net-value maximizing causal policy (`CausalUpliftPolicy`).
4. Enforces deterministic safety guardrails, qualitative LLM sanity-checking (`gpt-4.1-mini`), and auditable SQLite execution via a compiled LangGraph workflow.

---

## 2. Repository Architecture

```
recovery-intelligence-engine/
├── simulator/          Stateful synthetic payment generator & counterfactual ground truth
│   ├── customer_generator.py      Customer profiles (observable + hidden)
│   ├── subscription_generator.py  Subscription records
│   ├── payment_generator.py       Failed payment records
│   ├── ground_truth.py            Hidden counterfactual probabilities
│   └── generate.py                Simulation orchestrator
│
├── policy/             Recovery decision policies & action cost configuration
│   ├── base.py                    Base policy interface
│   ├── cost_config.py             Canonical action costs (WAIT: 0, RETRY: 5, NUDGE: 15, ESCALATE: 250)
│   ├── rule_based_policy.py       Day 3 heuristic baseline
│   ├── rule_based_policy_v2.py    Day 5 canonical cost-aware rule baseline
│   └── run_policies.py            Policy batch execution runner
│
├── ml/                 Causal inference & machine learning layer
│   ├── generate_causal_training_data.py  Uniform randomized treatment logger
│   ├── data/causal_training_data.csv     Randomized logging dataset (N = 30,472)
│   ├── dataset.py                        Feature contract (X, T, Y)
│   ├── firewall.py                       Leakage firewall (rejects hidden simulator probabilities)
│   ├── splits.py                         Customer-level 5-fold GroupKFold
│   ├── t_learner.py                      T-Learner multi-arm logistic meta-learner
│   ├── treatment_effects.py              Out-of-fold CATE estimator
│   ├── decision.py                       Net-value maximizing CausalUpliftPolicy
│   └── evaluation/                       Out-of-sample evaluation scripts and reports
│
├── decision_engine/    Day 6 LangGraph orchestration, guardrails & SQLite audit
│   ├── state.py                          RecoveryState schema with audit reducer
│   ├── context_node.py                   Payment context retrieval & partition separation
│   ├── estimation_node.py                Causal uplift prediction node
│   ├── reasoning_node.py                 Azure AI Foundry gpt-4.1-mini reasoning node
│   ├── guardrails.py                     5 deterministic safety rules with strict precedence
│   ├── guardrail_node.py                 Authoritative safety wrapper node
│   ├── execution_node.py                 Mock action dispatcher & audit logger
│   ├── graph.py                          Compiled LangGraph workflow
│   ├── audit.py                          SQLite persistence with true UPSERT semantics
│   └── run_day6d_evaluation.py          20-payment demonstration & UPSERT verification
│
├── models/             Canonical data contracts & Action/Decision dataclasses
│   └── schemas.py
│
├── evaluation/         Baseline policy evaluation framework & reporting
│   ├── evaluator.py                      Single source of truth for policy metrics
│   ├── report.py                         Aggregate report generator
│   └── test_evaluator.py                 Evaluator unit tests
│
└── data/v2/            Canonical payment scenarios and ground truth datasets
```

---

## 3. Validated Research Results (Day 5 Freeze)

`CausalUpliftPolicy` was evaluated using the frozen evaluation harness (`evaluation/evaluator.py`) across out-of-sample (OOS) populations and benchmarked against the canonical cost-aware `RuleBasedPolicyV2`:

| Population | Policy | Net Recovered Value (INR) | Gross Recovery Rate | Action Cost (INR) | Causal Net Lead | Rank |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Seed 777 (OOS)** | **CausalUpliftPolicy** | **₹16,486,990.85** | **27.63%** | **₹253,805.00** | **+₹376,348.36 (+2.34%)** | **#1** |
| Seed 777 (OOS) | RuleBasedPolicyV2 | ₹16,110,642.49 | 27.24% | ₹493,520.00 | Baseline | #2 |
| Seed 777 (OOS) | AlwaysRetryPolicy | ₹15,360,307.59 | 25.49% | ₹148,595.00 | — | #3 |
| Seed 777 (OOS) | RuleBasedPolicy (D3) | ₹14,433,756.53 | 25.93% | ₹1,554,965.00 | — | #4 |
| | | | | | | |
| **Seed 555 (OOS)** | **CausalUpliftPolicy** | **₹17,491,398.62** | **27.59%** | **₹265,335.00** | **+₹347,800.01 (+2.03%)** | **#1** |
| Seed 555 (OOS) | RuleBasedPolicyV2 | ₹17,143,598.61 | 27.23% | ₹503,630.00 | Baseline | #2 |
| Seed 555 (OOS) | AlwaysRetryPolicy | ₹16,249,456.48 | 25.44% | ₹151,445.00 | — | #3 |
| Seed 555 (OOS) | RuleBasedPolicy (D3) | ₹14,817,685.72 | 25.69% | ₹1,568,825.00 | — | #4 |

**Key Takeaway**: `CausalUpliftPolicy` beats all heuristic and rule-based baselines across both out-of-sample populations by selectively concentrating high-touch actions on accounts where incremental recovery exceeds intervention costs while saving over ₹240,000 in unnecessary action expenditures.

---

## 4. Decision Engine Workflow & Safety Guardrails (Day 6)

```
START ──► context_node ──► estimation_node ──► [conditional route]
                                                     │
               ┌─────────────────────────────────────┴─────────────────────────────────────┐
               ▼                                                                           ▼
         (error present)                                                               (no error)
         error_fallback                                                              reasoning_node (gpt-4.1-mini)
               │                                                                           │
               │                                                                           ▼
               │                                                                     guardrail_node (Authoritative)
               │                                                                           │
               └─────────────────────────────────────┬─────────────────────────────────────┘
                                                     │
                                                     ▼
                                              execution_node
                                         (SQLite UPSERT Persistence)
                                                     │
                                                     ▼
                                                    END
```

### Deterministic Precedence Rules:
1. **Invalid State Transition**: Rejects active actions on already recovered/successful payments → overrides to `Action.WAIT`.
2. **Consecutive Failure Stop**: Halts recovery after $\ge 3$ consecutive cycle failures → overrides to `Action.STOP`.
3. **Escalation Cap**: Caps lifetime human escalations at 1 per customer account → overrides to `Action.WAIT`.
4. **Retry Limit**: Restricts billing cycle retries to $< 3$ attempts → overrides to `Action.WAIT`.
5. **Intervention Window Limit**: Limits total active interventions to $< 2$ within any 7-day rolling window → overrides to `Action.WAIT`.

---

## 5. Verification & Testing

The test suite runs 100% offline, requires no network access or Azure credentials, and executes in ~10 seconds:

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run the complete test suite (138 tests)
pytest -q

# Run decision engine tests only (40 tests)
pytest decision_engine -v
```

---

## 6. License & Disclaimer

Internal research and development project — not for external redistribution.
Simulations represent controlled synthetic benchmarks with counterfactual ground truth and do not constitute production causal claims.
