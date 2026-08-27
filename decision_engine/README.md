# Decision Engine — LangGraph Reasoning, Deterministic Guardrails & SQLite Audit

## Overview

The `decision_engine` package is the production-oriented orchestration and safety layer of the **Recovery Intelligence Engine**. It consumes the validated Day 4/5 machine learning policy (`CausalUpliftPolicy`) as an upstream dependency and layers:
1. **Deterministic Guardrails**: Hard safety constraints and state-transition validation.
2. **Azure AI Foundry Qualitative Reasoning**: Structured natural-language sanity checks using `gpt-4.1-mini`.
3. **Mock Execution**: Simulated recovery action dispatcher.
4. **SQLite Audit Persistence**: Full end-to-end provenance logging with strict UPSERT semantics.

---

## Architectural Principles

- **No ML/Policy Duplication**: `decision_engine` does not recalculate probabilities, treatment effects, or net values. It consumes `ml.decision.CausalUpliftPolicy` directly.
- **Authoritative Deterministic Safety**: The LLM proposed action is purely advisory. Deterministic guardrails have final authority over every recovery decision.
- **Fail-Safe Robustness**: If the LLM call fails, times out, or produces unpermitted actions twice, the engine falls back deterministically to `argmax` over permitted net values. If context or inputs are malformed, it defaults to `Action.WAIT` without crashing.
- **Strict Numerical Provenance**: All financial and probability figures (`expected_incremental_value`) are strictly bound from Python state calculations, never hallucinated by the LLM.

---

## LangGraph Workflow Topology

```
                  START
                    │
                    ▼
              context_node
                    │
                    ▼
             estimation_node
                    │
         ┌──────────┴──────────┐
         │ (error present)     │ (no error)
         ▼                     ▼
    error_fallback      reasoning_node
         │                     │
         │                     ▼
         │               guardrail_node
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
              execution_node (Persists to SQLite decision_engine/audit.db)
                    │
                    ▼
                   END
```

### Node Specifications

1. **`context_node` (`decision_engine/context_node.py`)**:
   - Resolves payment records from canonical datasets (`data/v2/payment_scenarios.csv` or `ml/evaluation/seed777_data/payment_scenarios.csv`).
   - Enforces strict partition separation across `observable_features`, `payment_context`, and `customer_history`.
   - Populates `permitted_actions` based on terminal/active state.

2. **`estimation_node` (`decision_engine/estimation_node.py`)**:
   - Consumes the existing CausalUpliftPolicy to obtain treatment-specific probabilities and cost-aware net values.
   - Computes expected net value: $\text{net\_val}(a) = \hat{P}(Y=1 \mid X, T=a) \times \text{amount} - \text{cost}(a)$.
   - Preserves error state without invoking models if upstream context failed.

3. **`reasoning_node` (`decision_engine/reasoning_node.py`)**:
   - Uses LangChain's `AzureAIOpenAIApiChatModel` configured with Microsoft Foundry's `/openai/v1` service endpoint.
   - Deployment: pinned to `gpt-4.1-mini` with `temperature = 0`.
   - Structured Pydantic output: validates schema `LLMDecision` (`decision`, `confidence`, `reasoning`, `risk_level`).
   - Retry logic: on validation error or out-of-permitted action, re-prompts with explicit correction instructions.
   - Fallback: if two attempts fail, executes deterministic `argmax` over permitted arm net values (`decision_source = "fallback_no_llm"`).

4. **`guardrail_node` (`decision_engine/guardrail_node.py`)**:
   - Evaluates the proposed action against `decision_engine.guardrails.check`.
   - Enforces 5 deterministic safety rules with hard precedence.
   - If overridden, replaces `final_action` with the guardrail verdict and records the override reason.

5. **`execution_node` (`decision_engine/execution_node.py`)**:
   - Mock execution: logs the action that would be dispatched to payment gateways or customer channels (no live external side-effects).
   - Calls `save_decision_audit` to persist the complete decision record to SQLite.

---

## Deterministic Guardrails Layer (`decision_engine/guardrails.py`)

Hard-coded safety constants and deterministic precedence order:

| Precedence | Rule Name | Trigger Condition | Overridden Action |
|:---:|:---|:---|:---:|
| **1** | **Invalid State Transition** | Payment status $\in$ `{"RECOVERED", "SUCCESS", "COMPLETED"}` | `Action.WAIT` |
| **2** | **Consecutive Failure Stop** | Consecutive failures $\ge 3$ (`MAX_CONSECUTIVE_FAILURES`) | `Action.STOP` |
| **3** | **Escalation Cap** | Proposed action is `ESCALATE` and lifetime escalations $\ge 1$ (`MAX_LIFETIME_ESCALATIONS`) | `Action.WAIT` |
| **4** | **Retry Limit** | Proposed action is `RETRY` and billing cycle retries $\ge 3$ (`MAX_RETRIES_PER_BILLING_CYCLE`) | `Action.WAIT` |
| **5** | **Intervention Window Limit** | Proposed action $\in$ `{RETRY, RETRY_NUDGE, ESCALATE}` and 7-day interventions $\ge 2$ | `Action.WAIT` |

---

## SQLite Audit Persistence (`decision_engine/audit.py`)

- **Database Path**: `decision_engine/audit.db` (ignored by Git via `.gitignore`).
- **Entity Model**: Reuses canonical `models.schemas.Decision` dataclass.
- **UPSERT Guarantee**: Enforced by SQLite database engine:
  ```sql
  INSERT INTO decision_audit (...) VALUES (...)
  ON CONFLICT(payment_id) DO UPDATE SET ...;
  ```
- **Persisted Fields**: `payment_id` (PK), `decision_id`, `raw_arm_probabilities`, `raw_arm_net_values`, `llm_proposed_decision`, `llm_confidence`, `llm_reasoning`, `llm_risk_level`, `expected_incremental_value`, `guardrail_verdict`, `guardrail_reason`, `final_action`, `decision_source`, `error_status`, `timestamp`.
- **Security**: No API keys or service endpoints are stored.

---

## Test Suite & Validation

The `decision_engine` maintains **40 dedicated tests** running 100% offline with zero external network or credential dependencies:

```bash
# Run decision engine tests
.venv\Scripts\python.exe -m pytest decision_engine -v
```

### Coverage Areas:
- `test_guardrails.py` (8 tests): Deterministic safety rules, precedence order, and input coercion.
- `test_reasoning_node.py` (5 tests): Structured output parsing, retry on failure, and fallback to argmax.
- `test_graph.py` (11 tests): Graph compilation, reducer audit trails, and end-to-end demonstration cases (Cases A, B, C).
- `test_audit.py` (7 tests): SQLite database initialization, record persistence, and direct SQL UPSERT verification.
- `test_validation_6e.py` (9 tests): Exhaustive regression suite verifying escalation caps, out-of-permitted action rejection, and graph determinism.
