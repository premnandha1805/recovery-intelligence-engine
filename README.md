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
4. Enforces deterministic safety guardrails, qualitative LLM sanity-checking (`gpt-4.1-mini`), and auditable PostgreSQL/SQLite execution via a compiled LangGraph workflow.
5. Exposes a resilient, production-ready NestJS API gateway coupled to a high-performance Python FastAPI decision microservice.

---

## 2. End-to-End System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           NestJS API Gateway                            │
│                     (Validation, Gateway Routing)                       │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ HTTP (X-Request-Id, JSON payload)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FastAPI Python Decision Engine                     │
│                (/evaluate, /health, Async Semaphore Pool)               │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       DecisionRepository Protocol                       │
│        ├── PostgreSQL (DEFAULT / Authoritative Production Mode)         │
│        └── SQLite (Explicit Backward-Compatible Test Mode)              │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      LangGraph Orchestration Core                       │
│    Context Retrieval ──► CATE Estimation ──► LLM Reasoner (gpt-4.1)    │
│                                                     │                   │
│    Audit Persistence ◄── Execution Node  ◄── Safety Guardrails          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities:
- **NestJS Gateway**: Client-facing public API gateway providing request validation, security headers, timeout management (8s deadline budget), and unified error response normalization.
- **FastAPI Decision Service**: High-concurrency async Python service exposing `/evaluate` and `/health`, managing concurrency locks, structured logging, and LangGraph workflow invocation.
- **DecisionRepository Abstraction**: Implementation-neutral storage contract isolating database persistence from domain logic.
- **PostgreSQL Production Backend**: Authoritative persistence for current decision states and immutable audit event ledgers.
- **SQLite Compatibility Backend**: Dedicated backward-compatible adapter strictly active when explicitly requested via configuration.
- **NO Silent Fallback**: If PostgreSQL is configured and becomes unavailable, the system **fails fast** with controlled error envelopes — it never silently degrades or falls back to SQLite.

---

## 3. Persistence Architecture

### PostgreSQL Configuration:
- **Engine**: PostgreSQL 17
- **Driver**: `psycopg` v3 with `psycopg_pool.AsyncConnectionPool`
- **Migration Strategy**: Native forward-only raw SQL migration files (no ORM, no Alembic, no asyncpg, no pgembed).
- **Connection Pool Configuration**:
  - `DB_POOL_MIN=2`
  - `DB_POOL_MAX=12`
  - `DB_CONNECT_TIMEOUT_MS=3000`

### Pool Lifecycle:
- **Startup**: Pool initializes in FastAPI lifespan context. The database is probed (`SELECT 1`). If the connection cannot be established within the timeout budget, the service aborts startup immediately (`RuntimeError`).
- **Shutdown**: Active pool connections are gracefully closed during FastAPI shutdown.

---

## 4. Database Schema & Migrations

### Migrations
Migrations are raw SQL files applied sequentially and tracked via the `schema_migrations` table:
1. `decision_engine/persistence/migrations/001_initial_decision_tables.sql` — Creates `decision_audit` and `decision_audit_events`.
2. `decision_engine/persistence/migrations/002_add_indexes.sql` — Creates performance indexes on `decision_audit_events(payment_id, evaluated_at ASC)`.

### Table: `decision_audit` (Current State)
Maintains exactly one current decision record per payment with true UPSERT semantics on `payment_id`:

| Column | Type | Description |
|:---|:---|:---|
| `payment_id` | `TEXT PRIMARY KEY` | Unique payment identifier |
| `decision_id` | `TEXT` | ID of the current decision |
| `request_id` | `TEXT` | Correlation ID of the evaluation request |
| `raw_arm_probabilities` | `JSONB` | Predicted recovery probabilities across actions |
| `raw_arm_net_values` | `JSONB` | Estimated net monetary value across actions |
| `llm_proposed_decision`| `TEXT` | Action proposed by the LLM reasoner |
| `llm_confidence` | `DOUBLE PRECISION`| LLM confidence score [0.0 - 1.0] |
| `llm_reasoning` | `TEXT` | Qualitative reasoning generated by LLM |
| `llm_risk_level` | `TEXT` | Risk categorization (`low`, `medium`, `high`) |
| `expected_incremental_value` | `DOUBLE PRECISION`| Estimated net gain over WAIT |
| `guardrail_verdict` | `TEXT` | Verdict (`passed`, `overridden`, `bypassed`) |
| `guardrail_reason` | `TEXT` | Explanation of guardrail safety rule |
| `final_action` | `TEXT` | Authoritative action executed |
| `decision_source` | `TEXT` | Decision provenance (`model`, `llm`, `guardrail`, `cache`) |
| `error` | `TEXT` | Error status or null |
| `evaluated_at` | `TIMESTAMPTZ` | Timestamp of evaluation |
| `state_fingerprint` | `TEXT` | SHA-256 state fingerprint hash |

### Table: `decision_audit_events` (Append-Only Event Ledger)
Immutable chronological audit history recording every evaluation attempt:

| Column | Type | Description |
|:---|:---|:---|
| `decision_id` | `TEXT PRIMARY KEY` | Unique decision event identifier |
| `payment_id` | `TEXT` | Associated payment ID |
| `request_id` | `TEXT` | Request correlation ID |
| `evaluated_at` | `TIMESTAMPTZ` | Event evaluation timestamp |
| `decision_source` | `TEXT` | Source of the decision |
| `final_action` | `TEXT` | Final action taken |
| `model_decision` | `TEXT` | Meta-learner ML optimal action |
| `llm_proposed_decision`| `TEXT` | LLM proposed action |
| `guardrail_overridden`| `BOOLEAN` | Whether guardrails modified the action |
| `guardrail_reason` | `TEXT` | Specific rule triggered |
| `state_fingerprint` | `TEXT` | State fingerprint at time of evaluation |

---

## 5. Authoritative Caching & State Fingerprints

The engine incorporates a PostgreSQL-backed authoritative cache to prevent redundant LLM invocations and unnecessary database writes.

### State Fingerprint Algorithm:
A SHA-256 hex digest computed over the canonical representation of payment features:
```python
fingerprint = sha256(
    f"{payment_id}|{status}|{attempt_number}|{consecutive_failures}|{retry_count}|{interventions_7d}".encode()
).hexdigest()
```

### Cache Decision Matrix:
- **Cache Hit**: Reusable when `force_recompute=false`, current decision exists in PostgreSQL, and persisted `state_fingerprint == current_state_fingerprint`.
  - Emits `db_cache_hit` event.
  - Returns cached evaluation with `decision_source = "cache"`.
  - **Zero** writes to `decision_audit` or `decision_audit_events`.
- **Cache Miss**: Occurs on new payments, state changes (fingerprint mismatch), or when `force_recompute=true`.
  - Emits `db_cache_miss` event.
  - Executes full LangGraph workflow.
  - Performs atomic write to `decision_audit` and `decision_audit_events`.

---

## 6. Transaction Atomicity & Rollback Guarantees

Every fresh evaluation persists both current state and event ledger rows inside a single atomic transaction:

```python
async with conn.transaction():
    # 1. Upsert current state into decision_audit
    await cur.execute(upsert_decision_sql, decision_params)
    # 2. Append immutable record into decision_audit_events
    await cur.execute(insert_event_sql, event_params)
```

- **Rollback Guarantee**: If the second write fails, the entire transaction is rolled back by `psycopg`. No orphaned or partial rows are ever committed.
- Validated via forced-failure testing with zero lingering database state.

---

## 7. Concurrency & Locking Architecture

### Validated Concurrency (Day 8F):
- **20 Different Payments Concurrently**: 20 requests dispatched simultaneously -> 20 successful evaluations, 20 `decision_audit` rows, 20 `decision_audit_events` rows, 0 database lock errors.
- **20 Same Payment Requests Concurrently**: 20 requests dispatched simultaneously for the identical payment -> 1 fresh LLM evaluation, 19 cache hits, 1 `decision_audit` row, 1 `decision_audit_events` row, 20 identical successful responses.

### ⚠️ Concurrency Limitation & Multi-Worker Scaling:
- **Current Implementation**: Uses an in-process asyncio lock dictionary (`app.state.payment_locks`) protected by an `asyncio.Lock()`.
- **Limitation**: This lock is **process-local**. It does **NOT** coordinate across multiple Uvicorn worker processes or multiple container replicas.
- **Multi-Replica Production Deployment**: Multi-worker deployments require distributed coordination such as **PostgreSQL Advisory Locks** (`pg_advisory_xact_lock(hashtext(payment_id))`) or Redis distributed locks.

---

## 8. Request Correlation & Structured Observability

Every request carries a unique `request_id` (via `X-Request-Id` header) that propagates across the entire stack:
```text
HTTP Request (X-Request-Id) ──► FastAPI ──► Repository ──► PostgreSQL ──► Structured JSON Logs
```

### Structured PostgreSQL Observability Events (Day 8J):
Emitted as standardized JSON payloads containing `timestamp`, `service="python-decision-engine"`, `event`, and `request_id`:

1. `db_connection_acquired` — Connection borrowed from pool.
2. `db_transaction_started` — Transaction block entered.
3. `db_transaction_committed` — Transaction successfully committed.
4. `db_transaction_rolled_back` — Transaction failed and was rolled back (includes `error_type`).
5. `db_cache_hit` — Valid fingerprint match found in PostgreSQL cache.
6. `db_cache_miss` — Cache miss or force_recompute triggered.
7. `db_persistence_failed` — Repository write operation failed.

### Security Restrictions:
Logs are strictly sanitized. They **never** contain:
- `DATABASE_URL` or `TEST_DATABASE_URL`
- Passwords or credentials
- Bearer tokens or API keys
- Raw SQL queries with parameter values
- Sensitive PII or LLM prompt texts

---

## 9. Failure & Recovery Behavior

### Startup Fail-Fast:
- If PostgreSQL is unreachable at FastAPI startup, the connection probe times out (3000ms budget).
- The application fails fast with `RuntimeError` and refuses to accept traffic.
- **Zero silent SQLite fallback**.

### Mid-Request Connection Loss:
- If the database fails during an in-flight request, the transaction is rolled back.
- The request terminates cleanly and returns a sanitized `500 INTERNAL_ERROR` envelope with the original `x-request-id` header.
- No raw tracebacks or SQL errors are exposed to the client.
- No false persistence is reported.
- **No automatic client/server retry loop** (prevents thundering herd on recovering databases).

---

## 10. Health & Readiness Endpoints

### Python `/health` and `/ready`:
- Performs an active lightweight probe (`SELECT 1;`) against the PostgreSQL pool with a strict 2-second timeout.

**Healthy Response (200 OK):**
```json
{
  "status": "ok",
  "dependencies": {
    "database": "ok",
    "decision_engine": "ok"
  }
}
```

**Degraded Response (200 OK with degraded status):**
```json
{
  "status": "degraded",
  "dependencies": {
    "database": "unavailable",
    "decision_engine": "ok"
  }
}
```

### NestJS Health Integration:
- NestJS monitors Python `/health`.
- If Python returns `status: "degraded"`, NestJS reports overall status as `degraded` and returns `503 Service Unavailable` with `dependencies: { decision_engine: "unreachable" }`.

---

## 11. Environment Variables

| Variable | Default | Purpose |
|:---|:---|:---|
| `PERSISTENCE_BACKEND` | `postgres` | Persistence engine (`postgres` or `sqlite`) |
| `DATABASE_URL` | `postgresql://...` | Production PostgreSQL connection URL |
| `TEST_DATABASE_URL` | `postgresql://...` | Isolated PostgreSQL test database URL (Port 5433) |
| `DB_POOL_MIN` | `2` | Minimum pool connections |
| `DB_POOL_MAX` | `12` | Maximum pool connections |
| `DB_CONNECT_TIMEOUT_MS` | `3000` | Database connection timeout in milliseconds |
| `AZURE_AI_PROJECT_ENDPOINT` | — | Azure AI Foundry endpoint for `gpt-4.1-mini` |
| `AZURE_AI_MODEL_DEPLOYMENT_NAME` | `gpt-4.1-mini` | Azure deployment model name |

---

## 12. Local Setup & Testing

### Prerequisites:
- Python 3.11+ (virtual environment at `.venv`)
- Node.js 18+ & npm
- Docker Desktop running

### Starting Local Test Infrastructure:
```powershell
# Start PostgreSQL Test Instance (Port 5433, db: recovery_test)
docker compose -f docker-compose.test.yml up -d

# Start PostgreSQL Development Instance (Port 5432)
docker compose up -d
```

### Running Test Suites:
```powershell
# 1. Full Python Suite (261 tests offline with deterministic mocks)
.venv\Scripts\python.exe -m pytest -q

# 2. Persistence Test Suite
.venv\Scripts\python.exe -m pytest decision_engine/persistence/ -v

# 3. Concurrency Test Suite (Day 8F)
.venv\Scripts\python.exe -m pytest decision_engine/test_day8f_concurrency.py -v -s

# 4. PostgreSQL Cutover & Health Suites (Day 8G, 8H)
.venv\Scripts\python.exe -m pytest decision_engine/test_day8g_cutover.py decision_engine/test_day8h_health.py -v

# 5. Failure Recovery & Observability Suites (Day 8I, 8J)
.venv\Scripts\python.exe -m pytest decision_engine/test_day8i_failure_recovery.py decision_engine/test_day8j_observability.py -v

# 6. NestJS Gateway Test Suite & Build
cd backend
npm test
npm run build
cd ..
```

### Explicit SQLite Compatibility Mode:
```powershell
# Run SQLite tests explicitly
$env:PERSISTENCE_BACKEND="sqlite"
.venv\Scripts\python.exe -m pytest decision_engine/test_service.py decision_engine/test_audit.py -v
$env:PERSISTENCE_BACKEND="postgres"
```

---

## 13. Known Limitations

1. **Warm Dataset Cache**: The in-memory dataset reference is loaded at process startup for high-speed feature lookups. External updates to historical dataset CSVs require a service restart or explicit reload mechanism.
2. **Process-Local Concurrency Lock**: Same-payment deduplication uses an in-process asyncio lock. Multi-worker or multi-replica deployments must implement PostgreSQL advisory locks.
3. **Database Recovery Lifecycle**: A severed connection pool requires process restart or pool reinitialization; it does not implement continuous speculative reconnect loops.

---

## 14. License & Disclaimer

Internal research and development project — not for external redistribution.
Simulations represent controlled synthetic benchmarks with counterfactual ground truth and do not constitute external production claims.
