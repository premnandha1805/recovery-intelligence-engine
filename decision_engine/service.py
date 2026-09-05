"""
decision_engine/service.py
==========================
FastAPI Decision Engine HTTP Service.

This is the integration boundary between NestJS and the Python Decision Engine.
Wraps the Day 6 LangGraph architecture, deterministic guardrails, CausalUpliftPolicy,
and SQLite audit persistence with concurrency-safe idempotency and deadline-aware
LLM execution.

RUN MODE — IMPORTANT:
This service MUST run as a single worker process:
    uvicorn decision_engine.service:app --port 8000 --workers 1

The in-memory per-payment lock (app.state.payment_locks) and the LLM concurrency
semaphore (app.state.llm_semaphore) are process-local. Running multiple workers would
silently break both idempotency-under-concurrency and the concurrency cap. If horizontal
scaling is needed later, that requires an external coordination layer (e.g. Redis) —
explicitly out of scope for Day 7. [FIX-4]
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv(".env.test")
load_dotenv()

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from contextlib import asynccontextmanager
import datetime
import json
import logging
import os
import pathlib
import time
from typing import Any, Optional
import uuid

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from decision_engine.persistence import (
    DecisionRepository,
    PostgresDecisionRepository,
    SqliteDecisionRepository,
    create_postgres_pool,
    close_postgres_pool,
    check_pool_health,
)
from decision_engine.persistence.migrate import run_migrations, MigrationError
from decision_engine.persistence.sqlite import open_sqlite_repository

from decision_engine.state import RecoveryState
from decision_engine.graph import create_recovery_graph
from decision_engine.audit import (
    DEFAULT_AUDIT_DB_PATH,
    compute_state_fingerprint,
)
from decision_engine.context_node import get_payment_state, _get_dataset
import math

from decision_engine.config import (
    DecisionEngineConfig,
    load_and_validate_config,
    ConfigValidationError,
)
from ml.decision import CausalUpliftPolicy
from ml.dataset import OBSERVABLE_FEATURES, FORBIDDEN_PREFIXES
from models.schemas import Action, PaymentMethod, FailureReason
from decision_engine.structured_logger import emit_log

# Configure structured logging
logger = logging.getLogger("decision_engine.service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Canonical Inference Vocabularies (Sourced from Repository Constants) ─────
VALID_PAYMENT_METHODS: frozenset[str] = frozenset(m.value for m in PaymentMethod)
VALID_FAILURE_REASONS: frozenset[str] = frozenset(r.value for r in FailureReason) | {
    "network_error",
    "insufficient_funds",
    "temporary_bank_issue",
    "bank_decline",
    "expired_card",
    "authentication_failure",
}
ALLOWED_FEATURE_KEYS: frozenset[str] = frozenset(OBSERVABLE_FEATURES) | {
    "status",
    "consecutive_failures",
    "retry_count_current_cycle",
    "retry_count",
    "lifetime_escalations",
    "interventions_last_7_days",
    "interventions_7d",
    "customer_id",
    "days_active",
}


# ── Request / Response Schemas ───────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    payment_id: str
    request_id: Optional[str] = None
    force_recompute: bool = False
    features: Optional[dict[str, Any]] = None


# ── Feature Validation & State Construction ──────────────────────────────────

def validate_and_parse_features(
    payment_id: str,
    raw_features: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Strict validation and partition construction for caller-supplied payment features.

    Validates:
    - Zero forbidden simulator/counterfactual fields (ml.dataset.FORBIDDEN_PREFIXES).
    - Presence of all 9 required observable model features (ml.dataset.OBSERVABLE_FEATURES).
    - Rejection of undeclared/extra fields.
    - Type and range safety (rejection of NaN, Infinity, negative counts, invalid categoricals).
    - String length safety.

    Applies documented conservative cold-start defaults for operational context:
    - status: "failed" (conservative default for recovery intelligence evaluation)
    - lifetime_escalations: 0 (cold start: no prior human escalations on record)
    - interventions_last_7_days: 0 (cold start: no prior interventions on record)

    Enforces authoritative operational bounds against tampering:
    - consecutive_failures >= consecutive_failed_cycles
    - retry_count_current_cycle >= max(0, attempt_number - 1)

    Returns
    -------
    tuple of (observable_features, payment_context, customer_history)
    """
    if not isinstance(raw_features, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="features must be a dictionary/object",
        )

    # 1. Leakage Firewall: Reject forbidden simulator/counterfactual fields
    for k in raw_features:
        for prefix in FORBIDDEN_PREFIXES:
            if k == prefix or k.startswith(prefix):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Forbidden simulator/counterfactual feature detected: {k!r}",
                )

    # 2. Strict Whitelisting: Reject extra / undeclared properties
    extra_keys = [k for k in raw_features if k not in ALLOWED_FEATURE_KEYS]
    if extra_keys:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Undeclared feature fields rejected: {extra_keys}",
        )

    # 3. Completeness: Ensure all 9 required model features are present and non-null
    missing = [col for col in OBSERVABLE_FEATURES if col not in raw_features or raw_features[col] is None]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required observable features: {missing}",
        )

    # 4. Strict Numeric & Type Validation
    # amount: float > 0
    amt_raw = raw_features["amount"]
    if not isinstance(amt_raw, (int, float)) or isinstance(amt_raw, bool):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="amount must be a numeric value")
    if math.isnan(amt_raw) or math.isinf(amt_raw) or amt_raw <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="amount must be a finite number > 0")
    amount = float(amt_raw)

    # attempt_number: int >= 1
    att_raw = raw_features["attempt_number"]
    if not isinstance(att_raw, int) or isinstance(att_raw, bool) or att_raw < 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="attempt_number must be an integer >= 1")
    attempt_number = int(att_raw)

    # dynamic_success_rate: float in [0.0, 1.0]
    dsr_raw = raw_features["dynamic_success_rate"]
    if not isinstance(dsr_raw, (int, float)) or isinstance(dsr_raw, bool) or math.isnan(dsr_raw) or math.isinf(dsr_raw) or not (0.0 <= dsr_raw <= 1.0):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="dynamic_success_rate must be a finite number in [0.0, 1.0]")
    dynamic_success_rate = float(dsr_raw)

    # cumulative_failures: int >= 0
    cf_raw = raw_features["cumulative_failures"]
    if not isinstance(cf_raw, int) or isinstance(cf_raw, bool) or cf_raw < 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="cumulative_failures must be an integer >= 0")
    cumulative_failures = int(cf_raw)

    # consecutive_failed_cycles: int >= 0
    cfc_raw = raw_features["consecutive_failed_cycles"]
    if not isinstance(cfc_raw, int) or isinstance(cfc_raw, bool) or cfc_raw < 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="consecutive_failed_cycles must be an integer >= 0")
    consecutive_failed_cycles = int(cfc_raw)

    # notification_engagement_score: float in [0.0, 1.0]
    nes_raw = raw_features["notification_engagement_score"]
    if not isinstance(nes_raw, (int, float)) or isinstance(nes_raw, bool) or math.isnan(nes_raw) or math.isinf(nes_raw) or not (0.0 <= nes_raw <= 1.0):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="notification_engagement_score must be a finite number in [0.0, 1.0]")
    notification_engagement_score = float(nes_raw)

    # contact_response_score: float in [0.0, 1.0]
    crs_raw = raw_features["contact_response_score"]
    if not isinstance(crs_raw, (int, float)) or isinstance(crs_raw, bool) or math.isnan(crs_raw) or math.isinf(crs_raw) or not (0.0 <= crs_raw <= 1.0):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="contact_response_score must be a finite number in [0.0, 1.0]")
    contact_response_score = float(crs_raw)

    # payment_method: valid categorical
    pm_raw = raw_features["payment_method"]
    if not isinstance(pm_raw, str) or len(pm_raw) > 255 or pm_raw not in VALID_PAYMENT_METHODS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"payment_method must be one of {sorted(VALID_PAYMENT_METHODS)}, got {pm_raw!r}",
        )
    payment_method = str(pm_raw)

    # failure_reason: valid categorical
    fr_raw = raw_features["failure_reason"]
    if not isinstance(fr_raw, str) or len(fr_raw) > 255 or fr_raw not in VALID_FAILURE_REASONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"failure_reason must be one of {sorted(VALID_FAILURE_REASONS)}, got {fr_raw!r}",
        )
    failure_reason = str(fr_raw)

    # 5. Optional Operational / Guardrail Overrides (with cold-start defaults & anti-tamper bounds)
    if "status" in raw_features and raw_features["status"] is not None:
        st_raw = str(raw_features["status"]).strip().lower()
        if st_raw not in {"failed", "pending", "success", "recovered"}:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid status: {raw_features['status']!r}")
        status_val = st_raw
    else:
        # Documented cold-start default: payment recovery engine evaluates failed payments
        status_val = "failed"

    if "consecutive_failures" in raw_features and raw_features["consecutive_failures"] is not None:
        cf_in = raw_features["consecutive_failures"]
        if not isinstance(cf_in, int) or isinstance(cf_in, bool) or cf_in < 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="consecutive_failures must be an integer >= 0")
        effective_cf = max(int(cf_in), consecutive_failed_cycles)
    else:
        effective_cf = consecutive_failed_cycles

    raw_rc = raw_features.get("retry_count_current_cycle", raw_features.get("retry_count"))
    if raw_rc is not None:
        if not isinstance(raw_rc, int) or isinstance(raw_rc, bool) or raw_rc < 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="retry_count_current_cycle must be an integer >= 0")
        effective_rc = max(int(raw_rc), max(0, attempt_number - 1))
    else:
        effective_rc = max(0, attempt_number - 1)

    if "lifetime_escalations" in raw_features and raw_features["lifetime_escalations"] is not None:
        le_in = raw_features["lifetime_escalations"]
        if not isinstance(le_in, int) or isinstance(le_in, bool) or le_in < 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="lifetime_escalations must be an integer >= 0")
        effective_le = int(le_in)
    else:
        # Documented cold-start default: 0 lifetime escalations
        effective_le = 0

    raw_intv = raw_features.get("interventions_last_7_days", raw_features.get("interventions_7d"))
    if raw_intv is not None:
        if not isinstance(raw_intv, int) or isinstance(raw_intv, bool) or raw_intv < 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="interventions_last_7_days must be an integer >= 0")
        effective_intv = int(raw_intv)
    else:
        # Documented cold-start default: 0 prior interventions
        effective_intv = 0

    # 6. Partition Construction
    observable_features = {
        "amount": amount,
        "attempt_number": attempt_number,
        "dynamic_success_rate": dynamic_success_rate,
        "cumulative_failures": cumulative_failures,
        "consecutive_failed_cycles": consecutive_failed_cycles,
        "notification_engagement_score": notification_engagement_score,
        "contact_response_score": contact_response_score,
        "payment_method": payment_method,
        "failure_reason": failure_reason,
    }

    payment_context = {
        "payment_id": payment_id,
        "amount": amount,
        "status": status_val,
        "attempt_number": attempt_number,
        "payment_method": payment_method,
        "failure_reason": failure_reason,
        "consecutive_failures": effective_cf,
        "retry_count_current_cycle": effective_rc,
    }

    raw_days = raw_features.get("days_active", 0)
    days_active = int(raw_days) if isinstance(raw_days, int) and not isinstance(raw_days, bool) and raw_days >= 0 else 0

    customer_history = {
        "customer_id": str(raw_features.get("customer_id", ""))[:255],
        "days_active": days_active,
        "dynamic_success_rate": dynamic_success_rate,
        "cumulative_failures": cumulative_failures,
        "consecutive_failed_cycles": consecutive_failed_cycles,
        "notification_engagement_score": notification_engagement_score,
        "contact_response_score": contact_response_score,
        "lifetime_escalations": effective_le,
        "interventions_last_7_days": effective_intv,
    }

    return observable_features, payment_context, customer_history



# ── Response DTO Formatting ──────────────────────────────────────────────────

def format_response_dto(
    payment_id: str,
    final_action: str,
    arm_net_values: dict[str, float],
    llm_decision: dict[str, Any],
    guardrail_result: dict[str, Any],
    request_id: str,
    error: Optional[str] = None,
) -> dict[str, Any]:
    """Format standardized output response DTO."""
    if error:
        short_code = "PAYMENT_NOT_FOUND" if "not found" in error.lower() else "INVALID_CONTEXT"
        return {
            "payment_id": payment_id,
            "model_decision": "N/A — error path",
            "llm_decision": "N/A — error path",
            "guardrail_overridden": False,
            "guardrail_reason": f"Bypassed due to error: {short_code}",
            "final_action": "WAIT",
            "confidence": 0.0,
            "risk_level": "none",
            "reasoning": f"Error: {short_code}",
            "decision_source": "error_path",
            "request_id": request_id,
        }

    # Model recommendation: argmax over net values
    if arm_net_values:
        model_decision = max(arm_net_values, key=arm_net_values.get)
    else:
        model_decision = "WAIT"

    llm_action = llm_decision.get("decision", "WAIT")
    guardrail_overridden = bool(guardrail_result.get("overridden", False))
    guardrail_reason = guardrail_result.get("reason", "")
    confidence = float(llm_decision.get("confidence", 1.0))
    risk_level = str(llm_decision.get("risk_level", "medium"))
    reasoning = str(llm_decision.get("reasoning", ""))
    decision_source = str(llm_decision.get("decision_source", "llm"))

    return {
        "payment_id": payment_id,
        "model_decision": model_decision,
        "llm_decision": llm_action,
        "guardrail_overridden": guardrail_overridden,
        "guardrail_reason": guardrail_reason,
        "final_action": final_action,
        "confidence": confidence,
        "risk_level": risk_level,
        "reasoning": reasoning,
        "decision_source": decision_source,
        "request_id": request_id,
    }


# ── Concurrency Primitives ───────────────────────────────────────────────────

async def get_payment_lock(app_instance: FastAPI, payment_id: str) -> asyncio.Lock:
    """Get or create process-local asyncio.Lock for payment_id safely [FIX-1]."""
    async with app_instance.state.locks_mutex:
        if payment_id not in app_instance.state.payment_locks:
            app_instance.state.payment_locks[payment_id] = asyncio.Lock()
        return app_instance.state.payment_locks[payment_id]


# ── Lifespan Handler ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    # 0. Startup Configuration Validation (Day 9E Centralized Contract)
    # Validates PERSISTENCE_BACKEND, DATABASE_URL, pool settings, Azure Foundry credentials
    # at startup before serving traffic. Fails fast with secret redaction.
    config: DecisionEngineConfig = load_and_validate_config()
    app_instance.state.config = config
    app_instance.state.persistence_backend = config.persistence_backend
    app_instance.state.db_pool = None
    app_instance.state.db = None
    app_instance.state.repository = None

    # Day 9F: Track migration readiness for health gating
    app_instance.state.migrations_applied = False

    if config.persistence_backend == "postgres":
        logger.info("Initializing PostgreSQL AsyncConnectionPool (PERSISTENCE_BACKEND=postgres)...")
        app_instance.state.db_pool = await create_postgres_pool(
            database_url=config.database_url,
            min_size=config.db_pool_min,
            max_size=config.db_pool_max,
            timeout_ms=config.db_connect_timeout_ms,
        )
        app_instance.state.repository = PostgresDecisionRepository(pool=app_instance.state.db_pool)

        # Day 9F: Run forward-only migrations BEFORE accepting traffic.
        # Synchronous psycopg connection — migrations are DDL and must complete
        # before the async pool serves application queries.
        logger.info("Running database migrations...")
        try:
            applied = run_migrations(database_url=config.database_url)
            if applied:
                logger.info(f"Applied {len(applied)} migration(s): {applied}")
            else:
                logger.info("No new migrations to apply. Schema is up to date.")
            app_instance.state.migrations_applied = True
        except MigrationError as exc:
            logger.error(f"Migration failed: {exc}. Service will not become ready.")
            # Do NOT set migrations_applied = True; health will report not-ready.
        except Exception as exc:
            logger.error(f"Unexpected migration error: {type(exc).__name__}. Service will not become ready.")

    elif config.persistence_backend == "sqlite":
        # SQLite persistence backend (Day 7 default / backward compatibility)
        db_path = str(DEFAULT_AUDIT_DB_PATH)
        logger.info(f"Opening SQLite repository at {db_path}...")
        db, repo = await open_sqlite_repository(db_path)
        app_instance.state.db = db
        app_instance.state.repository = repo
        # SQLite migrations are handled inline by aiosqlite CREATE TABLE IF NOT EXISTS
        app_instance.state.migrations_applied = True
    else:
        raise ConfigValidationError(
            f"Unsupported PERSISTENCE_BACKEND: {config.persistence_backend!r}. Must be 'postgres' or 'sqlite'."
        )

    # 1. Initialize CausalUpliftPolicy exactly once
    logger.info("Initializing CausalUpliftPolicy...")
    try:
        app_instance.state.policy = CausalUpliftPolicy()
    except Exception as exc:
        logger.error(f"Failed to initialize CausalUpliftPolicy: {exc}")
        app_instance.state.policy = None

    # Warm canonical dataset in-memory cache once at startup [warm-process design]
    try:
        logger.info("Warming canonical dataset cache...")
        app_instance.state.dataset = _get_dataset()
    except Exception as exc:
        logger.warning(f"Could not warm dataset cache: {exc}")
        app_instance.state.dataset = None

    # 3. Create app.state.llm_semaphore = asyncio.Semaphore(5) [FIX-2]
    app_instance.state.llm_semaphore = asyncio.Semaphore(5)

    # 4. Create app.state.payment_locks with lock mutex [FIX-1]
    app_instance.state.payment_locks = {}
    app_instance.state.locks_mutex = asyncio.Lock()

    # 5. Compile canonical LangGraph exactly once using create_recovery_graph [FIX-4]
    if app_instance.state.policy is not None:
        logger.info("Compiling canonical service LangGraph (use_async=True)...")
        app_instance.state.graph = create_recovery_graph(
            policy=app_instance.state.policy,
            use_async=True,
        )
    else:
        app_instance.state.graph = None

    emit_log(
        logger,
        logging.INFO,
        event="service_startup",
        request_id="system-startup",
        persistence_backend=config.persistence_backend,
        status="ready",
    )

    yield

    # 6. On shutdown (lifespan exit), close database connections cleanly
    if hasattr(app_instance.state, "db") and app_instance.state.db is not None:
        logger.info("Closing SQLite connection...")
        await app_instance.state.db.close()
        app_instance.state.db = None

    if getattr(app_instance.state, "db_pool", None) is not None:
        logger.info("Closing PostgreSQL connection pool...")
        await close_postgres_pool(app_instance.state.db_pool)

    app_instance.state.repository = None

    emit_log(
        logger,
        logging.INFO,
        event="service_shutdown",
        request_id="system-shutdown",
        persistence_backend=getattr(app_instance.state, "persistence_backend", "unknown"),
        status="clean_shutdown",
    )


# ── FastAPI App Instance ─────────────────────────────────────────────────────

app = FastAPI(
    title="Recovery Decision Engine API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Exception Handlers ───────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    req_id = request.headers.get("x-request-id")
    headers = {"x-request-id": req_id} if req_id else None
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": {"code": "VALIDATION_ERROR", "message": "Invalid request payload"}},
        headers=headers,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    code = "VALIDATION_ERROR" if exc.status_code == 400 else "INTERNAL_ERROR"
    msg = exc.detail if isinstance(exc.detail, str) else "An error occurred"
    req_id = request.headers.get("x-request-id")
    headers = {"x-request-id": req_id} if req_id else None
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": code, "message": msg}},
        headers=headers,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    req_id = request.headers.get("x-request-id")
    emit_log(
        logger,
        logging.ERROR,
        "unhandled_exception",
        req_id or "unknown",
        error=str(exc),
        error_type=type(exc).__name__,
    )
    headers = {"x-request-id": req_id} if req_id else None
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"code": "INTERNAL_ERROR", "message": "Internal server error"}},
        headers=headers,
    )


# ── Health Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
@app.get("/ready")
async def health_check():
    """
    Return structured health status including database dependency probe.

    PostgreSQL mode: probes pool via SELECT 1 with ~2s timeout.
    SQLite mode: reports decision_engine readiness only (no database dependency).
    """
    health_start = time.monotonic()
    policy_ready = getattr(app.state, "policy", None) is not None
    graph_ready = getattr(app.state, "graph", None) is not None
    repo_ready = (
        getattr(app.state, "repository", None) is not None
        or getattr(app.state, "db", None) is not None
        or getattr(app.state, "db_pool", None) is not None
    )
    migrations_ok = getattr(app.state, "migrations_applied", False)

    engine_ok = policy_ready and graph_ready and repo_ready and migrations_ok

    persistence_backend = getattr(app.state, "persistence_backend", "sqlite")

    if persistence_backend == "postgres":
        pool = getattr(app.state, "db_pool", None)
        db_status = await check_pool_health(pool)
    else:
        # SQLite mode: no PostgreSQL dependency to probe
        db_status = "ok"

    duration_ms = round((time.monotonic() - health_start) * 1000, 2)
    overall_status = "ok" if db_status == "ok" else "degraded"

    emit_log(
        logger,
        logging.INFO,
        event="health_check",
        request_id="system-health",
        status=overall_status,
        database=db_status,
        duration_ms=duration_ms,
    )

    if not engine_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine components not fully initialized",
        )

    return {
        "status": overall_status,
        "dependencies": {
            "database": db_status,
            "decision_engine": "ok",
        },
    }


# ── Decision Evaluation Endpoint ─────────────────────────────────────────────

@app.post("/evaluate")
async def evaluate_decision(req: EvaluateRequest, request: Request, response: Response):
    """
    Evaluate recovery decision with concurrency-safe idempotency and deadline-aware LLM execution.
    """
    request_start = time.monotonic()
    llm_deadline = request_start + 6.0

    # 1. Validate payment_id
    if not req.payment_id or not isinstance(req.payment_id, str) or not req.payment_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="payment_id is required and cannot be empty",
        )
    payment_id = req.payment_id.strip()

    # Correlation ID:
    # 1. Accept X-Request-Id header.
    # 2. Preserve it if present.
    # 3. Otherwise use request_id from the JSON body if present.
    # 4. Otherwise generate one UUID.
    header_req_id = request.headers.get("x-request-id")
    if header_req_id and header_req_id.strip():
        request_id = header_req_id.strip()
    elif req.request_id and req.request_id.strip():
        request_id = req.request_id.strip()
    else:
        request_id = str(uuid.uuid4())

    # Return X-Request-Id in response header
    response.headers["x-request-id"] = request_id

    # Event: request_received & evaluate_received
    emit_log(
        logger,
        logging.INFO,
        "request_received",
        request_id,
        payment_id=payment_id,
        force_recompute=req.force_recompute,
        request_start=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    emit_log(
        logger,
        logging.INFO,
        "evaluate_received",
        request_id,
        payment_id=payment_id,
        force_recompute=req.force_recompute,
    )

    # 2. Acquire this payment's asyncio.Lock [FIX-1]
    lock = await get_payment_lock(app, payment_id)
    if lock.locked():
        # Event B: lock_blocked_duplicate (only emitted when request actually has to wait for a lock)
        emit_log(
            logger,
            logging.INFO,
            "lock_blocked_duplicate",
            request_id,
            payment_id=payment_id,
        )

    lock_start = time.monotonic()
    async with lock:
        lock_wait_ms = round((time.monotonic() - lock_start) * 1000, 2)
        # Event C: lock_acquired
        emit_log(
            logger,
            logging.INFO,
            "lock_acquired",
            request_id,
            payment_id=payment_id,
            wait_ms=lock_wait_ms,
        )

        # FIX 2: Obtain CURRENT payment/customer state before cache lookup
        pre_populated_state = None
        if req.features is not None:
            # Caller-supplied feature evaluation (new/unseen payment path)
            obs_feat, pmt_ctx, cust_hist = validate_and_parse_features(
                payment_id=payment_id,
                raw_features=req.features,
            )
            current_fingerprint = compute_state_fingerprint(
                payment_id=payment_id,
                status=pmt_ctx["status"],
                attempt_number=pmt_ctx["attempt_number"],
                consecutive_failures=pmt_ctx["consecutive_failures"],
                retry_count=pmt_ctx["retry_count_current_cycle"],
                interventions_7d=cust_hist["interventions_last_7_days"],
                decision_features=obs_feat,
            )
            pre_populated_state = {
                "observable_features": obs_feat,
                "payment_context": pmt_ctx,
                "customer_history": cust_hist,
            }
        else:
            # Standard CSV lookup path (known payment path)
            dataset = getattr(app.state, "dataset", None)
            current_state = get_payment_state(payment_id, dataset=dataset)
            current_fingerprint = None
            if current_state is not None:
                current_fingerprint = compute_state_fingerprint(
                    payment_id=current_state["payment_id"],
                    status=current_state["status"],
                    attempt_number=current_state["attempt_number"],
                    consecutive_failures=current_state["consecutive_failures"],
                    retry_count=current_state["retry_count"],
                    interventions_7d=current_state["interventions_7d"],
                )

        # 3. Inside the lock:
        repository: Optional[DecisionRepository] = getattr(app.state, "repository", None)
        active_db = getattr(app.state, "db", None)
        if active_db is not None:
            if not isinstance(repository, SqliteDecisionRepository) or repository.db is not active_db:
                repository = SqliteDecisionRepository(active_db)

        # a. Check repository cache if force_recompute is False and state fingerprint matches
        cache_lookup_start = time.monotonic()
        if not req.force_recompute and current_fingerprint is not None and repository is not None:
            persisted = await repository.get_current_decision(payment_id, request_id=request_id)
            if (
                persisted is not None
                and persisted.get("state_fingerprint") is not None
                and persisted.get("state_fingerprint") == current_fingerprint
            ):
                cache_duration_ms = round((time.monotonic() - cache_lookup_start) * 1000, 2)
                # Structured DB observability: db_cache_hit & cache_hit
                emit_log(
                    logger,
                    logging.INFO,
                    "db_cache_hit",
                    request_id,
                    payment_id=payment_id,
                    duration_ms=cache_duration_ms,
                )
                # Event D: cache_hit
                emit_log(
                    logger,
                    logging.INFO,
                    "cache_hit",
                    request_id,
                    payment_id=payment_id,
                    duration_ms=cache_duration_ms,
                )
                net_vals = persisted.get("raw_arm_net_values")
                if isinstance(net_vals, str):
                    try:
                        net_vals = json.loads(net_vals)
                    except Exception:
                        net_vals = {}
                elif not isinstance(net_vals, dict):
                    net_vals = {}

                if net_vals:
                    model_decision = max(net_vals, key=net_vals.get)
                else:
                    model_decision = "WAIT"

                verdict = str(persisted.get("guardrail_verdict") or "")
                overridden = verdict.lower() == "overridden"

                cached_dto = {
                    "payment_id": payment_id,
                    "model_decision": model_decision,
                    "llm_decision": str(persisted.get("llm_proposed_decision") or "WAIT"),
                    "guardrail_overridden": overridden,
                    "guardrail_reason": str(persisted.get("guardrail_reason") or ""),
                    "final_action": str(persisted.get("final_action") or "WAIT"),
                    "confidence": float(persisted.get("llm_confidence") or 0.0),
                    "risk_level": str(persisted.get("llm_risk_level") or "none"),
                    "reasoning": str(persisted.get("llm_reasoning") or ""),
                    "decision_source": "cache",
                    "request_id": request_id,
                }
                duration_ms = round((time.monotonic() - request_start) * 1000, 2)
                # Event: decision_completed & evaluate_completed on cache-hit
                emit_log(
                    logger,
                    logging.INFO,
                    "decision_completed",
                    request_id,
                    payment_id=payment_id,
                    duration_ms=duration_ms,
                    lock_wait_duration_ms=lock_wait_ms,
                    final_action=cached_dto["final_action"],
                    decision_source="cache",
                )
                emit_log(
                    logger,
                    logging.INFO,
                    "evaluate_completed",
                    request_id,
                    payment_id=payment_id,
                    duration_ms=duration_ms,
                    final_action=cached_dto["final_action"],
                )
                return cached_dto

        # b. Cache miss or force_recompute=True: invoke canonical graph
        cache_miss_duration_ms = round((time.monotonic() - cache_lookup_start) * 1000, 2)
        # Structured DB observability: db_cache_miss & cache_miss
        emit_log(
            logger,
            logging.INFO,
            "db_cache_miss",
            request_id,
            payment_id=payment_id,
            duration_ms=cache_miss_duration_ms,
        )
        # Event E: cache_miss
        emit_log(
            logger,
            logging.INFO,
            "cache_miss",
            request_id,
            payment_id=payment_id,
            duration_ms=cache_miss_duration_ms,
        )
        
        # State contains strictly business domain fields (Issue 1)
        initial_state: RecoveryState = {
            "payment_id": payment_id,
            "audit_trail": [],
            "state_fingerprint": current_fingerprint,
        }
        if pre_populated_state is not None:
            initial_state.update(pre_populated_state)

        # Runtime context passed via standard RunnableConfig configurable dict (Issue 1 & 2)
        config = {
            "configurable": {
                "llm_deadline": llm_deadline,
                "llm_semaphore": app.state.llm_semaphore,
                "request_id": request_id,
                "db": getattr(app.state, "db", None) if repository is None else None,
                "repository": repository,
                "dataset": getattr(app.state, "dataset", None),
            }
        }

        # Event F: graph_started
        emit_log(
            logger,
            logging.INFO,
            "graph_started",
            request_id,
            payment_id=payment_id,
        )

        # Canonical async graph execution (timed as evaluation_duration_ms)
        eval_start = time.monotonic()
        final_state: RecoveryState = await app.state.graph.ainvoke(initial_state, config=config)
        evaluation_duration_ms = round((time.monotonic() - eval_start) * 1000, 2)

        dto = format_response_dto(
            payment_id=payment_id,
            final_action=final_state.get("final_action", "WAIT"),
            arm_net_values=final_state.get("arm_net_values", {}),
            llm_decision=final_state.get("llm_decision", {}),
            guardrail_result=final_state.get("guardrail_result", {}),
            request_id=request_id,
            error=final_state.get("error"),
        )

        # Persist through repository.save_decision_with_event(...)
        if repository is not None:
            llm_dec = final_state.get("llm_decision", {})
            g_res = final_state.get("guardrail_result", {})
            raw_nets = final_state.get("arm_net_values", {})
            model_dec = max(raw_nets, key=raw_nets.get) if raw_nets else "WAIT"

            await repository.save_decision_with_event(
                payment_id=payment_id,
                decision_id=f"dec_{payment_id}",
                request_id=request_id,
                raw_arm_probabilities=final_state.get("arm_probabilities"),
                raw_arm_net_values=final_state.get("arm_net_values"),
                llm_proposed_decision=llm_dec.get("decision", "WAIT"),
                llm_confidence=float(llm_dec.get("confidence", 1.0)) if llm_dec else 0.0,
                llm_reasoning=llm_dec.get("reasoning", ""),
                llm_risk_level=llm_dec.get("risk_level", "medium"),
                expected_incremental_value=float(llm_dec.get("expected_incremental_value", 0.0)) if llm_dec else 0.0,
                guardrail_verdict=g_res.get("status", "passed") if not final_state.get("error") else "N/A — error path",
                guardrail_reason=g_res.get("reason", "") if not final_state.get("error") else "Bypassed due to error",
                guardrail_overridden=bool(g_res.get("overridden", False)),
                final_action=dto["final_action"],
                decision_source=dto["decision_source"],
                model_decision=model_dec,
                error=final_state.get("error"),
                evaluated_at=datetime.datetime.now(datetime.timezone.utc),
                state_fingerprint=current_fingerprint,
                state=final_state,
            )
            emit_log(
                logger,
                logging.INFO,
                "audit_persisted",
                request_id,
                payment_id=payment_id,
                decision_id=f"dec_{payment_id}",
            )

        # Event G: final_action_selected
        emit_log(
            logger,
            logging.INFO,
            "final_action_selected",
            request_id,
            payment_id=payment_id,
            final_action=dto.get("final_action", "WAIT"),
        )

        # Event: decision_completed & evaluate_completed
        duration_ms = round((time.monotonic() - request_start) * 1000, 2)
        llm_duration_ms = (
            final_state.get("llm_decision", {}).get("llm_duration_ms")
            if isinstance(final_state.get("llm_decision"), dict)
            else None
        ) or final_state.get("llm_duration_ms")

        decision_kwargs = {
            "payment_id": payment_id,
            "duration_ms": duration_ms,
            "lock_wait_duration_ms": lock_wait_ms,
            "evaluation_duration_ms": evaluation_duration_ms,
            "final_action": dto.get("final_action", "WAIT"),
            "decision_source": dto.get("decision_source", "model"),
        }
        if llm_duration_ms is not None:
            decision_kwargs["llm_duration_ms"] = float(llm_duration_ms)

        emit_log(
            logger,
            logging.INFO,
            "decision_completed",
            request_id,
            **decision_kwargs,
        )
        emit_log(
            logger,
            logging.INFO,
            "evaluate_completed",
            request_id,
            payment_id=payment_id,
            duration_ms=duration_ms,
            final_action=dto.get("final_action", "WAIT"),
        )
        return dto
