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

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import pathlib
import time
from typing import Any, Optional
import uuid

import aiosqlite
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from decision_engine.state import RecoveryState
from decision_engine.graph import create_recovery_graph
from decision_engine.audit import (
    CREATE_TABLE_SQL,
    DEFAULT_AUDIT_DB_PATH,
    async_save_decision_audit,
)
from ml.decision import CausalUpliftPolicy
from models.schemas import Action
from decision_engine.structured_logger import emit_log

# Configure structured logging
logger = logging.getLogger("decision_engine.service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── Request / Response Schemas ───────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    payment_id: str
    request_id: Optional[str] = None
    force_recompute: bool = False


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
    # 1. Initialize CausalUpliftPolicy exactly once
    logger.info("Initializing CausalUpliftPolicy...")
    try:
        app_instance.state.policy = CausalUpliftPolicy()
    except Exception as exc:
        logger.error(f"Failed to initialize CausalUpliftPolicy: {exc}")
        app_instance.state.policy = None

    # 2. Open ONE aiosqlite write connection to decision_engine/audit.db [FIX-5]
    db_path = str(DEFAULT_AUDIT_DB_PATH)
    pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Opening aiosqlite write connection to {db_path}...")
    db = await aiosqlite.connect(db_path)
    await db.execute("PRAGMA journal_mode=WAL;")
    await db.execute("PRAGMA synchronous=NORMAL;")
    await db.execute("PRAGMA busy_timeout=5000;")
    await db.execute(CREATE_TABLE_SQL)
    # Ensure additive migration for request_id column if table pre-existed
    async with db.execute("PRAGMA table_info(decision_audit)") as cur:
        cols = [c[1] for c in await cur.fetchall()]
        if "request_id" not in cols:
            await db.execute("ALTER TABLE decision_audit ADD COLUMN request_id TEXT;")
    await db.commit()
    app_instance.state.db = db

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

    yield

    # 6. On shutdown (lifespan exit), close aiosqlite connection cleanly [FIX-8]
    logger.info("Closing aiosqlite connection...")
    if hasattr(app_instance.state, "db") and app_instance.state.db is not None:
        await app_instance.state.db.close()


# ── FastAPI App Instance ─────────────────────────────────────────────────────

app = FastAPI(
    title="Recovery Decision Engine API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Exception Handlers ───────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": {"code": "VALIDATION_ERROR", "message": "Invalid request payload"}},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    code = "VALIDATION_ERROR" if exc.status_code == 400 else "INTERNAL_ERROR"
    msg = exc.detail if isinstance(exc.detail, str) else "An error occurred"
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": code, "message": msg}},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"code": "INTERNAL_ERROR", "message": "Internal server error"}},
    )


# ── Health Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
@app.get("/ready")
async def health_check():
    """
    Return status ok ONLY if policy and compiled graph initialized successfully.
    """
    policy_ready = getattr(app.state, "policy", None) is not None
    graph_ready = getattr(app.state, "graph", None) is not None

    if not policy_ready or not graph_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine components not fully initialized",
        )

    return {"status": "ok"}


# ── Decision Evaluation Endpoint ─────────────────────────────────────────────

@app.post("/evaluate")
async def evaluate_decision(req: EvaluateRequest, request: Request, response: Response):
    """
    Evaluate recovery decision with concurrency-safe idempotency and deadline-aware LLM execution.
    """
    request_start = time.monotonic()
    llm_deadline = request_start + 5.0

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

    # Event A: evaluate_received
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

        # 3. Inside the lock:
        # a. Check audit.db for existing decision if force_recompute is False
        if not req.force_recompute:
            async with app.state.db.execute(
                "SELECT * FROM decision_audit WHERE payment_id = ?",
                (payment_id,),
            ) as cursor:
                cursor.row_factory = aiosqlite.Row
                row = await cursor.fetchone()

            if row is not None:
                # Event D: cache_hit
                emit_log(
                    logger,
                    logging.INFO,
                    "cache_hit",
                    request_id,
                    payment_id=payment_id,
                )
                try:
                    net_vals = json.loads(row["raw_arm_net_values"] or "{}")
                except Exception:
                    net_vals = {}

                if net_vals:
                    model_decision = max(net_vals, key=net_vals.get)
                else:
                    model_decision = "WAIT"

                verdict = str(row["guardrail_verdict"] or "")
                overridden = verdict.lower() == "overridden"

                cached_dto = {
                    "payment_id": payment_id,
                    "model_decision": model_decision,
                    "llm_decision": str(row["llm_proposed_decision"] or "WAIT"),
                    "guardrail_overridden": overridden,
                    "guardrail_reason": str(row["guardrail_reason"] or ""),
                    "final_action": str(row["final_action"] or "WAIT"),
                    "confidence": float(row["llm_confidence"] or 0.0),
                    "risk_level": str(row["llm_risk_level"] or "none"),
                    "reasoning": str(row["llm_reasoning"] or ""),
                    "decision_source": str(row["decision_source"] or "cache"),
                    "request_id": request_id,
                }
                duration_ms = round((time.monotonic() - request_start) * 1000, 2)
                # Event I: evaluate_completed on cache-hit
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
        # Event E: cache_miss
        emit_log(
            logger,
            logging.INFO,
            "cache_miss",
            request_id,
            payment_id=payment_id,
        )
        
        # State contains strictly business domain fields (Issue 1)
        initial_state: RecoveryState = {
            "payment_id": payment_id,
            "audit_trail": [],
        }

        # Runtime context passed via standard RunnableConfig configurable dict (Issue 1 & 2)
        config = {
            "configurable": {
                "llm_deadline": llm_deadline,
                "llm_semaphore": app.state.llm_semaphore,
                "request_id": request_id,
                "db": app.state.db,
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

        # Canonical async graph execution (persistence performed in execution node)
        final_state: RecoveryState = await app.state.graph.ainvoke(initial_state, config=config)

        dto = format_response_dto(
            payment_id=payment_id,
            final_action=final_state.get("final_action", "WAIT"),
            arm_net_values=final_state.get("arm_net_values", {}),
            llm_decision=final_state.get("llm_decision", {}),
            guardrail_result=final_state.get("guardrail_result", {}),
            request_id=request_id,
            error=final_state.get("error"),
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

        # Event I: evaluate_completed
        duration_ms = round((time.monotonic() - request_start) * 1000, 2)
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
