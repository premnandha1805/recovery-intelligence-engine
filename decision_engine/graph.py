"""
decision_engine/graph.py
========================
LangGraph state graph for Day 6 Recovery Intelligence Engine.

Complete Workflow Architecture:
    START
      │
      ▼
    context
      │
      ▼
    estimation
      │
      ├──────────────────────────────┐
      │ (error present)              │ (no error)
      ▼                              ▼
    error_fallback                reasoning
      │                              │
      │                              ▼
      │                          guardrail
      │                              │
      └──────────────┬───────────────┘
                     │
                     ▼
                 execution
                     │
                     ▼
                    END
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
from typing import Any, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from decision_engine.state import RecoveryState
from decision_engine.context_node import context_node
from decision_engine.estimation_node import estimation_node
from decision_engine.reasoning_node import reasoning_node, async_reasoning_node
from decision_engine.guardrail_node import guardrail_node
from decision_engine.execution_node import execution_node
from decision_engine.audit import async_save_decision_audit
from decision_engine.structured_logger import emit_log

logger = logging.getLogger("decision_engine.graph")



def error_fallback_node(state: RecoveryState) -> dict[str, Any]:
    """
    Deterministic error handler for malformed or missing payment requests.
    Sets safe default final_action='WAIT' and records error bypass events.
    """
    err = state.get("error", "Unknown error encountered")
    return {
        "final_action": "WAIT",
        "llm_decision": {
            "decision": "N/A — error path",
            "confidence": 0.0,
            "reasoning": f"Bypassed reasoning due to error: {err}",
            "risk_level": "none",
            "decision_source": "error_path",
            "expected_incremental_value": 0.0,
        },
        "guardrail_result": {
            "status": "N/A — error path",
            "proposed_action": "N/A — error path",
            "final_action": "WAIT",
            "overridden": False,
            "reason": "Bypassed guardrails due to error",
        },
        "audit_trail": [
            {
                "node": "error_fallback",
                "status": "error_routed",
                "final_action": "WAIT",
                "error": err,
            }
        ],
    }


def route_after_estimation(state: RecoveryState) -> Literal["error_fallback", "reasoning"]:
    """
    Conditional router: routes to error_fallback if error is present, otherwise reasoning.
    """
    if state.get("error"):
        return "error_fallback"
    return "reasoning"


def create_recovery_graph(
    policy: Any = None,
    llm: Any = None,
    dataset: Any = None,
    db_path: pathlib.Path | str | None = None,
    use_async: bool = False,
) -> Any:
    """
    Build and compile the canonical Recovery Intelligence Engine workflow.

    Parameters
    ----------
    policy : CausalUpliftPolicy, optional
        Injected policy instance (useful for unit testing).
    llm : BaseChatModel, optional
        Injected LLM instance (useful for unit testing with mocks).
    dataset : pd.DataFrame, optional
        Injected dataset for context retrieval testing.
    db_path : pathlib.Path | str, optional
        Custom SQLite audit database path.
    use_async : bool, default False
        If True, compiles with native asynchronous execution, thread-offloaded
        inference, and RunnableConfig runtime parameter support.

    Returns
    -------
    CompiledStateGraph
        The compiled executable LangGraph instance.
    """
    builder = StateGraph(RecoveryState)

    if not use_async:
        # Day 6 synchronous execution path
        def run_context(state: RecoveryState, config: RunnableConfig = None) -> dict[str, Any]:
            cfg = config.get("configurable", {}) if config else {}
            ds = cfg.get("dataset", dataset)
            res = context_node(state, dataset=ds)
            req_id = cfg.get("request_id")
            if req_id and "audit_trail" in res:
                for ev in res["audit_trail"]:
                    ev["request_id"] = req_id
            return res

        def run_estimation(state: RecoveryState, config: RunnableConfig = None) -> dict[str, Any]:
            res = estimation_node(state, policy=policy)
            req_id = (config.get("configurable", {}) if config else {}).get("request_id")
            if req_id and "audit_trail" in res:
                for ev in res["audit_trail"]:
                    ev["request_id"] = req_id
            return res

        def run_reasoning(state: RecoveryState, config: RunnableConfig = None) -> dict[str, Any]:
            res = reasoning_node(state, llm=llm)
            req_id = (config.get("configurable", {}) if config else {}).get("request_id")
            if req_id and "audit_trail" in res:
                for ev in res["audit_trail"]:
                    ev["request_id"] = req_id
            return res

        def run_guardrail(state: RecoveryState, config: RunnableConfig = None) -> dict[str, Any]:
            res = guardrail_node(state)
            req_id = (config.get("configurable", {}) if config else {}).get("request_id")
            if req_id and "audit_trail" in res:
                for ev in res["audit_trail"]:
                    ev["request_id"] = req_id
            return res

        def run_error_fallback(state: RecoveryState, config: RunnableConfig = None) -> dict[str, Any]:
            res = error_fallback_node(state)
            req_id = (config.get("configurable", {}) if config else {}).get("request_id")
            if req_id and "audit_trail" in res:
                for ev in res["audit_trail"]:
                    ev["request_id"] = req_id
            return res

        def run_execution(state: RecoveryState, config: RunnableConfig = None) -> dict[str, Any]:
            req_id = (config.get("configurable", {}) if config else {}).get("request_id")
            res = execution_node(state, db_path=db_path, request_id=req_id)
            if req_id and "audit_trail" in res:
                for ev in res["audit_trail"]:
                    ev["request_id"] = req_id
            payment_id = state.get("payment_id", "")
            decision_id = f"dec_{payment_id}"
            emit_log(
                logger,
                logging.INFO,
                "audit_persisted",
                req_id or "",
                payment_id=payment_id,
                decision_id=decision_id,
            )
            return res


    else:
        # Day 7 asynchronous execution path with RunnableConfig runtime context
        async def run_context(state: RecoveryState, config: RunnableConfig) -> dict[str, Any]:
            cfg = config.get("configurable", {}) if config else {}
            ds = cfg.get("dataset", dataset)
            res = context_node(state, dataset=ds)
            req_id = cfg.get("request_id")
            if req_id and "audit_trail" in res:
                for ev in res["audit_trail"]:
                    ev["request_id"] = req_id
            return res

        async def run_estimation(state: RecoveryState, config: RunnableConfig) -> dict[str, Any]:
            res = await asyncio.to_thread(estimation_node, state, policy=policy)
            req_id = (config.get("configurable", {}) if config else {}).get("request_id")
            if req_id and "audit_trail" in res:
                for ev in res["audit_trail"]:
                    ev["request_id"] = req_id
            return res

        async def run_reasoning(state: RecoveryState, config: RunnableConfig) -> dict[str, Any]:
            cfg = config.get("configurable", {}) if config else {}
            deadline = cfg.get("llm_deadline")
            sem = cfg.get("llm_semaphore")
            return await async_reasoning_node(state, llm=llm, deadline=deadline, semaphore=sem, config=config)

        async def run_guardrail(state: RecoveryState, config: RunnableConfig) -> dict[str, Any]:
            res = guardrail_node(state)
            req_id = (config.get("configurable", {}) if config else {}).get("request_id")
            if req_id and "audit_trail" in res:
                for ev in res["audit_trail"]:
                    ev["request_id"] = req_id
            return res

        async def run_error_fallback(state: RecoveryState, config: RunnableConfig) -> dict[str, Any]:
            res = error_fallback_node(state)
            req_id = (config.get("configurable", {}) if config else {}).get("request_id")
            if req_id and "audit_trail" in res:
                for ev in res["audit_trail"]:
                    ev["request_id"] = req_id
            return res

        async def run_execution(state: RecoveryState, config: RunnableConfig) -> dict[str, Any]:
            cfg = config.get("configurable", {}) if config else {}
            repository = cfg.get("repository")
            db = cfg.get("db")
            req_id = cfg.get("request_id")
            if repository is not None:
                # Persistence is handled by service layer using repository.save_decision_with_event
                pass
            elif db is not None:
                await async_save_decision_audit(state, db, request_id=req_id)
            else:
                await asyncio.to_thread(execution_node, state, db_path=db_path, request_id=req_id)

            final_action = state.get("final_action", "WAIT")
            error = state.get("error")
            payment_id = state.get("payment_id", "")
            decision_id = f"dec_{payment_id}"
            emit_log(
                logger,
                logging.INFO,
                "audit_persisted",
                req_id or "",
                payment_id=payment_id,
                decision_id=decision_id,
            )

            status_str = "error_halted" if error else "executed"
            audit_event = {
                "node": "execution_node",
                "status": status_str,
                "final_action": final_action,
                "payment_id": state.get("payment_id"),
            }
            if req_id:
                audit_event["request_id"] = req_id
            if error:
                audit_event["error"] = error
            return {
                "final_action": final_action,
                "audit_trail": [audit_event],
            }


    builder.add_node("context", run_context)
    builder.add_node("estimation", run_estimation)
    builder.add_node("reasoning", run_reasoning)
    builder.add_node("guardrail", run_guardrail)
    builder.add_node("error_fallback", run_error_fallback)
    builder.add_node("execution", run_execution)

    builder.add_edge(START, "context")
    builder.add_edge("context", "estimation")
    builder.add_conditional_edges(
        "estimation",
        route_after_estimation,
        {
            "error_fallback": "error_fallback",
            "reasoning": "reasoning",
        },
    )
    builder.add_edge("error_fallback", "execution")
    builder.add_edge("reasoning", "guardrail")
    builder.add_edge("guardrail", "execution")
    builder.add_edge("execution", END)

    return builder.compile()
