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

import pathlib
from typing import Any, Literal
from langgraph.graph import StateGraph, START, END

from decision_engine.state import RecoveryState
from decision_engine.context_node import context_node
from decision_engine.estimation_node import estimation_node
from decision_engine.reasoning_node import reasoning_node
from decision_engine.guardrail_node import guardrail_node
from decision_engine.execution_node import execution_node


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
) -> Any:
    """
    Build and compile the Day 6 LangGraph workflow.

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

    Returns
    -------
    CompiledStateGraph
        The compiled executable LangGraph instance.
    """
    builder = StateGraph(RecoveryState)

    # 1. Define nodes
    def run_context(state: RecoveryState) -> dict[str, Any]:
        return context_node(state, dataset=dataset)

    def run_estimation(state: RecoveryState) -> dict[str, Any]:
        return estimation_node(state, policy=policy)

    def run_reasoning(state: RecoveryState) -> dict[str, Any]:
        return reasoning_node(state, llm=llm)

    def run_guardrail(state: RecoveryState) -> dict[str, Any]:
        return guardrail_node(state)

    def run_execution(state: RecoveryState) -> dict[str, Any]:
        return execution_node(state, db_path=db_path)

    builder.add_node("context", run_context)
    builder.add_node("estimation", run_estimation)
    builder.add_node("reasoning", run_reasoning)
    builder.add_node("guardrail", run_guardrail)
    builder.add_node("error_fallback", error_fallback_node)
    builder.add_node("execution", run_execution)

    # 2. Define edges and routing
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
    # Error path routes from error_fallback to execution
    builder.add_edge("error_fallback", "execution")

    # Normal path routes: reasoning -> guardrail -> execution
    builder.add_edge("reasoning", "guardrail")
    builder.add_edge("guardrail", "execution")

    # Both paths converge at execution and terminate at END
    builder.add_edge("execution", END)

    return builder.compile()
