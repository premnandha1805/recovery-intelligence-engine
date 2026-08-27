"""
decision_engine/graph.py
========================
LangGraph state graph for Day 6 Recovery Intelligence Engine.

Graph Architecture:
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
      └──────────────┬───────────────┘
                     │
                     ▼
                    END
"""

from __future__ import annotations

from typing import Any, Literal
from langgraph.graph import StateGraph, START, END

from decision_engine.state import RecoveryState
from decision_engine.context_node import context_node
from decision_engine.estimation_node import estimation_node
from decision_engine.reasoning_node import reasoning_node


def error_fallback_node(state: RecoveryState) -> dict[str, Any]:
    """
    Deterministic error handler for malformed or missing payment requests.
    Sets safe default final_action='WAIT' without invoking the LLM.
    """
    err = state.get("error", "Unknown error encountered")
    return {
        "final_action": "WAIT",
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

    builder.add_node("context", run_context)
    builder.add_node("estimation", run_estimation)
    builder.add_node("reasoning", run_reasoning)
    builder.add_node("error_fallback", error_fallback_node)

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
    builder.add_edge("error_fallback", END)
    builder.add_edge("reasoning", END)

    return builder.compile()
