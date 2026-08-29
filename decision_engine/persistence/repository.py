"""
decision_engine/persistence/repository.py
=========================================
Abstract repository interface for Decision Engine persistence.

Implementation-neutral Protocol with zero driver or database dependencies.
The rest of the system (LangGraph nodes, service.py, evaluation) depends only
on this abstraction, enabling pure in-memory testing without a PostgreSQL dependency.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DecisionRepository(Protocol):
    """
    Abstract interface for Decision persistence.

    Implementations must support:
    - Current decision retrieval and UPSERT by payment_id
    - Immutable audit event appending and chronological retrieval
    """

    async def get_current_decision(
        self, payment_id: str
    ) -> dict[str, Any] | None:
        """
        Retrieve the latest decision audit record for payment_id.

        Returns
        -------
        dict[str, Any] | None
            Decision audit record matching the canonical schema, or None if not found.
        """
        ...

    async def save_current_decision(
        self, **kwargs: Any
    ) -> None:
        """
        Persist or update the current decision state with UPSERT semantics.

        Parameters
        ----------
        **kwargs : Any
            Must include payment_id.
            Canonical fields:
            - payment_id: str (required)
            - decision_id: str
            - request_id: str | None
            - raw_arm_probabilities: dict[str, float] | None
            - raw_arm_net_values: dict[str, float] | None
            - llm_proposed_decision: str | None
            - llm_confidence: float | None
            - llm_reasoning: str | None
            - llm_risk_level: str | None
            - expected_incremental_value: float | None
            - guardrail_verdict: str | None
            - guardrail_reason: str | None
            - final_action: str
            - decision_source: str
            - error: str | None
            - evaluated_at: datetime | str
            - state_fingerprint: str | None
        """
        ...

    async def append_decision_event(
        self, **kwargs: Any
    ) -> None:
        """
        Append an immutable audit event record.

        Parameters
        ----------
        **kwargs : Any
            Must include payment_id.
            Canonical fields:
            - decision_id: str
            - payment_id: str (required)
            - request_id: str | None
            - evaluated_at: datetime | str
            - decision_source: str | None
            - final_action: str
            - model_decision: str | None
            - llm_proposed_decision: str | None
            - guardrail_overridden: bool | None
            - guardrail_reason: str | None
            - state_fingerprint: str | None
        """
        ...

    async def get_events(
        self, payment_id: str
    ) -> list[dict[str, Any]]:
        """
        Retrieve all immutable audit events for payment_id ordered chronologically
        by evaluated_at ASC.

        Returns
        -------
        list[dict[str, Any]]
            List of audit event records, or empty list if none exist.
        """
        ...
