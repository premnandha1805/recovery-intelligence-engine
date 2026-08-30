"""
decision_engine/persistence/in_memory.py
========================================
In-memory implementation of DecisionRepository using plain Python dictionaries and lists.
No database dependencies, enabling fast, isolated unit testing.
"""

from __future__ import annotations

import datetime
from typing import Any


class InMemoryDecisionRepository:
    """
    In-memory concrete implementation of DecisionRepository.

    Maintains:
    - _decisions: dict[str, dict[str, Any]] keyed by payment_id (UPSERT semantics)
    - _events: dict[str, list[dict[str, Any]]] keyed by payment_id (append-only list)
    """

    def __init__(self) -> None:
        self._decisions: dict[str, dict[str, Any]] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}

    async def get_current_decision(self, payment_id: str) -> dict[str, Any] | None:
        """Retrieve latest decision audit record for payment_id, or None if absent."""
        record = self._decisions.get(payment_id)
        return dict(record) if record is not None else None

    async def save_current_decision(self, **kwargs: Any) -> None:
        """
        Persist or update the current decision state with UPSERT semantics.
        Overwrites any previous decision for the given payment_id.
        """
        payment_id = kwargs.get("payment_id")
        if not payment_id:
            raise ValueError("payment_id is required for save_current_decision")

        # Copy data to prevent caller mutation side-effects
        self._decisions[payment_id] = dict(kwargs)

    async def append_decision_event(self, **kwargs: Any) -> None:
        """
        Append an immutable decision audit event row.
        Never overwrites previous events.
        """
        payment_id = kwargs.get("payment_id")
        if not payment_id:
            raise ValueError("payment_id is required for append_decision_event")

        event_list = self._events.setdefault(payment_id, [])
        event_list.append(dict(kwargs))

    async def get_events(self, payment_id: str) -> list[dict[str, Any]]:
        """
        Retrieve all audit events for payment_id, sorted chronologically by evaluated_at ASC.
        Returns [] if no events are recorded.
        """
        events = self._events.get(payment_id, [])

        def _sort_key(ev: dict[str, Any]) -> Any:
            val = ev.get("evaluated_at")
            if isinstance(val, (datetime.datetime, datetime.date)):
                return val.isoformat()
            return str(val or "")

        sorted_events = sorted(events, key=_sort_key)
        return [dict(ev) for ev in sorted_events]

    async def save_decision_with_event(self, **kwargs: Any) -> None:
        """
        Atomically persist a current-decision record and append an audit event record.
        Maintains all-or-nothing rollback semantics if either operation fails.
        """
        payment_id = kwargs.get("payment_id")
        if not payment_id:
            raise ValueError("payment_id is required for save_decision_with_event")

        prev_decision = dict(self._decisions[payment_id]) if payment_id in self._decisions else None
        prev_events_len = len(self._events.get(payment_id, []))

        await self.save_current_decision(**kwargs)

        try:
            event_data = dict(kwargs)
            if "event_decision_id" in kwargs:
                event_data["decision_id"] = kwargs["event_decision_id"]
            await self.append_decision_event(**event_data)
        except Exception:
            # Rollback current decision
            if prev_decision is None:
                self._decisions.pop(payment_id, None)
            else:
                self._decisions[payment_id] = prev_decision
            # Rollback appended event
            if payment_id in self._events:
                self._events[payment_id] = self._events[payment_id][:prev_events_len]
            raise

    def clear(self) -> None:
        """Clear all stored in-memory decisions and event logs for test isolation."""
        self._decisions.clear()
        self._events.clear()
