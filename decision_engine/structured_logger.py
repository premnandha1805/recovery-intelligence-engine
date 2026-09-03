"""
decision_engine/structured_logger.py
====================================
Structured JSON logging helper for Python Decision Engine [Day 7G].
Every event contains: timestamp, service="python-decision-engine", event, request_id.
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any

SERVICE_NAME = "python-decision-engine"


def emit_log(
    logger_instance: logging.Logger,
    level: int,
    event: str,
    request_id: str,
    **kwargs: Any,
) -> None:
    """
    Emit a single structured JSON log event.

    Schema:
    {
        "timestamp": "<ISO-8601 UTC timestamp>",
        "service": "python-decision-engine",
        "event": "<event_name>",
        "request_id": "<correlation_id>",
        ...additional structured fields...
    }
    """
    payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "service": SERVICE_NAME,
        "event": event,
        "request_id": request_id,
        **kwargs,
    }
    logger_instance.log(level, json.dumps(payload))
