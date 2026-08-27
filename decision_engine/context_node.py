"""
decision_engine/context_node.py
===============================
Context retrieval node for the Recovery Decision Engine.

Extracts observable features, payment context, and customer history from
the canonical dataset without using LLMs or leaking hidden ground truth.
"""

from __future__ import annotations

import pathlib
from typing import Any, Mapping
import pandas as pd

from decision_engine.state import RecoveryState
from ml.dataset import OBSERVABLE_FEATURES

# Canonical dataset paths
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CANONICAL_DATA_PATH = _REPO_ROOT / "data" / "v2" / "payment_scenarios.csv"
SEED777_DATA_PATH = _REPO_ROOT / "ml" / "evaluation" / "seed777_data" / "payment_scenarios.csv"

# Global lazy cache for the dataset to ensure fast lookups across nodes/tests
_DATASET_CACHE: dict[str, pd.DataFrame] = {}


def _get_dataset(custom_path: pathlib.Path | str | None = None) -> pd.DataFrame:
    """Load canonical payment scenarios dataset with caching."""
    path_key = str(custom_path) if custom_path else str(CANONICAL_DATA_PATH)
    if path_key in _DATASET_CACHE:
        return _DATASET_CACHE[path_key]

    resolved_path = pathlib.Path(path_key)
    if not resolved_path.exists():
        if SEED777_DATA_PATH.exists():
            resolved_path = SEED777_DATA_PATH
        else:
            raise FileNotFoundError(f"Canonical dataset not found at {resolved_path}")

    df = pd.read_csv(resolved_path)
    _DATASET_CACHE[path_key] = df
    return df


def context_node(
    state: RecoveryState,
    dataset: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Validate payment_id and load context partitions.

    Parameters
    ----------
    state : RecoveryState
        Current LangGraph workflow state.
    dataset : pd.DataFrame, optional
        Explicit dataset dataframe for testing or custom scenarios.

    Returns
    -------
    dict[str, Any]
        Partial state dictionary update.
    """
    payment_id = state.get("payment_id")

    # 1. Validate payment_id
    if not payment_id or not isinstance(payment_id, str) or not payment_id.strip():
        reason = "Invalid or missing payment_id"
        return {
            "error": reason,
            "audit_trail": [
                {
                    "node": "context_node",
                    "status": "error",
                    "reason": reason,
                }
            ],
        }

    clean_pid = payment_id.strip()

    # 2. Check if context is already provided in state (e.g. synthetic test input)
    if (
        state.get("observable_features")
        and state.get("payment_context")
        and state.get("customer_history")
    ):
        return {
            "payment_id": clean_pid,
            "error": None,
            "audit_trail": [
                {
                    "node": "context_node",
                    "status": "success",
                    "payment_id": clean_pid,
                    "source": "pre_populated",
                }
            ],
        }

    # 3. Lookup in dataset
    try:
        df = dataset if dataset is not None else _get_dataset()
    except Exception as exc:
        reason = f"Failed to access canonical dataset: {exc}"
        return {
            "error": reason,
            "audit_trail": [
                {
                    "node": "context_node",
                    "status": "error",
                    "reason": reason,
                }
            ],
        }

    # Search in dataset (first primary dataset, fallback to seed777 if not present)
    matches = df[df["payment_id"] == clean_pid]
    if matches.empty and SEED777_DATA_PATH.exists() and dataset is None:
        try:
            df_seed = _get_dataset(SEED777_DATA_PATH)
            matches = df_seed[df_seed["payment_id"] == clean_pid]
        except Exception:
            pass

    if matches.empty:
        reason = f"Payment ID {clean_pid!r} not found in canonical dataset"
        return {
            "error": reason,
            "audit_trail": [
                {
                    "node": "context_node",
                    "status": "error",
                    "reason": reason,
                }
            ],
        }

    row = matches.iloc[0]

    # 4. Strict separation of partitions
    observable_features = {
        col: (float(row[col]) if pd.api.types.is_numeric_dtype(type(row[col])) else row[col])
        for col in OBSERVABLE_FEATURES
    }
    # Ensure types match ml.dataset contract
    observable_features["amount"] = float(row["amount"])
    observable_features["attempt_number"] = int(row["attempt_number"])
    observable_features["dynamic_success_rate"] = float(row["dynamic_success_rate"])
    observable_features["cumulative_failures"] = int(row["cumulative_failures"])
    observable_features["consecutive_failed_cycles"] = int(row["consecutive_failed_cycles"])
    observable_features["notification_engagement_score"] = float(row["notification_engagement_score"])
    observable_features["contact_response_score"] = float(row["contact_response_score"])
    observable_features["payment_method"] = str(row["payment_method"])
    observable_features["failure_reason"] = str(row["failure_reason"])

    payment_context = {
        "payment_id": clean_pid,
        "billing_cycle_id": str(row.get("billing_cycle_id", "")),
        "subscription_id": str(row.get("subscription_id", "")),
        "amount": float(row["amount"]),
        "status": "failed",
        "attempt_number": int(row["attempt_number"]),
        "payment_method": str(row["payment_method"]),
        "failure_reason": str(row["failure_reason"]),
        "consecutive_failures": int(row["consecutive_failed_cycles"]),
        "retry_count_current_cycle": max(0, int(row["attempt_number"]) - 1),
    }

    input_pmt_ctx = state.get("payment_context") or {}
    payment_context.update(input_pmt_ctx)

    input_cust_hist = state.get("customer_history") or {}
    customer_history = {
        "customer_id": str(row.get("customer_id", "")),
        "days_active": int(row.get("days_active", 0)),
        "dynamic_success_rate": float(row["dynamic_success_rate"]),
        "cumulative_failures": int(row["cumulative_failures"]),
        "consecutive_failed_cycles": int(row["consecutive_failed_cycles"]),
        "notification_engagement_score": float(row["notification_engagement_score"]),
        "contact_response_score": float(row["contact_response_score"]),
        "lifetime_escalations": 0,
        "interventions_last_7_days": 0,
    }
    customer_history.update(input_cust_hist)


    return {
        "payment_id": clean_pid,
        "observable_features": observable_features,
        "payment_context": payment_context,
        "customer_history": customer_history,
        "error": None,
        "audit_trail": [
            {
                "node": "context_node",
                "status": "success",
                "payment_id": clean_pid,
            }
        ],
    }
