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
from decision_engine.audit import compute_state_fingerprint
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


def get_payment_state(
    payment_id: str,
    dataset: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """
    Extract the 6 state-fingerprint inputs for a given payment_id using the in-memory dataset cache.
    Inputs: payment_id, status, attempt_number, consecutive_failures, retry_count, interventions_7d.
    """
    if not payment_id or not isinstance(payment_id, str) or not payment_id.strip():
        return None

    clean_pid = payment_id.strip()
    try:
        df = dataset if dataset is not None else _get_dataset()
    except Exception:
        return None

    matches = df[df["payment_id"] == clean_pid]
    if matches.empty and SEED777_DATA_PATH.exists() and dataset is None:
        try:
            df_seed = _get_dataset(SEED777_DATA_PATH)
            matches = df_seed[df_seed["payment_id"] == clean_pid]
        except Exception:
            pass

    if matches.empty:
        return None

    row = matches.iloc[0]

    status = str(row["status"]).strip() if "status" in row and pd.notna(row["status"]) else "failed"
    attempt_number = int(row["attempt_number"]) if "attempt_number" in row and pd.notna(row["attempt_number"]) else 1

    if "consecutive_failures" in row and pd.notna(row["consecutive_failures"]):
        consecutive_failures = int(row["consecutive_failures"])
    elif "consecutive_failed_cycles" in row and pd.notna(row["consecutive_failed_cycles"]):
        consecutive_failures = int(row["consecutive_failed_cycles"])
    else:
        consecutive_failures = 0

    if "retry_count" in row and pd.notna(row["retry_count"]):
        retry_count = int(row["retry_count"])
    elif "retry_count_current_cycle" in row and pd.notna(row["retry_count_current_cycle"]):
        retry_count = int(row["retry_count_current_cycle"])
    else:
        retry_count = max(0, attempt_number - 1)

    if "interventions_7d" in row and pd.notna(row["interventions_7d"]):
        interventions_7d = int(row["interventions_7d"])
    elif "interventions_last_7_days" in row and pd.notna(row["interventions_last_7_days"]):
        interventions_7d = int(row["interventions_last_7_days"])
    else:
        interventions_7d = 0

    return {
        "payment_id": clean_pid,
        "status": status,
        "attempt_number": attempt_number,
        "consecutive_failures": consecutive_failures,
        "retry_count": retry_count,
        "interventions_7d": interventions_7d,
    }


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
        pmt_ctx = state.get("payment_context", {})
        cust_hist = state.get("customer_history", {})
        obs_feat = state.get("observable_features", {})
        fingerprint = state.get("state_fingerprint") or compute_state_fingerprint(
            payment_id=clean_pid,
            status=pmt_ctx.get("status", "failed"),
            attempt_number=pmt_ctx.get("attempt_number", 1),
            consecutive_failures=pmt_ctx.get("consecutive_failures", 0),
            retry_count=pmt_ctx.get("retry_count_current_cycle", pmt_ctx.get("retry_count", 0)),
            interventions_7d=cust_hist.get("interventions_last_7_days", cust_hist.get("interventions_7d", 0)),
            decision_features=obs_feat,
        )
        return {
            "payment_id": clean_pid,
            "observable_features": obs_feat,
            "payment_context": pmt_ctx,
            "customer_history": cust_hist,
            "state_fingerprint": fingerprint,
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

    status = str(row["status"]).strip() if "status" in row and pd.notna(row["status"]) else "failed"
    attempt_num = int(row["attempt_number"])
    consec_fail = int(row["consecutive_failures"]) if "consecutive_failures" in row and pd.notna(row["consecutive_failures"]) else int(row.get("consecutive_failed_cycles", 0))
    retry_cnt = int(row["retry_count"]) if "retry_count" in row and pd.notna(row["retry_count"]) else max(0, attempt_num - 1)

    payment_context = {
        "payment_id": clean_pid,
        "billing_cycle_id": str(row.get("billing_cycle_id", "")),
        "subscription_id": str(row.get("subscription_id", "")),
        "amount": float(row["amount"]),
        "status": status,
        "attempt_number": attempt_num,
        "payment_method": str(row["payment_method"]),
        "failure_reason": str(row["failure_reason"]),
        "consecutive_failures": consec_fail,
        "retry_count_current_cycle": retry_cnt,
    }

    input_pmt_ctx = state.get("payment_context") or {}
    payment_context.update(input_pmt_ctx)

    interv_7d = int(row["interventions_7d"]) if "interventions_7d" in row and pd.notna(row["interventions_7d"]) else (int(row["interventions_last_7_days"]) if "interventions_last_7_days" in row and pd.notna(row["interventions_last_7_days"]) else 0)

    input_cust_hist = state.get("customer_history") or {}
    customer_history = {
        "customer_id": str(row.get("customer_id", "")),
        "days_active": int(row.get("days_active", 0)),
        "dynamic_success_rate": float(row["dynamic_success_rate"]),
        "cumulative_failures": int(row["cumulative_failures"]),
        "consecutive_failed_cycles": int(row["consecutive_failed_cycles"]),
        "notification_engagement_score": float(row["notification_engagement_score"]),
        "contact_response_score": float(row["contact_response_score"]),
        "lifetime_escalations": int(row["lifetime_escalations"]) if "lifetime_escalations" in row and pd.notna(row.get("lifetime_escalations")) else 0,
        "interventions_last_7_days": interv_7d,
    }
    customer_history.update(input_cust_hist)

    fingerprint = compute_state_fingerprint(
        payment_id=clean_pid,
        status=payment_context.get("status", "failed"),
        attempt_number=payment_context.get("attempt_number", 1),
        consecutive_failures=payment_context.get("consecutive_failures", 0),
        retry_count=payment_context.get("retry_count_current_cycle", payment_context.get("retry_count", 0)),
        interventions_7d=customer_history.get("interventions_last_7_days", 0),
    )

    return {
        "payment_id": clean_pid,
        "observable_features": observable_features,
        "payment_context": payment_context,
        "customer_history": customer_history,
        "state_fingerprint": fingerprint,
        "error": None,
        "audit_trail": [
            {
                "node": "context_node",
                "status": "success",
                "payment_id": clean_pid,
            }
        ],
    }
