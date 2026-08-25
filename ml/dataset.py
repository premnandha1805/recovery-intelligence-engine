"""
ml/dataset.py
=============
Causal training-data contract and leakage firewall.

This module defines the ONLY authorised way to load the Day 4B randomized-
logging dataset for downstream causal-uplift modelling.

Contract
--------
X  -- observable pre-treatment features only (no hidden simulator columns)
T  -- the randomized treatment assignment (one of WAIT / RETRY / RETRY_NUDGE / ESCALATE)
Y  -- the realized binary outcome: strictly 0 or 1, NEVER a probability

Hidden simulator probabilities (natural_recovery_probability,
retry_success_probability, etc.) were used *only* during Day 4B to draw Y
from a Bernoulli distribution.  They must never appear as ML features and
are explicitly blocked by this module.

Returning X, T, Y as three separate objects is intentional leakage
protection: downstream code cannot accidentally select hidden, identifier,
or probability columns from a single bundled dataframe.

Immutability
------------
This module does NOT train any model, estimate uplift, or modify any
policy, evaluator, or frozen Day 3 artefact.
"""

from __future__ import annotations

import pathlib

import pandas as pd

# ---------------------------------------------------------------------------
# Dataset contract — explicit constants, NOT inferred from CSV schema
# ---------------------------------------------------------------------------

OBSERVABLE_FEATURES: list[str] = [
    "amount",
    "attempt_number",
    "dynamic_success_rate",
    "cumulative_failures",
    "consecutive_failed_cycles",
    "notification_engagement_score",
    "contact_response_score",
    "payment_method",
    "failure_reason",
]

TREATMENT_COLUMN: str = "assigned_T"

OUTCOME_COLUMN: str = "realized_Y"

# realized_Y is a binary realized outcome: 0/1.
# It is NEVER a probability.

VALID_TREATMENTS: frozenset[str] = frozenset({"WAIT", "RETRY", "RETRY_NUDGE", "ESCALATE"})

FORBIDDEN_PREFIXES: list[str] = [
    "p_success",
    "hidden_",
    "natural_recovery",
    "retry_success_probability",
    "nudge_success_probability",
    "escalation_success_probability",
    "expected_recovery",
    "expected_net_value",
]

# Resolved once at import time so the path is robust regardless of CWD
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_TRAINING_DATA_PATH = _REPO_ROOT / "ml" / "data" / "causal_training_data.csv"


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _check_forbidden_columns(columns: list[str]) -> None:
    """Raise if any column name starts with or equals a forbidden prefix."""
    violations = []
    for col in columns:
        for prefix in FORBIDDEN_PREFIXES:
            if col == prefix or col.startswith(prefix):
                violations.append(f"  '{col}'  (matches forbidden prefix '{prefix}')")
                break
    if violations:
        raise ValueError(
            "Forbidden columns detected in training data — "
            "hidden simulator probabilities must not enter the ML pipeline:\n"
            + "\n".join(violations)
        )


def _check_required_columns(df: pd.DataFrame) -> None:
    """Raise if any required feature, treatment, or outcome column is missing."""
    missing_features = [c for c in OBSERVABLE_FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Required observable features missing from dataset: {missing_features}\n"
            "Do not infer a replacement column — fix the source data."
        )

    if TREATMENT_COLUMN not in df.columns:
        raise ValueError(
            f"Treatment column '{TREATMENT_COLUMN}' is missing from dataset."
        )

    if OUTCOME_COLUMN not in df.columns:
        raise ValueError(
            f"Outcome column '{OUTCOME_COLUMN}' is missing from dataset."
        )


def _check_no_missing_values(df: pd.DataFrame) -> None:
    """Raise if any required column contains NaN — no silent imputation."""
    required = OBSERVABLE_FEATURES + [TREATMENT_COLUMN, OUTCOME_COLUMN]
    null_counts = {c: int(df[c].isna().sum()) for c in required if df[c].isna().any()}
    if null_counts:
        raise ValueError(
            "Missing values detected — this module does not impute:\n"
            + "\n".join(f"  '{c}': {n} null(s)" for c, n in null_counts.items())
        )


def _check_treatment_values(t: pd.Series) -> None:
    """Raise if T contains unexpected or null treatment labels."""
    if t.isna().any():
        raise ValueError(
            f"Treatment column '{TREATMENT_COLUMN}' contains missing values."
        )
    unexpected = set(t.unique()) - VALID_TREATMENTS
    if unexpected:
        raise ValueError(
            f"Unexpected treatment values in '{TREATMENT_COLUMN}': {unexpected}\n"
            f"Allowed values: {sorted(VALID_TREATMENTS)}"
        )


def _check_binary_outcome(y: pd.Series) -> None:
    """Raise if Y contains anything other than integer 0 or 1.

    Rejects: probabilities (0.137, 0.5), out-of-range integers (-1, 2),
    floats that happen to equal 0.0 or 1.0 only if the dtype is float
    with non-integer values present, and NaN.

    Does NOT round, threshold, or cast.
    """
    if y.isna().any():
        raise ValueError(
            f"Outcome column '{OUTCOME_COLUMN}' contains NaN values. "
            "Y must be strictly binary 0/1."
        )

    # Check dtype: must be integer-compatible
    if not pd.api.types.is_integer_dtype(y):
        # Allow float dtype only if every value is exactly 0.0 or 1.0
        # (e.g. CSV loads integers as float64 sometimes)
        non_binary = y[~y.isin([0.0, 1.0])]
        if not non_binary.empty:
            sample = non_binary.head(5).tolist()
            raise ValueError(
                f"Outcome column '{OUTCOME_COLUMN}' contains non-binary values: {sample}\n"
                "Y must be strictly 0 or 1. "
                "Do not round, threshold, or cast probabilities."
            )
        # All values are 0.0 / 1.0 — acceptable, will be int-like in practice
    else:
        # Integer dtype — just check no values outside {0, 1}
        non_binary = y[~y.isin([0, 1])]
        if not non_binary.empty:
            sample = non_binary.head(5).tolist()
            raise ValueError(
                f"Outcome column '{OUTCOME_COLUMN}' contains non-binary values: {sample}\n"
                "Y must be strictly 0 or 1."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_causal_dataset(
    path: pathlib.Path | str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load and validate the Day 4B causal training dataset.

    Returns
    -------
    X : pd.DataFrame
        Exactly the nine observable pre-treatment features defined in
        OBSERVABLE_FEATURES.  payment_id and all other columns are excluded.

    T : pd.Series
        Treatment assignments from TREATMENT_COLUMN ('assigned_T').
        Contains only WAIT / RETRY / RETRY_NUDGE / ESCALATE.

    Y : pd.Series
        Realized binary outcomes from OUTCOME_COLUMN ('realized_Y').
        Contains only integers 0 or 1 — never a probability.

    Raises
    ------
    ValueError
        On any contract violation: missing columns, forbidden columns,
        non-binary Y, unexpected T values, or missing values.
    FileNotFoundError
        If the training data CSV cannot be found.
    """
    resolved_path = pathlib.Path(path) if path is not None else DEFAULT_TRAINING_DATA_PATH

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Causal training data not found at: {resolved_path}\n"
            "Run ml/generate_causal_training_data.py first (Day 4B)."
        )

    df = pd.read_csv(resolved_path)

    # --- A. Forbidden-column firewall (runs before anything else) --------
    # Must check the raw CSV columns, not just the ones we plan to select.
    _check_forbidden_columns(list(df.columns))

    # --- B/C. Required columns exist -------------------------------------
    _check_required_columns(df)

    # --- D/E. Missing values in required columns -------------------------
    _check_no_missing_values(df)

    # --- F. Treatment values valid and complete --------------------------
    _check_treatment_values(df[TREATMENT_COLUMN])

    # --- G. Outcome is strictly binary -----------------------------------
    _check_binary_outcome(df[OUTCOME_COLUMN])

    # --- H. Separate X / T / Y (leakage firewall) -----------------------
    # payment_id intentionally excluded from X even if present in CSV.
    X = df[OBSERVABLE_FEATURES].copy()
    T = df[TREATMENT_COLUMN].copy()
    Y = df[OUTCOME_COLUMN].copy()

    return X, T, Y
