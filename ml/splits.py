"""
ml/splits.py
============
Customer-level GroupKFold cross-validation splits for the causal ML pipeline.

Design rationale
----------------
Each customer generates multiple payment rows.  A standard KFold or random
train/test split would allow the same customer to appear in both training and
validation, causing data leakage at the customer level (the model could
implicitly memorise per-customer patterns and over-report generalisation).

GroupKFold assigns whole customers to either train or validation in every
fold.  This guarantees:

    For every fold:  train_customers ∩ validation_customers == empty set

Customer-ID source
------------------
``customer_id`` does NOT appear in ``ml/data/causal_training_data.csv``
(which contains only observable X features, assigned_T, and realized_Y —
see ml/dataset.py).  The mapping is recovered by joining on ``payment_id``
against ``data/v2/payment_scenarios.csv``, which is the V2 simulator
observable dataset and is NOT modified by this module.

The pre-computed CV fold assignments in ``data/v2/cv_groups.csv`` confirm
the customer universe but are not used to drive GroupKFold here; instead
we pass the raw ``customer_id`` array directly to GroupKFold so sklearn
computes the splits deterministically without assuming a fixed fold count.

Immutability
------------
This module does NOT:
  - train any model
  - estimate treatment effects
  - modify any baseline policy or evaluator
  - modify any data file
"""

from __future__ import annotations

import pathlib
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# ---------------------------------------------------------------------------
# Repository-root-safe paths (resolved once at import time)
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

DEFAULT_TRAINING_DATA_PATH = _REPO_ROOT / "ml" / "data" / "causal_training_data.csv"
DEFAULT_OBS_PATH = _REPO_ROOT / "data" / "v2" / "payment_scenarios.csv"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_customer_groups(
    training_data_path: pathlib.Path | str | None = None,
    obs_path: pathlib.Path | str | None = None,
) -> pd.Series:
    """Return a Series of customer_id values aligned to the causal training data.

    The series has the same index and row-order as the causal training CSV
    so it can be passed directly as the ``groups`` argument to
    sklearn GroupKFold.

    Data lineage
    ------------
    causal_training_data.csv  --[payment_id join]-->  data/v2/payment_scenarios.csv
                                                              |
                                                        customer_id

    Parameters
    ----------
    training_data_path:
        Path to the approved causal training CSV (ml/data/causal_training_data.csv).
    obs_path:
        Path to the V2 observable dataset (data/v2/payment_scenarios.csv).

    Returns
    -------
    pd.Series
        customer_id values, one per row of the training data, in matching order.

    Raises
    ------
    FileNotFoundError
        If either source file is missing.
    ValueError
        If the payment_id join loses any rows (data integrity violation).
    """
    t_path = pathlib.Path(training_data_path) if training_data_path else DEFAULT_TRAINING_DATA_PATH
    o_path = pathlib.Path(obs_path) if obs_path else DEFAULT_OBS_PATH

    if not t_path.exists():
        raise FileNotFoundError(
            f"Causal training data not found: {t_path}\n"
            "Run ml/generate_causal_training_data.py first (Day 4B)."
        )
    if not o_path.exists():
        raise FileNotFoundError(
            f"Observable dataset not found: {o_path}"
        )

    train = pd.read_csv(t_path)
    obs   = pd.read_csv(o_path, usecols=["payment_id", "customer_id"])

    if "payment_id" not in train.columns:
        raise ValueError("'payment_id' column missing from causal training data.")
    if "customer_id" not in obs.columns:
        raise ValueError("'customer_id' column missing from observable dataset.")

    merged = train[["payment_id"]].merge(obs, on="payment_id", how="left")

    n_missing = merged["customer_id"].isna().sum()
    if n_missing > 0:
        raise ValueError(
            f"{n_missing} payment_id rows in the training data could not be "
            "mapped to a customer_id.  Do not invent or approximate groupings."
        )

    if len(merged) != len(train):
        raise ValueError(
            f"Join produced {len(merged):,} rows but training data has "
            f"{len(train):,} rows.  Data integrity violation."
        )

    return merged["customer_id"].reset_index(drop=True)


def create_group_kfold_splits(
    n_splits: int = 5,
    training_data_path: pathlib.Path | str | None = None,
    obs_path: pathlib.Path | str | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, pd.Series]]:
    """Yield customer-level GroupKFold (train_idx, val_idx, groups) tuples.

    Payments belonging to the same customer are always kept together in
    either the training set or the validation set — never split across both.

    Parameters
    ----------
    n_splits:
        Number of folds for GroupKFold (default 5, matching cv_groups.csv).
    training_data_path:
        Optional override for the causal training CSV path.
    obs_path:
        Optional override for the observable dataset path.

    Yields
    ------
    train_idx : np.ndarray
        Row indices (into the training data) that belong to the training fold.
    val_idx : np.ndarray
        Row indices that belong to the validation fold.
    groups : pd.Series
        The full customer_id series (same for every fold; provided for
        convenience so callers do not need to reconstruct it).

    Notes
    -----
    GroupKFold is deterministic; no random seed is needed.
    The function uses customer_id as the ``groups`` argument to GroupKFold,
    never payment_id or any randomly reconstructed grouping.
    """
    groups = load_customer_groups(training_data_path, obs_path)

    n_rows = len(groups)
    X_placeholder = np.zeros((n_rows, 1))  # GroupKFold only needs the shape

    gkf = GroupKFold(n_splits=n_splits)

    for train_idx, val_idx in gkf.split(X_placeholder, groups=groups):
        yield train_idx, val_idx, groups
