"""
ml/t_learner.py
===============
T-Learner: four treatment-specific binary classifiers for causal uplift.

Each arm model estimates P(Y=1 | X, T=arm) using only its own treatment
rows.  The models use logistic regression by default but the classifier
is swappable through `make_classifier()`.

Architecture
------------
TLearner
  .arm_models : dict[str, Pipeline]
  .fit(X, T, Y)          -- fit all four arms
  .predict_arm_proba(X, arm) -- P(Y=1 | X, T=arm)
  .predict_proba(X)      -- all four arm probabilities

Data flow
---------
1. Load via  ml.dataset.load_causal_dataset() -> X, T, Y
2. Firewall  ml.firewall.assert_no_leakage(training_frame)  per arm
3. Split via ml.splits.create_group_kfold_splits()
4. Per fold, per arm: train on fold-train ∩ arm rows; evaluate on
   fold-val ∩ arm rows

Immutability
------------
This module does NOT:
  - implement treatment-effect calculations
  - implement CausalUpliftPolicy or decision logic
  - modify any frozen Day 3 policy or evaluator
  - modify any data file
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.dataset import (
    OBSERVABLE_FEATURES,
    OUTCOME_COLUMN,
    TREATMENT_COLUMN,
    VALID_TREATMENTS,
    load_causal_dataset,
)
from ml.firewall import assert_no_leakage
from ml.splits import create_group_kfold_splits

# ---------------------------------------------------------------------------
# Feature specification (derived from the canonical contract)
# ---------------------------------------------------------------------------

NUMERIC_FEATURES: list[str] = [
    "amount",
    "attempt_number",
    "dynamic_success_rate",
    "cumulative_failures",
    "consecutive_failed_cycles",
    "notification_engagement_score",
    "contact_response_score",
]

CATEGORICAL_FEATURES: list[str] = [
    "payment_method",
    "failure_reason",
]

# Sanity: union must equal OBSERVABLE_FEATURES
assert sorted(NUMERIC_FEATURES + CATEGORICAL_FEATURES) == sorted(OBSERVABLE_FEATURES), (
    "NUMERIC_FEATURES + CATEGORICAL_FEATURES must equal OBSERVABLE_FEATURES"
)

# Treatment arms in canonical order
ARMS: list[str] = sorted(VALID_TREATMENTS)  # ['ESCALATE', 'RETRY', 'RETRY_NUDGE', 'WAIT']


# ---------------------------------------------------------------------------
# Model factory (swappable classifier)
# ---------------------------------------------------------------------------

def make_classifier(**kwargs: Any) -> Pipeline:
    """Build the default sklearn Pipeline for one arm model.

    Currently returns:
        ColumnTransformer (numeric -> StandardScaler, categorical -> OneHotEncoder)
        -> LogisticRegression

    To swap the classifier later, replace only the final step or
    create an alternative factory; the T-Learner architecture and
    preprocessing remain unchanged.

    Parameters
    ----------
    **kwargs
        Forwarded to LogisticRegression (e.g. max_iter, C, solver).

    Returns
    -------
    Pipeline
        An unfitted sklearn Pipeline ready for .fit(X, y).
    """
    lr_defaults: dict[str, Any] = {
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": 42,
    }
    lr_defaults.update(kwargs)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # never leak non-feature columns
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(**lr_defaults)),
    ])


# ---------------------------------------------------------------------------
# TLearner
# ---------------------------------------------------------------------------

class TLearner:
    """Four treatment-specific binary classifiers.

    Each arm model is trained on rows where T == arm.  Predictions give
    P(Y=1 | X, T=arm) per arm.

    Does NOT calculate treatment effects or make policy decisions.
    """

    def __init__(self, classifier_factory=make_classifier, **factory_kwargs: Any):
        self.classifier_factory = classifier_factory
        self.factory_kwargs = factory_kwargs
        self.arm_models: dict[str, Pipeline] = {}
        self._is_fitted: bool = False

    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> "TLearner":
        """Fit one model per treatment arm.

        For each arm in ARMS:
          1. Select rows where T == arm
          2. Run leakage firewall on the arm training frame
          3. Fit the arm-specific pipeline on (X_arm, Y_arm)

        Parameters
        ----------
        X : pd.DataFrame  with columns == OBSERVABLE_FEATURES
        T : pd.Series     treatment assignments
        Y : pd.Series     binary realized outcomes (0/1)
        """
        assert list(X.columns) == OBSERVABLE_FEATURES, (
            f"X must have exactly OBSERVABLE_FEATURES columns, got {list(X.columns)}"
        )

        for arm in ARMS:
            mask = T == arm
            X_arm = X.loc[mask].reset_index(drop=True)
            Y_arm = Y.loc[mask].reset_index(drop=True)

            if len(X_arm) == 0:
                raise ValueError(f"No training rows for arm '{arm}'. Cannot fit.")

            # Leakage firewall: build the training frame that would enter the
            # pipeline and check it.  Include both features and outcome.
            training_frame = X_arm.copy()
            training_frame[OUTCOME_COLUMN] = Y_arm.values
            assert_no_leakage(training_frame)

            # Fit arm model
            model = self.classifier_factory(**self.factory_kwargs)
            model.fit(X_arm, Y_arm)
            self.arm_models[arm] = model

        self._is_fitted = True
        return self

    def predict_arm_proba(self, X: pd.DataFrame, arm: str) -> np.ndarray:
        """Return P(Y=1 | X, T=arm) as a 1-D array.

        Parameters
        ----------
        X : pd.DataFrame  with columns == OBSERVABLE_FEATURES
        arm : str  one of ARMS

        Returns
        -------
        np.ndarray of shape (n_samples,) with values in [0, 1]
        """
        if not self._is_fitted:
            raise RuntimeError("TLearner has not been fitted. Call .fit() first.")
        if arm not in self.arm_models:
            raise ValueError(f"Unknown arm '{arm}'. Expected one of {ARMS}.")

        proba = self.arm_models[arm].predict_proba(X)
        # predict_proba returns (n, 2); column 1 = P(Y=1)
        return proba[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return P(Y=1 | X, T=arm) for all four arms.

        Returns
        -------
        pd.DataFrame with columns named by arm, values in [0, 1].
        """
        return pd.DataFrame(
            {arm: self.predict_arm_proba(X, arm) for arm in ARMS},
            index=X.index,
        )


# ---------------------------------------------------------------------------
# Cross-validation evaluation
# ---------------------------------------------------------------------------

def cross_validate_t_learner(
    n_splits: int = 5,
    classifier_factory=make_classifier,
    **factory_kwargs: Any,
) -> pd.DataFrame:
    """Run GroupKFold cross-validation and report per-fold, per-arm metrics.

    Returns a DataFrame with columns:
        fold, arm, val_n, positive_rate, auc, brier,
        baseline_brier, signal_assessment

    The baseline is the constant arm-mean predictor using the training
    set's arm-specific positive rate.  This is NOT the validation mean.
    """
    X, T, Y = load_causal_dataset()

    records: list[dict[str, Any]] = []

    for fold_i, (train_idx, val_idx, _groups) in enumerate(
        create_group_kfold_splits(n_splits=n_splits)
    ):
        X_train, T_train, Y_train = X.iloc[train_idx], T.iloc[train_idx], Y.iloc[train_idx]
        X_val, T_val, Y_val = X.iloc[val_idx], T.iloc[val_idx], Y.iloc[val_idx]

        # Fit a T-Learner on this fold's training data
        learner = TLearner(classifier_factory=classifier_factory, **factory_kwargs)
        learner.fit(X_train, T_train, Y_train)

        # Evaluate each arm separately on its validation rows
        for arm in ARMS:
            val_arm_mask = T_val == arm
            n_val_arm = int(val_arm_mask.sum())

            if n_val_arm == 0:
                records.append({
                    "fold": fold_i, "arm": arm, "val_n": 0,
                    "positive_rate": float("nan"), "auc": float("nan"),
                    "brier": float("nan"), "baseline_brier": float("nan"),
                    "signal_assessment": "NO_VAL_ROWS",
                })
                continue

            X_val_arm = X_val.loc[val_arm_mask]
            Y_val_arm = Y_val.loc[val_arm_mask]

            proba = learner.predict_arm_proba(X_val_arm, arm)
            pos_rate = float(Y_val_arm.mean())

            # AUC — requires both classes present
            if len(Y_val_arm.unique()) < 2:
                auc = float("nan")
            else:
                auc = roc_auc_score(Y_val_arm, proba)

            # Brier score
            brier = brier_score_loss(Y_val_arm, proba)

            # Baseline: constant predictor using training arm mean
            train_arm_mask = T_train == arm
            train_arm_mean = float(Y_train.loc[train_arm_mask].mean())
            baseline_brier = brier_score_loss(
                Y_val_arm,
                np.full(n_val_arm, train_arm_mean),
            )

            # Signal assessment
            if np.isnan(auc):
                signal = "AUC_UNAVAILABLE"
            elif auc < 0.55:
                signal = "WEAK"
            elif auc < 0.65:
                signal = "MODERATE"
            else:
                signal = "STRONG"

            records.append({
                "fold": fold_i,
                "arm": arm,
                "val_n": n_val_arm,
                "positive_rate": pos_rate,
                "auc": auc,
                "brier": brier,
                "baseline_brier": baseline_brier,
                "signal_assessment": signal,
            })

    return pd.DataFrame(records)


def fit_final_models(
    classifier_factory=make_classifier,
    **factory_kwargs: Any,
) -> TLearner:
    """Fit the four arm models on the FULL approved training dataset.

    Call this only AFTER cross-validation metrics have been reported and
    reviewed.

    Returns
    -------
    TLearner
        A fitted TLearner with all four arm models.
    """
    X, T, Y = load_causal_dataset()
    learner = TLearner(classifier_factory=classifier_factory, **factory_kwargs)
    learner.fit(X, T, Y)
    return learner


# ---------------------------------------------------------------------------
# CLI — run CV and print results
# ---------------------------------------------------------------------------

def main() -> None:
    """Run cross-validation and print the full 4F report."""
    import sys

    print("=" * 75)
    print("  T-Learner Cross-Validation (4F)")
    print("=" * 75)

    # Dataset summary
    X, T, Y = load_causal_dataset()
    print(f"\nDataset: {len(X):,} rows, {X.shape[1]} features")
    print(f"Features: {list(X.columns)}")
    print(f"Treatment distribution:")
    for arm in ARMS:
        n = int((T == arm).sum())
        pos = float(Y[T == arm].mean())
        print(f"  {arm:<14}: {n:,} rows  positive_rate={pos:.4f}")

    # Cross-validation
    print("\nRunning 5-fold customer-level GroupKFold cross-validation...")
    results = cross_validate_t_learner(n_splits=5)

    # Per-fold, per-arm table
    print("\n" + "-" * 75)
    print(f"{'Fold':>4} | {'Arm':<14} | {'N':>6} | {'Pos Rate':>8} | {'AUC':>7} | {'Brier':>7} | {'Base Brier':>10} | Signal")
    print("-" * 75)
    for _, row in results.iterrows():
        auc_str = f"{row['auc']:.4f}" if not np.isnan(row['auc']) else "   N/A"
        brier_str = f"{row['brier']:.4f}" if not np.isnan(row['brier']) else "   N/A"
        base_str = f"{row['baseline_brier']:.4f}" if not np.isnan(row['baseline_brier']) else "      N/A"
        pr_str = f"{row['positive_rate']:.4f}" if not np.isnan(row['positive_rate']) else "    N/A"
        print(
            f"{int(row['fold']):>4} | {row['arm']:<14} | {int(row['val_n']):>6} | "
            f"{pr_str:>8} | {auc_str:>7} | {brier_str:>7} | {base_str:>10} | {row['signal_assessment']}"
        )

    # Arm-level summary
    print("\n" + "=" * 75)
    print("  ARM-LEVEL SUMMARY")
    print("=" * 75)
    print(f"\n{'Arm':<14} | {'Mean AUC':>8} | {'Std AUC':>7} | {'Mean Brier':>10} | {'Mean Base':>9} | Signal")
    print("-" * 75)
    for arm in ARMS:
        arm_df = results[results["arm"] == arm]
        valid_auc = arm_df["auc"].dropna()
        mean_auc = valid_auc.mean() if len(valid_auc) > 0 else float("nan")
        std_auc = valid_auc.std() if len(valid_auc) > 1 else float("nan")
        mean_brier = arm_df["brier"].mean()
        mean_base = arm_df["baseline_brier"].mean()

        if np.isnan(mean_auc):
            signal = "AUC_UNAVAILABLE"
        elif mean_auc < 0.55:
            signal = "WEAK / LITTLE PREDICTIVE SIGNAL"
        elif mean_auc < 0.65:
            signal = "MODERATE"
        else:
            signal = "STRONG"

        auc_s = f"{mean_auc:.4f}" if not np.isnan(mean_auc) else "    N/A"
        std_s = f"{std_auc:.4f}" if not np.isnan(std_auc) else "   N/A"
        br_s = f"{mean_brier:.4f}" if not np.isnan(mean_brier) else "      N/A"
        bas_s = f"{mean_base:.4f}" if not np.isnan(mean_base) else "     N/A"

        brier_note = ""
        if not np.isnan(mean_brier) and not np.isnan(mean_base):
            improvement = mean_base - mean_brier
            if improvement < 0.001:
                brier_note = " (no improvement over baseline)"
            else:
                brier_note = f" (improves baseline by {improvement:.4f})"

        print(f"{arm:<14} | {auc_s:>8} | {std_s:>7} | {br_s:>10} | {bas_s:>9} | {signal}{brier_note}")

    # Governance notes
    print("\n" + "=" * 75)
    print("  NOTES")
    print("=" * 75)
    print("  - Predictive quality (AUC) is NOT treatment-effect quality.")
    print("  - A good AUC does not prove causal uplift quality.")
    print("  - A weak AUC does not automatically prove no treatment effect.")
    print("  - No treatment effects or policy decisions were implemented.")
    print("  - Customer-level GroupKFold used throughout (ml.splits).")
    print("  - Leakage firewall (ml.firewall) called before each arm fit.")
    print("=" * 75)


if __name__ == "__main__":
    main()
