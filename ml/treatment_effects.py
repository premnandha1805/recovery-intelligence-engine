"""
ml/treatment_effects.py
=======================
Individual treatment effect (ITE) estimation from a fitted T-Learner.

This module takes a fitted TLearner and observable feature matrix X and
produces:

  1. Four estimated potential outcomes:
       y_hat_wait, y_hat_retry, y_hat_retry_nudge, y_hat_escalate

  2. Three treatment effects relative to WAIT (the control arm):
       tau_retry     = y_hat_retry       - y_hat_wait
       tau_nudge     = y_hat_retry_nudge - y_hat_wait
       tau_escalate  = y_hat_escalate    - y_hat_wait

The tau values are T-Learner estimates of conditional average treatment
effects (CATE):
    E[Y | X, T=a] - E[Y | X, T=WAIT]

They are NOT hidden ground-truth uplift probabilities.

No clipping, calibration, rescaling, or normalisation is applied.
Negative treatment effects are valid and preserved.

Immutability
------------
This module does NOT:
  - implement any decision policy (CausalUpliftPolicy)
  - modify any frozen Day 3 artefact
  - retrain or modify the T-Learner architecture
  - modify any data file
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.dataset import OBSERVABLE_FEATURES, load_causal_dataset
from ml.splits import create_group_kfold_splits
from ml.t_learner import ARMS, TLearner, make_classifier


# ---------------------------------------------------------------------------
# Public API — potential outcomes and treatment effects
# ---------------------------------------------------------------------------

def estimate_potential_outcomes(
    t_learner: TLearner,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Return model-estimated P(Y=1 | X, T=arm) for all four arms.

    Parameters
    ----------
    t_learner : TLearner
        A fitted TLearner (from ml.t_learner).
    X : pd.DataFrame
        Observable features with columns == OBSERVABLE_FEATURES.

    Returns
    -------
    pd.DataFrame with columns:
        y_hat_wait, y_hat_retry, y_hat_retry_nudge, y_hat_escalate
    All values are model-predicted probabilities in [0, 1].
    """
    assert list(X.columns) == OBSERVABLE_FEATURES, (
        f"X must have exactly OBSERVABLE_FEATURES columns, got {list(X.columns)}"
    )

    return pd.DataFrame(
        {
            "y_hat_wait": t_learner.predict_arm_proba(X, "WAIT"),
            "y_hat_retry": t_learner.predict_arm_proba(X, "RETRY"),
            "y_hat_retry_nudge": t_learner.predict_arm_proba(X, "RETRY_NUDGE"),
            "y_hat_escalate": t_learner.predict_arm_proba(X, "ESCALATE"),
        },
        index=X.index,
    )


def estimate_treatment_effects(
    t_learner: TLearner,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Return estimated individual treatment effects relative to WAIT.

    tau_arm = P(Y=1 | X, T=arm) - P(Y=1 | X, T=WAIT)

    No clipping, calibration, rescaling, or normalisation is applied.
    Negative treatment effects are valid and preserved.

    Parameters
    ----------
    t_learner : TLearner
        A fitted TLearner (from ml.t_learner).
    X : pd.DataFrame
        Observable features with columns == OBSERVABLE_FEATURES.

    Returns
    -------
    pd.DataFrame with columns:
        tau_retry, tau_nudge, tau_escalate
    Values are in probability units (e.g., 0.10 = +10 percentage points).
    """
    po = estimate_potential_outcomes(t_learner, X)

    return pd.DataFrame(
        {
            "tau_retry": po["y_hat_retry"] - po["y_hat_wait"],
            "tau_nudge": po["y_hat_retry_nudge"] - po["y_hat_wait"],
            "tau_escalate": po["y_hat_escalate"] - po["y_hat_wait"],
        },
        index=X.index,
    )


# ---------------------------------------------------------------------------
# Out-of-fold treatment effect estimation
# ---------------------------------------------------------------------------

def compute_oof_treatment_effects(
    n_splits: int = 5,
    classifier_factory=make_classifier,
    **factory_kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute out-of-fold potential outcomes and treatment effects.

    For every GroupKFold fold:
      1. Fit a FRESH TLearner on training rows only
      2. Predict potential outcomes on validation rows only
      3. Compute treatment effects on those validation predictions

    Returns
    -------
    oof_potential_outcomes : pd.DataFrame
        Columns: y_hat_wait, y_hat_retry, y_hat_retry_nudge, y_hat_escalate
        One row per payment in the dataset (30,472 total).
        Each row's predictions come from a model that NEVER saw that row
        during training.

    oof_treatment_effects : pd.DataFrame
        Columns: tau_retry, tau_nudge, tau_escalate
        Computed as arm - wait on the OOF predictions.
    """
    X, T, Y = load_causal_dataset()
    n_total = len(X)

    # Pre-allocate OOF arrays
    oof_po = pd.DataFrame(
        np.nan,
        index=X.index,
        columns=["y_hat_wait", "y_hat_retry", "y_hat_retry_nudge", "y_hat_escalate"],
    )

    fold_record: list[dict] = []

    for fold_i, (train_idx, val_idx, groups) in enumerate(
        create_group_kfold_splits(n_splits=n_splits)
    ):
        # Verify customer-level separation
        train_cust = set(groups.iloc[train_idx].unique())
        val_cust = set(groups.iloc[val_idx].unique())
        overlap = train_cust & val_cust
        if overlap:
            raise RuntimeError(
                f"Fold {fold_i}: customer overlap detected: {overlap}. "
                "GroupKFold integrity violated."
            )

        # Fit a FRESH TLearner on training rows ONLY
        X_train = X.iloc[train_idx]
        T_train = T.iloc[train_idx]
        Y_train = Y.iloc[train_idx]

        learner = TLearner(
            classifier_factory=classifier_factory, **factory_kwargs
        )
        learner.fit(X_train, T_train, Y_train)

        # Predict potential outcomes on validation rows ONLY
        X_val = X.iloc[val_idx]
        po_val = estimate_potential_outcomes(learner, X_val)

        # Store OOF predictions at the correct indices
        oof_po.iloc[val_idx] = po_val.values

        fold_record.append({
            "fold": fold_i,
            "train_rows": len(train_idx),
            "val_rows": len(val_idx),
            "train_customers": len(train_cust),
            "val_customers": len(val_cust),
            "customer_overlap": len(overlap),
        })

    # Verify full coverage — every row should have predictions
    missing_mask = oof_po.isna().any(axis=1)
    n_missing = int(missing_mask.sum())
    if n_missing > 0:
        raise RuntimeError(
            f"{n_missing} rows have no OOF predictions. "
            "Not all rows were covered by the GroupKFold splits."
        )

    # Compute treatment effects from OOF potential outcomes
    oof_te = pd.DataFrame(
        {
            "tau_retry": oof_po["y_hat_retry"] - oof_po["y_hat_wait"],
            "tau_nudge": oof_po["y_hat_retry_nudge"] - oof_po["y_hat_wait"],
            "tau_escalate": oof_po["y_hat_escalate"] - oof_po["y_hat_wait"],
        },
        index=oof_po.index,
    )

    return oof_po, oof_te


# ---------------------------------------------------------------------------
# Day 3 ground-truth reference values (sanity check ONLY)
# ---------------------------------------------------------------------------

DAY3_REFERENCE = {
    "tau_retry": {"mean_pp": 10.7, "std_pp": 12.8},
    "tau_nudge": {"mean_pp": 9.2, "std_pp": 14.0},
    "tau_escalate": {"mean_pp": 0.2, "std_pp": 5.4},
}


def _flat_uplift_check(
    effect_name: str, estimated_std: float, day3_std_pp: float
) -> str:
    """Check whether estimated std is substantially flatter than Day 3 reference.

    Returns a human-readable assessment string.
    """
    estimated_std_pp = estimated_std * 100
    ratio = estimated_std_pp / day3_std_pp if day3_std_pp > 0 else float("inf")

    if ratio < 0.15:
        return (
            f"WARNING: estimated treatment effect for {effect_name} shows the "
            f"no-learnable-heterogeneity / flat-uplift failure mode. "
            f"Estimated std = {estimated_std_pp:.2f} pp vs Day 3 ref = {day3_std_pp:.1f} pp "
            f"(ratio = {ratio:.2f})."
        )
    elif ratio < 0.40:
        return (
            f"CAUTION: {effect_name} distribution is substantially flatter than Day 3 reference. "
            f"Estimated std = {estimated_std_pp:.2f} pp vs Day 3 ref = {day3_std_pp:.1f} pp "
            f"(ratio = {ratio:.2f})."
        )
    else:
        return (
            f"OK: {effect_name} std is not materially flatter. "
            f"Estimated std = {estimated_std_pp:.2f} pp vs Day 3 ref = {day3_std_pp:.1f} pp "
            f"(ratio = {ratio:.2f})."
        )


# ---------------------------------------------------------------------------
# CLI — run OOF estimation and print report
# ---------------------------------------------------------------------------

def main() -> None:
    """Run OOF treatment-effect estimation and print the full 4G report."""
    sep = "=" * 75

    print(sep)
    print("  4G — Out-of-Fold Treatment Effect Estimation")
    print(sep)

    # ── Run OOF estimation ────────────────────────────────────────────────
    print("\nRunning 5-fold customer-level GroupKFold OOF estimation...")
    oof_po, oof_te = compute_oof_treatment_effects(n_splits=5)

    # ── A. OOF coverage ──────────────────────────────────────────────────
    # Recover payment_id from the training CSV for coverage check
    import pathlib
    _REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
    train_df = pd.read_csv(_REPO_ROOT / "ml" / "data" / "causal_training_data.csv")
    source_rows = len(train_df)

    print(f"\n{'-' * 75}")
    print("  A. OOF COVERAGE")
    print(f"{'-' * 75}")
    print(f"  Source rows           : {source_rows:,}")
    print(f"  OOF prediction rows   : {len(oof_te):,}")
    n_unique = len(oof_po)
    # Check for duplicate indices
    n_dupes = int(oof_po.index.duplicated().sum())
    n_missing = source_rows - n_unique
    print(f"  Unique row indices    : {n_unique:,}")
    print(f"  Duplicate indices     : {n_dupes}")
    print(f"  Missing rows          : {n_missing}")
    coverage_ok = (len(oof_te) == source_rows) and (n_dupes == 0)
    print(f"  Coverage              : {'PASS' if coverage_ok else 'FAIL'}")

    # ── B. Potential outcome summary ──────────────────────────────────────
    print(f"\n{'-' * 75}")
    print("  POTENTIAL OUTCOME SUMMARY (OOF predictions)")
    print(f"{'-' * 75}")
    for col in oof_po.columns:
        vals = oof_po[col]
        print(
            f"  {col:<22}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
            f"min={vals.min():.4f}  max={vals.max():.4f}"
        )
        assert vals.min() >= 0.0 and vals.max() <= 1.0, (
            f"{col} predictions out of [0,1] range"
        )

    # ── C. Treatment effect distribution ─────────────────────────────────
    print(f"\n{'-' * 75}")
    print("  B. TREATMENT EFFECT DISTRIBUTION (OOF)")
    print(f"{'-' * 75}")
    effect_names = ["tau_retry", "tau_nudge", "tau_escalate"]
    te_stats: dict[str, dict[str, float]] = {}

    for name in effect_names:
        vals = oof_te[name]
        stats = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
        te_stats[name] = stats
        mean_pp = stats["mean"] * 100
        std_pp = stats["std"] * 100
        min_pp = stats["min"] * 100
        max_pp = stats["max"] * 100
        print(f"\n  {name}:")
        print(f"    mean = {stats['mean']:.4f}  ({mean_pp:+.1f} pp)")
        print(f"    std  = {stats['std']:.4f}  ({std_pp:.1f} pp)")
        print(f"    min  = {stats['min']:.4f}  ({min_pp:+.1f} pp)")
        print(f"    max  = {stats['max']:.4f}  ({max_pp:+.1f} pp)")

    # ── D. Day 3 sanity comparison ───────────────────────────────────────
    print(f"\n{'-' * 75}")
    print("  C. DAY 3 SANITY COMPARISON")
    print(f"{'-' * 75}")
    print(f"\n  {'Effect':<16} | {'Est Mean':>10} | {'Day3 Mean':>10} | "
          f"{'Est Std':>10} | {'Day3 Std':>10}")
    print(f"  {'-' * 68}")

    any_flat = False
    flat_warnings: list[str] = []

    for name in effect_names:
        est_mean_pp = te_stats[name]["mean"] * 100
        est_std_pp = te_stats[name]["std"] * 100
        d3 = DAY3_REFERENCE[name]

        print(
            f"  {name:<16} | {est_mean_pp:>+9.1f}pp | {d3['mean_pp']:>+9.1f}pp | "
            f"{est_std_pp:>9.1f}pp | {d3['std_pp']:>9.1f}pp"
        )

    # ── E. Flat-uplift check ──────────────────────────────────────────────
    print(f"\n{'-' * 75}")
    print("  D. FLAT-UPLIFT CHECK")
    print(f"{'-' * 75}")

    for name in effect_names:
        d3 = DAY3_REFERENCE[name]
        assessment = _flat_uplift_check(name, te_stats[name]["std"], d3["std_pp"])
        print(f"\n  {assessment}")
        if "WARNING" in assessment:
            any_flat = True
            flat_warnings.append(assessment)
        elif "CAUTION" in assessment:
            flat_warnings.append(assessment)

    # ── F. Interpretation ────────────────────────────────────────────────
    print(f"\n{'-' * 75}")
    print("  E. INTERPRETATION")
    print(f"{'-' * 75}")

    for name in effect_names:
        est = te_stats[name]
        d3 = DAY3_REFERENCE[name]
        est_mean_pp = est["mean"] * 100
        est_std_pp = est["std"] * 100

        mean_diff = abs(est_mean_pp - d3["mean_pp"])
        std_ratio = est_std_pp / d3["std_pp"] if d3["std_pp"] > 0 else float("inf")

        notes = []
        # Mean direction
        if est_mean_pp > 0 and d3["mean_pp"] > 0:
            notes.append("correct positive direction")
        elif est_mean_pp < 0 and d3["mean_pp"] > 0:
            notes.append("WRONG SIGN vs Day 3 (estimated negative, Day 3 positive)")

        # Mean magnitude
        if mean_diff < 3.0:
            notes.append("mean magnitude broadly similar to Day 3")
        elif mean_diff < 8.0:
            notes.append("mean magnitude moderately different from Day 3")
        else:
            notes.append("mean magnitude materially different from Day 3")

        # Spread
        if std_ratio < 0.15:
            notes.append("SUBSTANTIALLY FLATTER than Day 3 (flat-uplift failure)")
        elif std_ratio < 0.40:
            notes.append("substantially flatter than Day 3")
        elif std_ratio < 0.70:
            notes.append("moderately compressed vs Day 3")
        elif std_ratio < 1.30:
            notes.append("spread broadly similar to Day 3")
        else:
            notes.append("more variable than Day 3")

        print(f"\n  {name}:")
        for note in notes:
            print(f"    - {note}")

    # ── Final governance ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  CAUSAL INTERPRETATION NOTICE")
    print(sep)
    print("  - tau values are T-Learner CATE estimates, NOT hidden ground truth.")
    print("  - Day 3 numbers are simulation sanity checks, NOT targets.")
    print("  - Good match does not prove causal validity on real data.")
    print("  - No hidden ground-truth features entered X.")
    print("  - No treatment policy or decision policy was implemented.")
    print("  - No existing policy/evaluation file was modified.")

    if any_flat:
        print(f"\n{'!' * 75}")
        print("  FLAT-UPLIFT WARNING(S) - DO NOT PROCEED TO 4H AUTOMATICALLY")
        print(f"{'!' * 75}")
        for w in flat_warnings:
            print(f"  {w}")
    else:
        print(f"\n  No flat-uplift failure mode detected.")

    print(sep)


if __name__ == "__main__":
    main()
