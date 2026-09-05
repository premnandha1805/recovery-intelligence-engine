"""
ml/evaluation/run_calibration_analysis.py
===========================================
Probability Calibration Analysis for T-Learner Treatment Arms.

Evaluates out-of-fold probability calibration for:
  1. WAIT
  2. RETRY
  3. RETRY_NUDGE
  4. ESCALATE

Calculates for each arm:
  - Brier Score
  - Calibration Intercept (alpha) & Calibration Slope (beta) via logit-logistic regression
  - Mean Predicted Probability vs. Observed Outcome Rate
  - Decile Binned Calibration Table (10 bins)

Explanatory analysis ONLY — does NOT modify models, evaluators, or policies.
Does NOT fit calibration curves (no Platt scaling / isotonic regression applied).
"""

from __future__ import annotations

import pathlib
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

# Project root setup
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.dataset import load_causal_dataset
from ml.splits import create_group_kfold_splits
from ml.t_learner import ARMS, TLearner

ML_EVAL_DIR = _REPO_ROOT / "ml" / "evaluation"
TRAINING_DATA_PATH = _REPO_ROOT / "ml" / "data" / "causal_training_data.csv"


def logit(p: np.ndarray) -> np.ndarray:
    """Compute logit transform of probabilities with clipping."""
    p_clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p_clipped / (1 - p_clipped))


def compute_oof_arm_predictions() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute GroupKFold out-of-fold true outcomes and predicted probabilities per arm."""
    X, T, Y = load_causal_dataset(TRAINING_DATA_PATH)

    oof_y_true: dict[str, list[float]] = {arm: [] for arm in ARMS}
    oof_y_prob: dict[str, list[float]] = {arm: [] for arm in ARMS}

    for train_idx, val_idx, _groups in create_group_kfold_splits(training_data_path=TRAINING_DATA_PATH):
        X_train, T_train, Y_train = X.iloc[train_idx], T.iloc[train_idx], Y.iloc[train_idx]
        X_val, T_val, Y_val = X.iloc[val_idx], T.iloc[val_idx], Y.iloc[val_idx]

        learner = TLearner()
        learner.fit(X_train, T_train, Y_train)

        for arm in ARMS:
            val_arm_mask = T_val == arm
            if val_arm_mask.sum() > 0:
                X_val_arm = X_val.loc[val_arm_mask]
                Y_val_arm = Y_val.loc[val_arm_mask]
                proba = learner.predict_arm_proba(X_val_arm, arm)

                oof_y_true[arm].extend(Y_val_arm.values.tolist())
                oof_y_prob[arm].extend(proba.tolist())

    return (
        {arm: np.array(oof_y_true[arm]) for arm in ARMS},
        {arm: np.array(oof_y_prob[arm]) for arm in ARMS},
    )


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute calibration intercept (alpha), calibration slope (beta), Brier score, and AUC."""
    brier = brier_score_loss(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    mean_pred = float(np.mean(y_prob))
    mean_obs = float(np.mean(y_true))

    # Logit-logistic regression for calibration intercept and slope
    # logit(P(Y=1)) = alpha + beta * logit(p_pred)
    l_pred = logit(y_prob).reshape(-1, 1)

    # Fit unpenalized logistic regression
    lr = LogisticRegression(C=1e9, solver="lbfgs")
    lr.fit(l_pred, y_true)

    intercept = float(lr.intercept_[0])
    slope = float(lr.coef_[0][0])

    return {
        "auc": auc,
        "brier": brier,
        "mean_pred_prob": mean_pred,
        "observed_rate": mean_obs,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def compute_decile_calibration_table(y_true: np.ndarray, y_prob: np.ndarray, arm_name: str) -> pd.DataFrame:
    """Compute 10-decile binned calibration table for an arm."""
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["decile"] = pd.qcut(df["y_prob"], q=10, duplicates="drop", labels=False) + 1

    decile_rows = []
    for decile_id, group in df.groupby("decile"):
        decile_rows.append({
            "Arm": arm_name,
            "Decile": int(decile_id),
            "Count": len(group),
            "Min Prob": float(group["y_prob"].min()),
            "Max Prob": float(group["y_prob"].max()),
            "Mean Predicted Prob": float(group["y_prob"].mean()),
            "Observed Outcome Rate": float(group["y_true"].mean()),
            "Calibration Gap": float(group["y_prob"].mean() - group["y_true"].mean()),
        })

    return pd.DataFrame(decile_rows)


def main():
    print("=" * 80)
    print("  OUT-OF-FOLD PROBABILITY CALIBRATION ANALYSIS FOR T-LEARNER ARMS")
    print("=" * 80)

    print("\nComputing GroupKFold Out-of-Fold predictions for all 4 arms...")
    oof_true, oof_prob = compute_oof_arm_predictions()

    summary_rows = []
    decile_tables = []

    for arm in ARMS:
        y_t = oof_true[arm]
        y_p = oof_prob[arm]

        metrics = compute_calibration_metrics(y_t, y_p)
        summary_rows.append({
            "Arm": arm,
            "Count": len(y_t),
            "ROC AUC": f"{metrics['auc']:.4f}",
            "Brier Score": f"{metrics['brier']:.4f}",
            "Mean Pred Prob": f"{metrics['mean_pred_prob']*100:.2f}%",
            "Observed Rate": f"{metrics['observed_rate']*100:.2f}%",
            "Calib Intercept (alpha)": f"{metrics['calibration_intercept']:+.4f}",
            "Calib Slope (beta)": f"{metrics['calibration_slope']:.4f}",
        })

        dec_df = compute_decile_calibration_table(y_t, y_p, arm)
        decile_tables.append(dec_df)

    # Print Summary Table
    print("\n" + "=" * 80)
    print("  TABLE 1: ARM-LEVEL PROBABILITY CALIBRATION & DISCRIMINATION METRICS")
    print("=" * 80)
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    # Print Decile Calibration Tables for each arm
    full_decile_df = pd.concat(decile_tables, ignore_index=True)

    for arm in ARMS:
        print("\n" + "=" * 80)
        print(f"  DECILE CALIBRATION CURVE TABLE: {arm}")
        print("=" * 80)
        arm_dec = full_decile_df[full_decile_df["Arm"] == arm].drop(columns=["Arm"])
        print(arm_dec.to_string(index=False))

    # Save artifacts
    out_summary_csv = ML_EVAL_DIR / "t_learner_calibration_summary.csv"
    out_deciles_csv = ML_EVAL_DIR / "t_learner_calibration_deciles.csv"

    df_summary.to_csv(out_summary_csv, index=False)
    full_decile_df.to_csv(out_deciles_csv, index=False)

    print(f"\nSaved calibration analysis artifacts:")
    print(f"  - {out_summary_csv}")
    print(f"  - {out_deciles_csv}")


if __name__ == "__main__":
    main()
