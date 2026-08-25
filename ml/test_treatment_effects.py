"""
ml/test_treatment_effects.py
=============================
Focused tests for ml/treatment_effects.py.

Tests 1-12 from the specification, using small in-memory fixtures
for unit tests and the approved dataset for integration tests.

No model is tuned or selected. No decision policy is implemented.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from ml.dataset import OBSERVABLE_FEATURES
from ml.t_learner import ARMS, TLearner
from ml.treatment_effects import (
    estimate_potential_outcomes,
    estimate_treatment_effects,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_data():
    """Small balanced dataset with all four arms."""
    rng = np.random.default_rng(99)
    n = 200

    X = pd.DataFrame({
        "amount": rng.uniform(10, 500, n),
        "attempt_number": rng.integers(1, 5, n),
        "dynamic_success_rate": rng.uniform(0, 1, n),
        "cumulative_failures": rng.integers(0, 10, n),
        "consecutive_failed_cycles": rng.integers(0, 5, n),
        "notification_engagement_score": rng.uniform(0, 1, n),
        "contact_response_score": rng.uniform(0, 1, n),
        "payment_method": rng.choice(["card", "bank_transfer"], n),
        "failure_reason": rng.choice(["insufficient_funds", "expired_card"], n),
    })
    arms = ["WAIT", "RETRY", "RETRY_NUDGE", "ESCALATE"]
    T = pd.Series([arms[i % 4] for i in range(n)], name="assigned_T")
    Y = pd.Series(rng.integers(0, 2, n), name="realized_Y")

    return X, T, Y


@pytest.fixture
def fitted_learner(small_data):
    X, T, Y = small_data
    learner = TLearner()
    learner.fit(X, T, Y)
    return learner


# ---------------------------------------------------------------------------
# Test 1: Potential outcomes contain all four arms
# ---------------------------------------------------------------------------

class TestPotentialOutcomes:
    def test_contains_all_four_arms(self, fitted_learner, small_data):
        X, _, _ = small_data
        po = estimate_potential_outcomes(fitted_learner, X)
        expected_cols = {"y_hat_wait", "y_hat_retry", "y_hat_retry_nudge", "y_hat_escalate"}
        assert set(po.columns) == expected_cols

    def test_row_count_matches_input(self, fitted_learner, small_data):
        """Test 5: output row count equals input row count."""
        X, _, _ = small_data
        po = estimate_potential_outcomes(fitted_learner, X)
        assert len(po) == len(X)

    def test_predictions_are_probabilities(self, fitted_learner, small_data):
        """Test 6: predictions are in [0, 1]."""
        X, _, _ = small_data
        po = estimate_potential_outcomes(fitted_learner, X)
        for col in po.columns:
            assert po[col].min() >= 0.0, f"{col} has values < 0"
            assert po[col].max() <= 1.0, f"{col} has values > 1"


# ---------------------------------------------------------------------------
# Test 2: Treatment effects computed as arm - wait
# ---------------------------------------------------------------------------

class TestTreatmentEffectCalculation:
    def test_tau_is_arm_minus_wait(self, fitted_learner, small_data):
        X, _, _ = small_data
        po = estimate_potential_outcomes(fitted_learner, X)
        te = estimate_treatment_effects(fitted_learner, X)

        np.testing.assert_array_almost_equal(
            te["tau_retry"].values,
            (po["y_hat_retry"] - po["y_hat_wait"]).values,
        )
        np.testing.assert_array_almost_equal(
            te["tau_nudge"].values,
            (po["y_hat_retry_nudge"] - po["y_hat_wait"]).values,
        )
        np.testing.assert_array_almost_equal(
            te["tau_escalate"].values,
            (po["y_hat_escalate"] - po["y_hat_wait"]).values,
        )


# ---------------------------------------------------------------------------
# Test 3: Negative tau values are preserved
# ---------------------------------------------------------------------------

class TestNegativeTau:
    def test_negative_values_preserved(self, fitted_learner, small_data):
        X, _, _ = small_data
        te = estimate_treatment_effects(fitted_learner, X)
        # At least one tau column should have SOME variation (may include negatives)
        # With random data, some negatives are expected
        all_positive = all((te[col] >= 0).all() for col in te.columns)
        # We can't guarantee negatives with random data, but we verify no clipping
        # by checking raw values are preserved exactly
        po = estimate_potential_outcomes(fitted_learner, X)
        raw_tau = po["y_hat_retry"] - po["y_hat_wait"]
        np.testing.assert_array_equal(te["tau_retry"].values, raw_tau.values)


# ---------------------------------------------------------------------------
# Test 4: No clipping/rescaling occurs
# ---------------------------------------------------------------------------

class TestNoClipping:
    def test_no_rescaling(self, fitted_learner, small_data):
        """tau values are raw differences, not rescaled."""
        X, _, _ = small_data
        po = estimate_potential_outcomes(fitted_learner, X)
        te = estimate_treatment_effects(fitted_learner, X)

        # Exact equality (not approximate) proves no transformation
        for arm_col, tau_col in [
            ("y_hat_retry", "tau_retry"),
            ("y_hat_retry_nudge", "tau_nudge"),
            ("y_hat_escalate", "tau_escalate"),
        ]:
            expected = po[arm_col].values - po["y_hat_wait"].values
            np.testing.assert_array_equal(te[tau_col].values, expected)


# ---------------------------------------------------------------------------
# Test 5: Output row count equals input
# ---------------------------------------------------------------------------

class TestRowCount:
    def test_te_row_count(self, fitted_learner, small_data):
        X, _, _ = small_data
        te = estimate_treatment_effects(fitted_learner, X)
        assert len(te) == len(X)


# ---------------------------------------------------------------------------
# Test 7: Synthetic known predictions produce exact tau
# ---------------------------------------------------------------------------

class TestSyntheticExact:
    def test_exact_tau_from_known_predictions(self):
        """Given known arm probabilities, verify exact tau values."""
        # Create a mock TLearner that returns known probabilities
        mock_learner = MagicMock(spec=TLearner)
        mock_learner._is_fitted = True

        X = pd.DataFrame({feat: [1.0] for feat in OBSERVABLE_FEATURES})

        # Known probabilities: WAIT=0.10, RETRY=0.25, NUDGE=0.20, ESCALATE=0.12
        def mock_arm_proba(X_in, arm):
            values = {
                "WAIT": np.array([0.10]),
                "RETRY": np.array([0.25]),
                "RETRY_NUDGE": np.array([0.20]),
                "ESCALATE": np.array([0.12]),
            }
            return values[arm]

        mock_learner.predict_arm_proba = mock_arm_proba

        po = estimate_potential_outcomes(mock_learner, X)
        te = estimate_treatment_effects(mock_learner, X)

        # Exact expected values
        assert po["y_hat_wait"].iloc[0] == pytest.approx(0.10)
        assert po["y_hat_retry"].iloc[0] == pytest.approx(0.25)
        assert po["y_hat_retry_nudge"].iloc[0] == pytest.approx(0.20)
        assert po["y_hat_escalate"].iloc[0] == pytest.approx(0.12)

        assert te["tau_retry"].iloc[0] == pytest.approx(0.15)     # 0.25 - 0.10
        assert te["tau_nudge"].iloc[0] == pytest.approx(0.10)     # 0.20 - 0.10
        assert te["tau_escalate"].iloc[0] == pytest.approx(0.02)  # 0.12 - 0.10


# ---------------------------------------------------------------------------
# Test 8-10: OOF integration tests (using real data)
# ---------------------------------------------------------------------------

class TestOOFIntegration:
    """Integration tests using the approved causal training dataset."""

    @pytest.fixture(autouse=True)
    def run_oof(self):
        """Run OOF once for all tests in this class."""
        from ml.treatment_effects import compute_oof_treatment_effects
        self.oof_po, self.oof_te = compute_oof_treatment_effects(n_splits=5)

    def test_oof_row_count(self):
        """Test 8: exactly one OOF prediction per payment."""
        assert len(self.oof_te) == 30_472

    def test_no_nan_in_oof(self):
        """Every row must have a prediction."""
        assert self.oof_po.isna().sum().sum() == 0
        assert self.oof_te.isna().sum().sum() == 0

    def test_oof_po_are_probabilities(self):
        """All potential outcomes in [0, 1]."""
        for col in self.oof_po.columns:
            assert self.oof_po[col].min() >= 0.0
            assert self.oof_po[col].max() <= 1.0

    def test_all_five_folds_represented(self):
        """Test 9: the OOF covers all 30,472 rows from 5 folds."""
        # If any fold were missing, we'd have NaN rows
        assert self.oof_po.isna().sum().sum() == 0

    def test_tau_computed_as_arm_minus_wait(self):
        """Verify the OOF tau values are exactly po_arm - po_wait."""
        np.testing.assert_array_almost_equal(
            self.oof_te["tau_retry"].values,
            (self.oof_po["y_hat_retry"] - self.oof_po["y_hat_wait"]).values,
        )


# ---------------------------------------------------------------------------
# Test 10: Customer overlap remains zero
# ---------------------------------------------------------------------------

class TestCustomerOverlap:
    def test_customer_overlap_zero_all_folds(self):
        """Verify GroupKFold customer-level separation during OOF."""
        from ml.splits import create_group_kfold_splits

        for fold_i, (train_idx, val_idx, groups) in enumerate(
            create_group_kfold_splits(n_splits=5)
        ):
            train_cust = set(groups.iloc[train_idx].unique())
            val_cust = set(groups.iloc[val_idx].unique())
            overlap = train_cust & val_cust
            assert overlap == set(), f"Fold {fold_i}: customer overlap = {overlap}"


# ---------------------------------------------------------------------------
# Test 11: No forbidden columns in prediction features
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_observable_features_only(self, fitted_learner, small_data):
        """X must have exactly OBSERVABLE_FEATURES — no hidden columns."""
        X, _, _ = small_data
        assert list(X.columns) == OBSERVABLE_FEATURES

    def test_wrong_features_rejected(self, fitted_learner):
        """A dataframe with forbidden columns must not pass."""
        X_bad = pd.DataFrame({
            feat: [1.0] for feat in OBSERVABLE_FEATURES
        })
        X_bad["retry_success_probability"] = 0.5
        with pytest.raises(AssertionError):
            estimate_potential_outcomes(fitted_learner, X_bad)


# ---------------------------------------------------------------------------
# Test 12: Fresh TLearner per fold
# ---------------------------------------------------------------------------

class TestFreshModelPerFold:
    def test_fresh_learner_each_fold(self):
        """Verify compute_oof_treatment_effects creates new TLearner per fold."""
        from ml.treatment_effects import compute_oof_treatment_effects
        from ml.t_learner import TLearner as RealTLearner

        created_learners = []
        original_init = RealTLearner.__init__

        def tracking_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            created_learners.append(id(self))

        with patch.object(RealTLearner, "__init__", tracking_init):
            compute_oof_treatment_effects(n_splits=5)

        # At least 5 distinct TLearner instances (one per fold)
        assert len(set(created_learners)) >= 5


# ---------------------------------------------------------------------------
# Test: flat-uplift checker
# ---------------------------------------------------------------------------

class TestFlatUpliftCheck:
    def test_very_flat_triggers_warning(self):
        from ml.treatment_effects import _flat_uplift_check

        result = _flat_uplift_check("tau_test", 0.001, 12.8)
        assert "WARNING" in result

    def test_moderate_flat_triggers_caution(self):
        from ml.treatment_effects import _flat_uplift_check

        result = _flat_uplift_check("tau_test", 0.03, 12.8)
        assert "CAUTION" in result

    def test_normal_std_is_ok(self):
        from ml.treatment_effects import _flat_uplift_check

        result = _flat_uplift_check("tau_test", 0.10, 12.8)
        assert "OK" in result
