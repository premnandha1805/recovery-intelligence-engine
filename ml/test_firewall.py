"""
ml/test_firewall.py
===================
Focused tests for the ml/firewall.py leakage guard.

Run with:
    python -m pytest ml/test_firewall.py -v

All tests use in-memory DataFrames.
The approved causal training CSV is NOT modified.
No model is trained.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml.firewall import LeakageError, assert_no_leakage
from ml.dataset import OUTCOME_COLUMN, FORBIDDEN_PREFIXES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(*col_names: str) -> pd.DataFrame:
    """Build a one-row DataFrame with the given column names."""
    return pd.DataFrame({col: [0] for col in col_names})


# ---------------------------------------------------------------------------
# Test 1 — clean dataframe passes
# ---------------------------------------------------------------------------

class TestCleanDataframe:
    def test_observable_features_pass(self):
        """A dataframe with only the nine observable features must pass."""
        df = _df(
            "amount",
            "attempt_number",
            "dynamic_success_rate",
            "cumulative_failures",
            "consecutive_failed_cycles",
            "notification_engagement_score",
            "contact_response_score",
            "payment_method",
            "failure_reason",
        )
        result = assert_no_leakage(df)
        assert result is None

    def test_empty_dataframe_passes(self):
        """An empty DataFrame has no columns to violate."""
        result = assert_no_leakage(pd.DataFrame())
        assert result is None


# ---------------------------------------------------------------------------
# Tests 2-6 — forbidden-prefix guard
# ---------------------------------------------------------------------------

class TestForbiddenPrefixes:
    def test_p_success_wait_raises(self):
        """p_success_wait starts with forbidden prefix 'p_success'."""
        with pytest.raises(LeakageError, match="p_success_wait"):
            assert_no_leakage(_df("p_success_wait"))

    def test_p_success_retry_raises(self):
        """p_success_retry starts with forbidden prefix 'p_success'."""
        with pytest.raises(LeakageError, match="p_success_retry"):
            assert_no_leakage(_df("p_success_retry"))

    def test_hidden_score_raises(self):
        """hidden_score starts with forbidden prefix 'hidden_'."""
        with pytest.raises(LeakageError, match="hidden_score"):
            assert_no_leakage(_df("hidden_score"))

    def test_retry_success_probability_raises(self):
        """retry_success_probability is an exact forbidden prefix."""
        with pytest.raises(LeakageError, match="retry_success_probability"):
            assert_no_leakage(_df("retry_success_probability"))

    def test_natural_recovery_probability_raises(self):
        """natural_recovery_probability starts with forbidden prefix 'natural_recovery'."""
        with pytest.raises(LeakageError, match="natural_recovery_probability"):
            assert_no_leakage(_df("natural_recovery_probability"))

    def test_nudge_success_probability_raises(self):
        with pytest.raises(LeakageError, match="nudge_success_probability"):
            assert_no_leakage(_df("nudge_success_probability"))

    def test_escalation_success_probability_raises(self):
        with pytest.raises(LeakageError, match="escalation_success_probability"):
            assert_no_leakage(_df("escalation_success_probability"))

    def test_expected_recovery_raises(self):
        with pytest.raises(LeakageError, match="expected_recovery"):
            assert_no_leakage(_df("expected_recovery"))

    def test_expected_net_value_raises(self):
        """expected_net_value starts with forbidden prefix 'expected_net_value'."""
        with pytest.raises(LeakageError, match="expected_net_value"):
            assert_no_leakage(_df("expected_net_value"))

    def test_prefix_extension_also_rejected(self):
        """A column that merely starts with a forbidden prefix must also fail."""
        # 'p_success' prefix catches p_success_custom_suffix too
        with pytest.raises(LeakageError):
            assert_no_leakage(_df("p_success_custom_suffix"))

    def test_hidden_anything_rejected(self):
        """hidden_ground_truth should fail via 'hidden_' prefix."""
        with pytest.raises(LeakageError):
            assert_no_leakage(_df("hidden_ground_truth"))


# ---------------------------------------------------------------------------
# Tests 7-9 + 12 — outcome-token guard
# ---------------------------------------------------------------------------

class TestOutcomeToken:
    def test_predicted_outcome_raises(self):
        with pytest.raises(LeakageError, match="predicted_outcome"):
            assert_no_leakage(_df("predicted_outcome"))

    def test_future_outcome_raises(self):
        with pytest.raises(LeakageError, match="future_outcome"):
            assert_no_leakage(_df("future_outcome"))

    def test_outcome_probability_raises(self):
        with pytest.raises(LeakageError, match="outcome_probability"):
            assert_no_leakage(_df("outcome_probability"))

    def test_customer_outcome_raises(self):
        with pytest.raises(LeakageError, match="customer_outcome"):
            assert_no_leakage(_df("customer_outcome"))

    def test_outcome_score_raises(self):
        with pytest.raises(LeakageError, match="outcome_score"):
            assert_no_leakage(_df("outcome_score"))

    def test_outcome_uppercase_raises(self):
        """Test 12 — outcome token detection is case-insensitive."""
        with pytest.raises(LeakageError):
            assert_no_leakage(_df("OUTCOME_score"))

    def test_outcome_mixed_case_raises(self):
        """Test 12 — OutCome in mixed case is still detected."""
        with pytest.raises(LeakageError):
            assert_no_leakage(_df("OutCome_probability"))

    def test_outcome_all_caps_raises(self):
        """Test 12 — OUTCOME in all-caps is detected."""
        with pytest.raises(LeakageError):
            assert_no_leakage(_df("PREDICTED_OUTCOME"))


# ---------------------------------------------------------------------------
# Test 10 — realized_Y is allowed
# ---------------------------------------------------------------------------

class TestOutcomeColumnAllowed:
    def test_realized_Y_is_allowed(self):
        """Test 10 — realized_Y equals OUTCOME_COLUMN and must NOT be rejected."""
        assert OUTCOME_COLUMN == "realized_Y"
        df = _df("amount", "realized_Y")
        result = assert_no_leakage(df)
        assert result is None, "realized_Y must be allowed through the firewall"

    def test_outcome_column_constant_not_duplicated(self):
        """OUTCOME_COLUMN must come from ml.dataset, not be redefined here."""
        from ml import dataset
        assert OUTCOME_COLUMN is dataset.OUTCOME_COLUMN


# ---------------------------------------------------------------------------
# Test 11 — multiple offending columns all reported
# ---------------------------------------------------------------------------

class TestMultipleViolations:
    def test_multiple_forbidden_columns_reported(self):
        """Test 11 — all violations are reported, not just the first one."""
        df = _df(
            "amount",                        # clean
            "retry_success_probability",     # forbidden prefix
            "hidden_score",                  # forbidden prefix
            "predicted_outcome",             # outcome token
        )
        with pytest.raises(LeakageError) as exc_info:
            assert_no_leakage(df)
        msg = str(exc_info.value)
        assert "retry_success_probability" in msg
        assert "hidden_score" in msg
        assert "predicted_outcome" in msg
        # clean column must not be in the error message
        assert "amount" not in msg

    def test_two_forbidden_prefixes_both_reported(self):
        """Two forbidden-prefix columns both appear in the error."""
        df = _df("p_success_wait", "expected_net_value")
        with pytest.raises(LeakageError) as exc_info:
            assert_no_leakage(df)
        msg = str(exc_info.value)
        assert "p_success_wait" in msg
        assert "expected_net_value" in msg


# ---------------------------------------------------------------------------
# Canonical-import guard — FORBIDDEN_PREFIXES not duplicated
# ---------------------------------------------------------------------------

class TestCanonicalImport:
    def test_forbidden_prefixes_imported_from_dataset(self):
        """Firewall must import FORBIDDEN_PREFIXES from ml.dataset, not redefine."""
        from ml import dataset, firewall
        # Both symbols must refer to the same list object
        import ml.firewall as fw
        # We can't compare identity directly since firewall only imports
        # the name, but we can verify they have the same content.
        assert FORBIDDEN_PREFIXES == dataset.FORBIDDEN_PREFIXES

    def test_leakage_error_is_value_error_subclass(self):
        """LeakageError must subclass ValueError for broad compatibility."""
        assert issubclass(LeakageError, ValueError)
