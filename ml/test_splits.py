"""
ml/test_splits.py
=================
Focused tests for ml/splits.py customer-level GroupKFold infrastructure.

All unit tests use small in-memory fixtures.
Integration tests use the approved causal training dataset.
No model is trained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.splits import create_group_kfold_splits, load_customer_groups


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_obs(tmp_path):
    """Small observable dataset: 3 customers, 2 payments each."""
    obs = pd.DataFrame({
        "payment_id":   ["p1", "p2", "p3", "p4", "p5", "p6"],
        "customer_id":  ["cA", "cA", "cB", "cB", "cC", "cC"],
    })
    path = tmp_path / "payment_scenarios.csv"
    obs.to_csv(path, index=False)
    return path


@pytest.fixture
def tiny_train(tmp_path, tiny_obs):
    """Matching causal training CSV for the tiny fixture."""
    train = pd.DataFrame({
        "payment_id":  ["p1", "p2", "p3", "p4", "p5", "p6"],
        "amount":      [10, 20, 30, 40, 50, 60],
        "assigned_T":  ["WAIT", "RETRY", "WAIT", "RETRY_NUDGE", "ESCALATE", "WAIT"],
        "realized_Y":  [0, 1, 0, 0, 1, 0],
    })
    path = tmp_path / "causal_training_data.csv"
    train.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Unit tests — small in-memory fixtures
# ---------------------------------------------------------------------------

class TestLoadCustomerGroups:
    def test_groups_length_matches_training_rows(self, tiny_train, tiny_obs):
        groups = load_customer_groups(tiny_train, tiny_obs)
        assert len(groups) == 6

    def test_groups_values_are_customer_ids(self, tiny_train, tiny_obs):
        groups = load_customer_groups(tiny_train, tiny_obs)
        assert set(groups.unique()) == {"cA", "cB", "cC"}

    def test_groups_alignment(self, tiny_train, tiny_obs):
        """Row order must match training data row order."""
        groups = load_customer_groups(tiny_train, tiny_obs)
        expected = ["cA", "cA", "cB", "cB", "cC", "cC"]
        assert list(groups) == expected

    def test_missing_training_file_raises(self, tmp_path, tiny_obs):
        with pytest.raises(FileNotFoundError):
            load_customer_groups(tmp_path / "nonexistent.csv", tiny_obs)

    def test_missing_obs_file_raises(self, tmp_path, tiny_train):
        with pytest.raises(FileNotFoundError):
            load_customer_groups(tiny_train, tmp_path / "nonexistent.csv")

    def test_unmatched_payment_id_raises(self, tmp_path):
        """A payment_id in training with no obs row must raise, not silently drop."""
        obs = pd.DataFrame({
            "payment_id":  ["p1", "p2"],
            "customer_id": ["cA", "cA"],
        })
        obs_path = tmp_path / "obs.csv"
        obs.to_csv(obs_path, index=False)

        train = pd.DataFrame({
            "payment_id": ["p1", "p2", "p99"],   # p99 has no match
            "amount":     [10, 20, 30],
        })
        train_path = tmp_path / "train.csv"
        train.to_csv(train_path, index=False)

        with pytest.raises(ValueError, match="customer_id"):
            load_customer_groups(train_path, obs_path)


class TestGroupKFoldSplits:
    """Test 1-6 from the specification."""

    def _run_splits(self, tiny_train, tiny_obs, n_splits=3):
        return list(create_group_kfold_splits(
            n_splits=n_splits,
            training_data_path=tiny_train,
            obs_path=tiny_obs,
        ))

    def test_number_of_folds_correct(self, tiny_train, tiny_obs):
        """Test 3 — number of folds == n_splits."""
        splits = self._run_splits(tiny_train, tiny_obs, n_splits=3)
        assert len(splits) == 3

    def test_no_customer_in_both_train_and_val(self, tiny_train, tiny_obs):
        """Test 1 — same customer never in train AND val."""
        splits = self._run_splits(tiny_train, tiny_obs, n_splits=3)
        for fold_i, (train_idx, val_idx, groups) in enumerate(splits):
            train_cust = set(groups.iloc[train_idx].unique())
            val_cust   = set(groups.iloc[val_idx].unique())
            overlap    = train_cust & val_cust
            assert overlap == set(), (
                f"Fold {fold_i}: customer overlap detected: {overlap}"
            )

    def test_no_index_overlap(self, tiny_train, tiny_obs):
        """Test 2 — train_indices ∩ val_indices == empty."""
        splits = self._run_splits(tiny_train, tiny_obs, n_splits=3)
        for fold_i, (train_idx, val_idx, _) in enumerate(splits):
            overlap = set(train_idx) & set(val_idx)
            assert overlap == set(), (
                f"Fold {fold_i}: row-index overlap detected: {overlap}"
            )

    def test_every_row_validated_exactly_once(self, tiny_train, tiny_obs):
        """Test 4 — every row appears in validation exactly once."""
        splits = self._run_splits(tiny_train, tiny_obs, n_splits=3)
        all_val_indices = []
        for _, val_idx, _ in splits:
            all_val_indices.extend(val_idx.tolist())
        assert sorted(all_val_indices) == list(range(6))

    def test_groups_argument_uses_customer_id(self, tiny_train, tiny_obs):
        """Test 5 — the yielded groups series contains customer_id values."""
        splits = self._run_splits(tiny_train, tiny_obs, n_splits=3)
        _, _, groups = splits[0]
        assert set(groups.unique()) == {"cA", "cB", "cC"}
        # Verify it is NOT payment-level (which would be 6 unique values of p1-p6)
        assert len(groups.unique()) == 3

    def test_no_random_splitting(self, tiny_train, tiny_obs):
        """Test 6 — calling twice gives identical results (deterministic, not random)."""
        splits1 = self._run_splits(tiny_train, tiny_obs, n_splits=3)
        splits2 = self._run_splits(tiny_train, tiny_obs, n_splits=3)
        for (t1, v1, _), (t2, v2, _) in zip(splits1, splits2):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(v1, v2)


# ---------------------------------------------------------------------------
# Integration tests — actual approved causal training dataset
# ---------------------------------------------------------------------------

class TestIntegrationRealDataset:
    """Run against the real approved Day 4B causal training data."""

    @pytest.fixture(autouse=True, scope="class")
    def real_splits(self):
        pytest.importorskip("sklearn")
        return list(create_group_kfold_splits(n_splits=5))

    def test_five_folds_produced(self, real_splits):
        assert len(real_splits) == 5

    def test_no_customer_overlap_any_fold(self, real_splits):
        for fold_i, (train_idx, val_idx, groups) in enumerate(real_splits):
            train_cust = set(groups.iloc[train_idx].unique())
            val_cust   = set(groups.iloc[val_idx].unique())
            overlap    = train_cust & val_cust
            assert overlap == set(), f"Fold {fold_i}: overlap = {overlap}"

    def test_no_row_index_overlap_any_fold(self, real_splits):
        for fold_i, (train_idx, val_idx, _) in enumerate(real_splits):
            overlap = set(train_idx) & set(val_idx)
            assert overlap == set(), f"Fold {fold_i}: row overlap = {overlap}"

    def test_total_rows_is_30472(self, real_splits):
        # All validation indices together should cover every row exactly once
        all_val = []
        for _, val_idx, _ in real_splits:
            all_val.extend(val_idx.tolist())
        assert len(all_val) == 30472
        assert sorted(all_val) == list(range(30472))

    def test_groups_are_customer_ids_not_payment_ids(self, real_splits):
        _, _, groups = real_splits[0]
        # customer universe is 2502, payment universe is 30472
        assert groups.nunique() == 2502
