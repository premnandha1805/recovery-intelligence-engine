"""
ml/firewall.py
==============
Independent leakage firewall for the ml/ pipeline boundary.

Purpose
-------
Provides a second, independent guard (complementing ml/dataset.py) that
fails loudly if hidden simulator or ground-truth information accidentally
enters any training or prediction dataframe.

Usage contract
--------------
Every future ml/ entrypoint that receives a dataframe MUST call::

    from ml.firewall import assert_no_leakage
    assert_no_leakage(df)          # first line, before ANY processing

This applies to (at minimum):
    ml/train.py   -- call before feature encoding or model fitting
    ml/predict.py -- call before inference or feature selection

The firewall does NOT modify, sanitize, or repair the dataframe.
Its job is detection and loud failure only.

Canonical definitions
---------------------
FORBIDDEN_PREFIXES and OUTCOME_COLUMN are imported from ml.dataset.
They must NOT be duplicated here.
"""

from __future__ import annotations

import pandas as pd

from ml.dataset import FORBIDDEN_PREFIXES, OUTCOME_COLUMN


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Raise LeakageError if any column in *df* violates the leakage contract.

    Two independent checks are applied to every column name:

    A. Forbidden-prefix guard
        Any column whose name equals or starts with an entry in
        FORBIDDEN_PREFIXES is rejected.  This catches hidden simulator
        columns such as ``retry_success_probability``, ``p_success_wait``,
        ``natural_recovery_probability``, ``expected_net_value``, etc.

    B. Outcome-token guard
        Any column whose name contains the substring ``"outcome"``
        (case-insensitive) is rejected, EXCEPT for the declared
        OUTCOME_COLUMN (``realized_Y``).  This prevents columns such as
        ``predicted_outcome`` or ``outcome_probability`` from entering
        the feature matrix.

    Parameters
    ----------
    df:
        Any pandas DataFrame to be validated before ML processing.

    Returns
    -------
    None
        Returned only if the dataframe is clean.

    Raises
    ------
    LeakageError
        On the first batch of violations found.  The error message
        identifies every offending column and the reason it is forbidden.
    """
    violations: list[str] = []

    for col in df.columns:
        # -- A. Forbidden-prefix guard ------------------------------------
        for prefix in FORBIDDEN_PREFIXES:
            if col == prefix or col.startswith(prefix):
                violations.append(
                    f"  column {col!r} matches forbidden prefix {prefix!r}"
                )
                break  # one violation message per column is enough

        else:
            # -- B. Outcome-token guard (only if prefix guard didn't fire) -
            if col != OUTCOME_COLUMN and "outcome" in col.lower():
                violations.append(
                    f"  column {col!r} contains forbidden token 'outcome'"
                    f" (only the declared OUTCOME_COLUMN {OUTCOME_COLUMN!r} is allowed)"
                )

    if violations:
        raise LeakageError(
            "Leakage detected -- the following columns must not enter the ML pipeline:\n"
            + "\n".join(violations)
        )

    return None


class LeakageError(ValueError):
    """Raised when assert_no_leakage() detects a forbidden column.

    Subclasses ValueError so existing ``except ValueError`` handlers still
    catch it, while allowing callers to target leakage specifically via
    ``except LeakageError``.
    """
