"""
policy/base.py — Abstract Policy base class.

Hard constraint
---------------
NO policy file may reference hidden ground-truth data.  This module enforces
that at *import time* for every concrete subclass: if the subclass module's
source text contains any forbidden token, an AssertionError is raised before
the class can be instantiated.

Forbidden identifiers (case-insensitive):
  • "p_success"       — column prefix used in hidden_ground_truth.csv
  • "hidden"          — any column/variable with this word
  • "ground_truth"    — the hidden file itself
  • "hidden_ground_truth"  — explicit filename

These checks are intentionally paranoid: they scan the *source file* of every
concrete Policy subclass at class-creation time via __init_subclass__ so a
mistake is caught the moment the module is imported, not at runtime.
"""

from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # keep imports clean

# ── Tokens that must never appear in a policy source file ──────────────────
_FORBIDDEN_PATTERNS: list[re.Pattern[str]] = [
    # Matches "p_success" standalone or as a prefix (e.g. p_success_retry).
    # \b at the start ensures we don't match inside longer words like "top_success".
    re.compile(r"\bp_success", re.IGNORECASE),
    re.compile(r"\bhidden_ground_truth\b", re.IGNORECASE),
    re.compile(r"ground_truth\.csv", re.IGNORECASE),
    re.compile(r"observable_dataset\.csv", re.IGNORECASE | re.DOTALL),  # wrong file too
]

# hidden is a valid Python builtin name, so we only flag it when used as a
# dict key / column name pattern (e.g. features["hidden_..."] or "hidden_x")
_FORBIDDEN_COLUMN_PATTERN = re.compile(r'["\']_?hidden[_\w]*["\']', re.IGNORECASE)


def _check_source_for_forbidden_tokens(cls: type) -> None:
    """
    Read the *source file* that defines *cls* and raise AssertionError if any
    forbidden hidden-data token is found.

    Uses ``inspect.getfile`` + a direct ``open()`` read rather than
    ``inspect.getsource`` because ``getsource`` can fail silently at
    ``__init_subclass__`` time (the linecache may not be populated yet).

    Called automatically by __init_subclass__ for every concrete Policy.
    """
    try:
        source_file = inspect.getfile(cls)
    except (OSError, TypeError):
        # Built-in / frozen — can't read source; skip.
        return

    try:
        with open(source_file, encoding="utf-8") as fh:
            source = fh.read()
    except OSError:
        return

    for pattern in _FORBIDDEN_PATTERNS:
        match = pattern.search(source)
        assert match is None, (
            f"[Policy guard] {cls.__name__!r} in {source_file!r} references "
            f"forbidden hidden-data token {match.group()!r}. "
            f"Policies may only use observable features. "
            f"Remove all references to p_success / hidden ground-truth columns."
        )

    match = _FORBIDDEN_COLUMN_PATTERN.search(source)
    assert match is None, (
        f"[Policy guard] {cls.__name__!r} in {source_file!r} references a column "
        f"name that looks like a hidden feature: {match.group()!r}. "
        f"Policies may only use observable features."
    )


class Policy(ABC):
    """
    Abstract base for all recovery-action policies.

    Subclass contract
    -----------------
    • Override :meth:`decide` to implement decision logic.
    • ``payment_features`` must be built **exclusively** from columns in
      ``observable_dataset.csv`` / ``payment_scenarios.csv``.
    • Any subclass whose source file contains a reference to a forbidden
      hidden-data token will raise an ``AssertionError`` at import time.

    Observable columns (as of Day 1 simulator):
        payment_id, customer_id, subscription_id, amount, payment_method,
        failure_reason, attempt_number, days_since_last_success,
        historical_success_rate, previous_failure_count,
        recent_consecutive_failures, previous_retry_success_rate,
        previous_nudge_response
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Only guard concrete (non-abstract) subclasses.
        if not getattr(cls, "__abstractmethods__", None):
            _check_source_for_forbidden_tokens(cls)

    @abstractmethod
    def decide(self, payment_features: dict) -> "Action":  # type: ignore[name-defined]
        """
        Choose a recovery action for a single failed payment.

        Parameters
        ----------
        payment_features : dict
            Feature dict built ONLY from observable columns.  Keys map
            directly to CSV column names.

        Returns
        -------
        Action
            One of Action.WAIT | RETRY | RETRY_NUDGE | ESCALATE | STOP.
        """
        ...
