"""
policy/cost_config.py — Single source of truth for intervention costs (INR).

Every file that needs a cost imports from here.  No other file may
hardcode a rupee amount for an action.

Cost rationale
--------------
WAIT        :   0  — passive; no resource consumed
RETRY       :   5  — automated payment retry (gateway fee)
RETRY_NUDGE :  15  — retry + dunning notification (SMS/email)
ESCALATE    : 250  — human agent or high-value outreach channel
STOP        :   0  — mark as unrecoverable; no action taken
"""

from models.schemas import Action

# ── Intervention costs in Indian Rupees (INR) ─────────────────────────────
ACTION_COSTS: dict[Action, float] = {
    Action.WAIT:        0.0,
    Action.RETRY:       5.0,
    Action.RETRY_NUDGE: 15.0,
    Action.ESCALATE:    250.0,
    Action.STOP:        0.0,
}

# Sanity check: every Action must have a cost entry (catches future enum additions)
_missing = [a for a in Action if a not in ACTION_COSTS]
assert not _missing, (
    f"ACTION_COSTS is missing cost entries for: {_missing}. "
    f"Add them to policy/cost_config.py."
)
