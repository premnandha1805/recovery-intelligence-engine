"""
Verification script for the v2 dataset after regeneration.

Checks:
  1. New columns present in payment_scenarios.csv, absent from ground_truth.csv.
  2. Proxy columns are in [0.0, 1.0] with no NaN.
  3. consecutive_failed_cycles is non-negative with no NaN.
  4. Same-customer uniqueness: every attempt row for a given customer has
     the SAME notification_engagement_score and contact_response_score.
     (groupby customer_id → nunique == 1)
  5. Row count is within expected range.
  6. All four correlation requirements verified on the production dataset.
  7. Proxy columns not in ground_truth.csv (hidden file must stay untouched).

Run from project root:
    python scripts/verify_v2_dataset.py
"""

import sys
import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OBS_PATH = os.path.join(PROJECT_ROOT, "data", "v2", "payment_scenarios.csv")
GT_PATH  = os.path.join(PROJECT_ROOT, "data", "v2", "ground_truth.csv")

PROXY_COLS = ["notification_engagement_score", "contact_response_score"]
NEW_STATE_COLS = ["consecutive_failed_cycles"]
REQUIRED_OBS_COLS = [
    "payment_id", "billing_cycle_id", "customer_id", "subscription_id",
    "amount", "payment_method", "failure_reason", "attempt_number",
    "cumulative_failures", "consecutive_failed_cycles",
    "dynamic_success_rate", "days_active",
    "notification_engagement_score", "contact_response_score",
]

PASS = "[PASS]"
FAIL = "[FAIL]"
failures = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    print(f"  {status}  {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


print("=" * 65)
print("  V2 DATASET VERIFICATION")
print("=" * 65)

# ── Load files ────────────────────────────────────────────────────────────────
print("\n[1] Loading datasets...")
obs = pd.read_csv(OBS_PATH)
gt  = pd.read_csv(GT_PATH)
print(f"    payment_scenarios.csv: {len(obs):,} rows, {obs.shape[1]} columns")
print(f"    ground_truth.csv:      {len(gt):,}  rows, {gt.shape[1]} columns")
print(f"    Columns: {list(obs.columns)}")

# ── Check 1: Required columns present ────────────────────────────────────────
print("\n[2] Column presence checks...")
for col in REQUIRED_OBS_COLS:
    check(f"obs has '{col}'", col in obs.columns)

for col in PROXY_COLS:
    check(f"ground_truth does NOT have '{col}'", col not in gt.columns)

# ── Check 2: Range and NaN ────────────────────────────────────────────────────
print("\n[3] Range and NaN checks...")
for col in PROXY_COLS:
    if col in obs.columns:
        check(f"{col}: no NaN", obs[col].isna().sum() == 0,
              f"NaN count={obs[col].isna().sum()}")
        check(f"{col}: all in [0, 1]",
              (obs[col].min() >= 0.0) and (obs[col].max() <= 1.0),
              f"min={obs[col].min():.4f}, max={obs[col].max():.4f}")
    else:
        failures.append(f"{col} missing — skipping range check")

for col in NEW_STATE_COLS:
    if col in obs.columns:
        check(f"{col}: no NaN", obs[col].isna().sum() == 0)
        check(f"{col}: all >= 0", obs[col].min() >= 0,
              f"min={obs[col].min()}")
    else:
        failures.append(f"{col} missing — skipping range check")

# ── Check 3: Same-customer uniqueness for proxy scores ────────────────────────
print("\n[4] Same-customer uniqueness (proxy scores must be constant per customer)...")
for col in PROXY_COLS:
    if col in obs.columns:
        n_varied = (obs.groupby("customer_id")[col].nunique() > 1).sum()
        check(
            f"{col}: same score across all attempt rows per customer",
            n_varied == 0,
            f"{n_varied} customers have varying values",
        )

# ── Check 4: Row count in expected range ─────────────────────────────────────
print("\n[5] Row count check...")
check(
    f"Row count in expected range [25,000 – 50,000]",
    25_000 <= len(obs) <= 50_000,
    f"actual={len(obs):,}",
)
check(
    "payment_scenarios and ground_truth have same row count",
    len(obs) == len(gt),
    f"obs={len(obs):,}  gt={len(gt):,}",
)

# ── Check 5: Correlation requirements (on production dataset) ────────────────
print("\n[6] Correlation requirements (production dataset)...")
# Load hidden traits from ground_truth for correlation checks
# (this verification script may read hidden columns — it is never imported by policy/)
hidden_cols_gt = [c for c in gt.columns if c.startswith("_hidden_")]
if hidden_cols_gt:
    # Merge on payment_id to get hidden traits alongside proxy scores
    merged = obs[["payment_id", "customer_id"] + PROXY_COLS].merge(
        gt[["payment_id"] + hidden_cols_gt], on="payment_id"
    )
    # Deduplicate to one row per customer for correlation (proxies are per-customer)
    per_cust = merged.drop_duplicates("customer_id")

    nudge_resp  = per_cust["_hidden_nudge_responsiveness"].values  if "_hidden_nudge_responsiveness"  in per_cust else None
    retry_resp  = per_cust["_hidden_retry_responsiveness"].values  if "_hidden_retry_responsiveness"  in per_cust else None
    intrinsic   = per_cust["_hidden_intrinsic_recovery_prob"].values if "_hidden_intrinsic_recovery_prob" in per_cust else None

    if nudge_resp is not None and retry_resp is not None and intrinsic is not None:
        notif_score   = per_cust["notification_engagement_score"].values
        contact_score = per_cust["contact_response_score"].values
        tier_basic    = (obs.drop_duplicates("customer_id")["customer_tier"] == "basic").astype(float).values \
                        if "customer_tier" in obs.columns else np.zeros(len(per_cust))

        r_notif_nudge  = float(np.corrcoef(notif_score,   nudge_resp)[0, 1])
        r_notif_retry  = float(np.corrcoef(notif_score,   retry_resp)[0, 1])
        r_notif_intr   = float(np.corrcoef(notif_score,   intrinsic) [0, 1])

        r_contact_retry = float(np.corrcoef(contact_score, retry_resp)[0, 1])
        r_contact_nudge = float(np.corrcoef(contact_score, nudge_resp)[0, 1])
        r_contact_intr  = float(np.corrcoef(contact_score, intrinsic) [0, 1])

        print(f"    notification_engagement_score:")
        print(f"      r(intended=nudge_resp) = {r_notif_nudge:+.4f}")
        print(f"      r(other=retry_resp)    = {r_notif_retry:+.4f}")
        print(f"      r(intrinsic_recovery)  = {r_notif_intr:+.4f}")

        print(f"    contact_response_score:")
        print(f"      r(intended=retry_resp) = {r_contact_retry:+.4f}")
        print(f"      r(other=nudge_resp)    = {r_contact_nudge:+.4f}")
        print(f"      r(intrinsic_recovery)  = {r_contact_intr:+.4f}")

        check("notif: r(nudge_resp) in [0.40, 0.65]",   0.40 <= r_notif_nudge  <= 0.65, f"r={r_notif_nudge:+.4f}")
        check("notif: r(retry_resp) < r(nudge_resp)/3", abs(r_notif_retry) < abs(r_notif_nudge) / 3, f"ratio={abs(r_notif_nudge)/max(abs(r_notif_retry),1e-9):.2f}x")
        check("notif: r(intrinsic) < 0.30",             abs(r_notif_intr)  < 0.30, f"r={r_notif_intr:+.4f}")

        check("contact: r(retry_resp) in [0.40, 0.65]",  0.40 <= r_contact_retry <= 0.65, f"r={r_contact_retry:+.4f}")
        check("contact: r(nudge_resp) < r(retry_resp)/3", abs(r_contact_nudge) < abs(r_contact_retry) / 3, f"ratio={abs(r_contact_retry)/max(abs(r_contact_nudge),1e-9):.2f}x")
        check("contact: r(intrinsic) < 0.30",             abs(r_contact_intr)  < 0.30, f"r={r_contact_intr:+.4f}")
    else:
        print("    [SKIP] Hidden responsiveness columns not found in ground_truth.csv")
else:
    print("    [SKIP] No _hidden_ columns in ground_truth.csv")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
if failures:
    print(f"  RESULT: FAILED — {len(failures)} check(s) failed:")
    for f in failures:
        print(f"    - {f}")
    sys.exit(1)
else:
    print("  RESULT: ALL CHECKS PASSED")
print("=" * 65)
