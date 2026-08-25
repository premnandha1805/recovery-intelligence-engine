# Day 3 Frozen Reference Snapshot

**Branch captured:** `causal-uplift-modeling`  
**Commit captured:** `138c755`  
**Frozen on:** 2026-08-26

---

## Purpose

This directory is a **read-only reference snapshot** of the Day 3 baseline state.
It must never be modified or deleted. All Day 4+ work goes into the new `ml/` package.

---

## Contents

| Path (relative to repo root) | Frozen copy |
|---|---|
| `policy/__init__.py` | `policy/__init__.py` |
| `policy/base.py` | `policy/base.py` |
| `policy/cost_config.py` | `policy/cost_config.py` |
| `policy/nudge_policy.py` | `policy/nudge_policy.py` |
| `policy/retry_policy.py` | `policy/retry_policy.py` |
| `policy/rule_based_policy.py` | `policy/rule_based_policy.py` |
| `policy/run_policies.py` | `policy/run_policies.py` |
| `policy/wait_policy.py` | `policy/wait_policy.py` |
| `evaluation/output/baseline_report_seed42.csv` | `baseline_report_seed42.csv` |
| `evaluation/output/baseline_report_seed777.csv` | `baseline_report_seed777.csv` |

---

## Day 4+ Immutability Rules

The following **live** files are frozen as of the commit above.
They must **not** be modified by Day 4 work without explicit human approval:

- `policy/wait_policy.py`
- `policy/retry_policy.py`
- `policy/nudge_policy.py`
- `policy/rule_based_policy.py`
- `evaluation/evaluator.py`

If a genuine bug is found in any of these files, **stop work and report to the
project owner before making any change**. Do not silently fix them.

---

## Verification

To verify a live file has not drifted from this snapshot, run:

```bash
# Example: check wait_policy.py
diff policy/wait_policy.py evaluation/output/frozen_day3/policy/wait_policy.py
```

A clean diff means the file is unchanged. Any output means drift has occurred.
