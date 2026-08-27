"""
decision_engine/run_day6d_evaluation.py
=======================================
End-to-end evaluation runner for Day 6D.
Executes the compiled LangGraph workflow on 20 payments from Seed 777 evaluation set,
demonstrating:
  - Case A: Normal pass through guardrails
  - Case B: Authoritative guardrail override
  - Case C: Context error short-circuit
  - SQLite audit persistence with true UPSERT verification
"""

from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import sys
import pandas as pd
import dotenv

# Load environment credentials
dotenv.load_dotenv()

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from decision_engine.graph import create_recovery_graph
from decision_engine.audit import (
    DEFAULT_AUDIT_DB_PATH,
    init_audit_db,
    get_audit_records_count_for_payment,
    get_audit_row_count,
)

SEED777_PATH = _REPO_ROOT / "ml" / "evaluation" / "seed777_data" / "payment_scenarios.csv"


def main():
    print("=" * 80)
    print("DAY 6D — 20-PAYMENT END-TO-END EVALUATION & SQLITE AUDIT RUN")
    print("=" * 80)

    # 1. Load Seed 777 dataset
    print(f"Loading Seed 777 evaluation set from: {SEED777_PATH}")
    seed777_df = pd.read_csv(SEED777_PATH)
    print(f"Loaded {len(seed777_df)} rows.")

    # Select 20 distinct payments
    sample_pids = seed777_df["payment_id"].head(20).tolist()

    # Ensure clean audit db initialization
    init_audit_db()

    # Build compiled recovery graph using default Foundry model and canonical policy
    graph = create_recovery_graph(dataset=seed777_df)

    results = []
    case_a_trace = None
    case_b_trace = None
    case_c_trace = None

    print("\nExecuting 20 end-to-end payments through the LangGraph workflow...")

    for idx, pid in enumerate(sample_pids, start=1):
        # Case B: On Payment #2, inject caps to guarantee authoritative guardrail override
        if idx == 2:
            input_state = {
                "payment_id": pid,
                "payment_context": {"retry_count_current_cycle": 3},
                "customer_history": {
                    "lifetime_escalations": 1,
                    "interventions_last_7_days": 2,
                },
                "audit_trail": [],
            }
            output = graph.invoke(input_state)
            case_b_trace = output
            case_label = "Case B (Override Demo)"
        elif idx == 20:
            # Case C: On Payment #20, pass invalid payment_id to test error short-circuit
            input_state = {
                "payment_id": "pay_invalid_missing_seed777_999",
                "audit_trail": [],
            }
            output = graph.invoke(input_state)
            case_c_trace = output
            case_label = "Case C (Error Short-Circuit)"
        else:
            # Case A: Normal pass
            input_state = {
                "payment_id": pid,
                "audit_trail": [],
            }
            output = graph.invoke(input_state)
            if idx == 1:
                case_a_trace = output
            case_label = "Case A (Normal)"

        llm_prop = output.get("llm_decision", {}).get("decision", "N/A")
        guard_status = output.get("guardrail_result", {}).get("status", "N/A")
        final_act = output.get("final_action", "WAIT")
        err = output.get("error")

        print(
            f"[{idx:02d}/20] {output['payment_id']} | Type: {case_label:<25} | "
            f"LLM: {llm_prop:<12} | Guardrail: {guard_status:<10} | Final: {final_act:<12}"
        )

        results.append(
            {
                "idx": idx,
                "payment_id": output["payment_id"],
                "type": case_label,
                "llm_decision": llm_prop,
                "guardrail_status": guard_status,
                "final_action": final_act,
                "error": err,
            }
        )

    # ── SECTION 11: FULL TRACES FOR CASES A, B, C ────────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 11: THREE FULL STATE & AUDIT TRACES")
    print("=" * 80)

    def print_trace(title: str, state: dict):
        print(f"\n--- {title} ---")
        print(f"Payment ID: {state.get('payment_id')}")
        print(f"Error: {state.get('error')}")
        print(f"Observable Features Summary: amount={state.get('observable_features', {}).get('amount')}, method={state.get('observable_features', {}).get('payment_method')}")
        print(f"Arm Probabilities: {state.get('arm_probabilities')}")
        print(f"Arm Net Values: {state.get('arm_net_values')}")
        print(f"LLM Proposed Decision: {state.get('llm_decision', {}).get('decision')}")
        print(f"LLM Confidence: {state.get('llm_decision', {}).get('confidence')}")
        print(f"LLM Reasoning: {state.get('llm_decision', {}).get('reasoning')}")
        print(f"Guardrail Verdict: {state.get('guardrail_result', {}).get('status')}")
        print(f"Guardrail Reason: {state.get('guardrail_result', {}).get('reason')}")
        print(f"FINAL ACTION: {state.get('final_action')}")
        print("Audit Trail Events:")
        for ev in state.get("audit_trail", []):
            print(f"  • [{ev.get('node')}] status={ev.get('status')}, details={ev}")

    print_trace("CASE A — NORMAL PASS", case_a_trace)
    print_trace("CASE B — GUARDRAIL OVERRIDE", case_b_trace)
    print_trace("CASE C — ERROR SHORT-CIRCUIT", case_c_trace)

    # ── SECTION 12: UPSERT VERIFICATION ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("SECTION 12: SQLITE UPSERT VERIFICATION")
    print("=" * 80)

    test_payment_id = "pay_000001_a1"
    db_path = DEFAULT_AUDIT_DB_PATH

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM decision_audit WHERE payment_id = ?", (test_payment_id,))
        count_before = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM decision_audit")
        total_before = cur.fetchone()[0]

    print(f"Row count for {test_payment_id} before second run: {count_before}")
    print(f"Total audit table row count before second run: {total_before}")

    # Re-run graph with identical payment_id
    print(f"Re-running graph with payment_id: {test_payment_id} ...")
    graph.invoke({"payment_id": test_payment_id, "audit_trail": []})

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM decision_audit WHERE payment_id = ?", (test_payment_id,))
        count_after = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM decision_audit")
        total_after = cur.fetchone()[0]
        cur.execute("SELECT payment_id, final_action, timestamp FROM decision_audit WHERE payment_id = ?", (test_payment_id,))
        row = cur.fetchone()

    print(f"Row count for {test_payment_id} after second run: {count_after}")
    print(f"Total audit table row count after second run: {total_after}")
    print(f"Persisted record: {row}")

    assert count_before == 1, f"Expected 1 row before, got {count_before}"
    assert count_after == 1, f"Expected 1 row after, got {count_after}"
    assert total_after == total_before, f"Expected total row count unchanged ({total_before}), got {total_after}"

    print("\n[SUCCESS] UPSERT verified! Duplicate execution updated the existing row without inserting a new one.")


if __name__ == "__main__":
    main()
