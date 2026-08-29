-- Forward-only migration: 001_initial_decision_tables.sql
-- Scope Decision: Forward-only migrations (no down-migrations) by deliberate design.
-- Schema parity: Preserves exact field names and semantics from Day 7 decision_engine/audit.py.

CREATE TABLE IF NOT EXISTS decision_audit (
    payment_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    request_id TEXT,
    raw_arm_probabilities JSONB,
    raw_arm_net_values JSONB,
    llm_proposed_decision TEXT,
    llm_confidence DOUBLE PRECISION,
    llm_reasoning TEXT,
    llm_risk_level TEXT,
    expected_incremental_value DOUBLE PRECISION,
    guardrail_verdict TEXT,
    guardrail_reason TEXT,
    final_action TEXT NOT NULL,
    decision_source TEXT NOT NULL,
    error TEXT,
    evaluated_at TIMESTAMPTZ NOT NULL,
    state_fingerprint TEXT
);

CREATE TABLE IF NOT EXISTS decision_audit_events (
    decision_id TEXT PRIMARY KEY,
    payment_id TEXT NOT NULL,
    request_id TEXT,
    evaluated_at TIMESTAMPTZ NOT NULL,
    decision_source TEXT,
    final_action TEXT NOT NULL,
    model_decision TEXT,
    llm_proposed_decision TEXT,
    guardrail_overridden BOOLEAN,
    guardrail_reason TEXT,
    state_fingerprint TEXT
);
