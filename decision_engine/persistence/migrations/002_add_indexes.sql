-- Forward-only migration: 002_add_indexes.sql
-- Scope Decision: Forward-only migrations (no down-migrations) by deliberate design.
-- Stable indexes for lookup performance and audit history queries.

-- decision_audit indexes (payment_id is already indexed by PRIMARY KEY)
CREATE INDEX IF NOT EXISTS idx_decision_audit_request_id
    ON decision_audit (request_id);

CREATE INDEX IF NOT EXISTS idx_decision_audit_evaluated_at
    ON decision_audit (evaluated_at);

-- decision_audit_events indexes
CREATE INDEX IF NOT EXISTS idx_decision_audit_events_payment_id
    ON decision_audit_events (payment_id);

CREATE INDEX IF NOT EXISTS idx_decision_audit_events_request_id
    ON decision_audit_events (request_id);

CREATE INDEX IF NOT EXISTS idx_decision_audit_events_evaluated_at
    ON decision_audit_events (evaluated_at);

CREATE INDEX IF NOT EXISTS idx_decision_audit_events_payment_evaluated_desc
    ON decision_audit_events (payment_id, evaluated_at DESC);
