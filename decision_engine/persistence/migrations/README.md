# Recovery Decision Engine - Database Migrations

## Forward-Only Migrations Scope Decision

**Deliberate Architectural Scope Decision**:
This project strictly implements **forward-only migrations** (no down-migrations / rollbacks). This is a deliberate, reasoned engineering choice, not an oversight.

### Rationale
1. **Production Safety**: In event-sourced, audit-heavy, and financial systems, rolling back database schemas (down-migrations) frequently causes catastrophic data loss or corruption. Schema evolution must be additive, forward-compatible, and roll-forward only.
2. **Operational Simplicity**: Without ORMs or complex migration frameworks like Alembic (which would introduce unnecessary machinery given the project's architecture), numbered raw SQL files provide transparent, auditable, and deterministic state transitions.
3. **Roll-Forward Strategy**: Any required schema modifications, column adjustments, or corrections are applied via new numbered migration files (e.g., `003_...sql`), preserving a monotonic audit history.

## Architecture

- **Format**: Raw PostgreSQL DDL in numbered files (`001_...sql`, `002_...sql`).
- **Tracking**: Tracked in table `schema_migrations (id VARCHAR(255) PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)`.
- **Execution**: Managed by `decision_engine/persistence/migrate.py`. Each migration executes within an atomic transaction.
- **Driver**: `psycopg` v3 (`psycopg3`).
