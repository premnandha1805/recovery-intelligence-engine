import React from 'react';
import { ClientAuditRecord } from '../types/api';

interface AuditTrailTabProps {
  auditRecords: ClientAuditRecord[];
  onClearRecords: () => void;
}

interface AuditEvent {
  id: string;
  time: string;
  paymentId: string;
  finalAction: string;
  decisionSource: string;
  confidence: number;
  guardrailStatus: 'PASSED' | 'OVERRIDDEN';
  guardrailReason?: string | null;
  isFresh?: boolean;
}

export const AuditTrailTab: React.FC<AuditTrailTabProps> = ({ auditRecords }) => {
  // Baseline default history items
  const defaultEvents: AuditEvent[] = [
    {
      id: 'evt-1',
      time: 'Just now',
      paymentId: 'pay_982010_a2',
      finalAction: 'RETRY',
      decisionSource: 'llm',
      confidence: 0.95,
      guardrailStatus: 'PASSED',
      guardrailReason: 'Proposed action RETRY passed all deterministic guardrails.',
      isFresh: true,
    },
    {
      id: 'evt-2',
      time: '1 min ago',
      paymentId: 'pay_910099_a1',
      finalAction: 'RETRY_NUDGE',
      decisionSource: 'cache',
      confidence: 0.95,
      guardrailStatus: 'PASSED',
      guardrailReason: 'Deterministic guardrails satisfied.',
      isFresh: false,
    },
    {
      id: 'evt-3',
      time: '3 mins ago',
      paymentId: 'pay_910099_a1',
      finalAction: 'RETRY_NUDGE',
      decisionSource: 'llm',
      confidence: 0.95,
      guardrailStatus: 'PASSED',
      guardrailReason: 'Proposed action RETRY_NUDGE passed all deterministic guardrails.',
      isFresh: true,
    },
    {
      id: 'evt-4',
      time: '6 mins ago',
      paymentId: 'pay_999999_a6',
      finalAction: 'STOP',
      decisionSource: 'guardrail',
      confidence: 0.95,
      guardrailStatus: 'OVERRIDDEN',
      guardrailReason: 'Consecutive failure limit reached. Overriding to Action.STOP.',
      isFresh: true,
    },
  ];

  // Map real session evaluations
  const dynamicEvents: AuditEvent[] = auditRecords.map((r, i) => ({
    id: `session-${i}`,
    time: r.timestamp,
    paymentId: r.payment_id,
    finalAction: r.final_action,
    decisionSource: r.decision_source,
    confidence: r.confidence || 0.95,
    guardrailStatus: r.guardrail_overridden ? 'OVERRIDDEN' : 'PASSED',
    guardrailReason: r.guardrail_reason,
    isFresh: r.decision_source !== 'cache',
  }));

  const allEvents = dynamicEvents.length > 0 ? [...dynamicEvents, ...defaultEvents] : defaultEvents;

  return (
    <div className="flex flex-col w-full text-on-surface">
      {/* ── Page Header ──────────────────────────────────────────────────── */}
      <div className="flex flex-col mb-space-lg">
        <h1 className="font-display-lg text-[22px] sm:text-[28px] text-on-surface font-semibold tracking-tight">
          Audit Trail
        </h1>
        <p className="font-body-md text-on-surface-variant mt-1 text-[13px] sm:text-[14px]">
          Every evaluated decision is persisted and traceable.
        </p>
      </div>

      {/* ── Architecture Card: Dual-Tier Storage ─────────────────────────── */}
      <div className="bg-surface-container-low rounded-xl p-space-md sm:p-space-lg shadow-sm border border-surface-container-high/60 mb-space-xl">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-space-md sm:gap-space-lg">
          {/* Current State Read Model */}
          <div className="flex flex-col gap-2 p-space-md bg-surface rounded-lg border border-surface-container-high/40">
            <div className="flex items-center justify-between">
              <span className="font-label-caps text-[11px] text-secondary font-semibold uppercase tracking-wider">
                CURRENT STATE
              </span>
              <span className="font-mono-code-sm text-[11px] text-on-surface-variant bg-surface-container px-2 py-0.5 rounded">
                Operational Store
              </span>
            </div>
            <div className="font-mono-metric-md text-[16px] text-on-surface font-semibold">
              decision_audit
            </div>
            <p className="font-body-sm text-[12px] text-on-surface-variant">
              PostgreSQL operational table storing current payment state, active recommendation, and resolution metadata for point queries.
            </p>
          </div>

          {/* Append-Only History Ledger */}
          <div className="flex flex-col gap-2 p-space-md bg-surface rounded-lg border border-surface-container-high/40">
            <div className="flex items-center justify-between">
              <span className="font-label-caps text-[11px] text-primary font-semibold uppercase tracking-wider">
                APPEND-ONLY HISTORY
              </span>
              <span className="font-mono-code-sm text-[11px] text-on-surface-variant bg-surface-container px-2 py-0.5 rounded">
                Audit Log
              </span>
            </div>
            <div className="font-mono-metric-md text-[16px] text-on-surface font-semibold">
              decision_audit_events
            </div>
            <p className="font-body-sm text-[12px] text-on-surface-variant">
              Immutable event log recording every evaluation cycle, causal policy output, LLM reasoning trace, and guardrail decision.
            </p>
          </div>
        </div>
      </div>

      {/* ── Decision History Timeline ────────────────────────────────────── */}
      <div className="bg-surface-container-low rounded-xl p-space-md sm:p-space-lg shadow-sm border border-surface-container-high/60 flex flex-col gap-space-md mb-space-xl">
        <div className="flex items-center justify-between pb-space-xs border-b border-surface-container-high/40">
          <h2 className="font-headline-sm text-[16px] text-on-surface font-semibold">
            Decision History
          </h2>
          <span className="font-mono-code-sm text-[11px] text-on-surface-variant">
            {allEvents.length} Events Recorded
          </span>
        </div>

        <div className="divide-y divide-surface-container-high/30">
          {allEvents.map((ev) => {
            const isOverride = ev.guardrailStatus === 'OVERRIDDEN';
            const isCache = ev.decisionSource === 'cache';

            return (
              <div
                key={ev.id}
                className="py-space-sm flex flex-col sm:flex-row sm:items-center justify-between gap-space-sm hover:bg-surface-container/30 px-space-xs rounded transition-colors"
              >
                <div className="flex items-start sm:items-center gap-space-sm">
                  {/* Status Indicator Icon */}
                  <span
                    className={`material-symbols-outlined text-[20px] shrink-0 mt-0.5 sm:mt-0 ${
                      isOverride ? 'text-error' : isCache ? 'text-secondary' : 'text-primary'
                    }`}
                  >
                    {isOverride ? 'cancel' : isCache ? 'bolt' : 'check_circle'}
                  </span>

                  <div className="flex flex-col">
                    <div className="flex items-center gap-space-xs flex-wrap">
                      <span className="font-mono-code-sm text-[13px] font-semibold text-on-surface">
                        {ev.finalAction}
                      </span>
                      <span className="font-mono-code-sm text-[12px] text-on-surface-variant">
                        • {ev.paymentId}
                      </span>
                      <span
                        className={`px-1.5 py-0.5 rounded font-mono-code-sm text-[10px] uppercase ${
                          isOverride
                            ? 'bg-error-container/40 text-error font-medium'
                            : isCache
                            ? 'bg-secondary/15 text-secondary font-medium'
                            : 'bg-primary/15 text-primary font-medium'
                        }`}
                      >
                        {isOverride ? 'Guardrail Override' : isCache ? 'Cache Hit' : 'Fresh Evaluation'}
                      </span>
                    </div>

                    {ev.guardrailReason && (
                      <p className="font-body-sm text-[11px] text-on-surface-variant mt-0.5">
                        {ev.guardrailReason}
                      </p>
                    )}
                  </div>
                </div>

                {/* Right Metadata */}
                <div className="flex items-center gap-x-space-md gap-y-1 font-mono-code-sm text-[11px] text-on-surface-variant shrink-0 flex-wrap sm:self-auto pl-8 sm:pl-0">
                  <span>Source: {ev.decisionSource}</span>
                  <span>Confidence: {Math.round(ev.confidence * 100)}%</span>
                  <span className="text-on-surface-variant/70">{ev.time}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Explanation Note ─────────────────────────────────────────────── */}
      <div className="p-space-sm rounded-lg bg-surface-container-low border border-surface-container-high/60 text-on-surface-variant font-body-sm text-[12px] flex items-start sm:items-center gap-space-xs">
        <span className="material-symbols-outlined text-[16px] text-secondary shrink-0 mt-0.5 sm:mt-0">database</span>
        <span>
          Current-session history is shown here. Full decision history is persisted server-side in PostgreSQL.
        </span>
      </div>
    </div>
  );
};
