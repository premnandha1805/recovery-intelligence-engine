/**
 * Canonical payment rails supported by the Recovery Intelligence Engine.
 */
export type PaymentMethod = 'card' | 'upi' | 'netbanking' | 'wallet';

/**
 * Canonical failure reasons supported by the Recovery Intelligence Engine.
 */
export type FailureReason =
  | 'insufficient_funds'
  | 'bank_decline'
  | 'network_error'
  | 'expired_card'
  | 'authentication_failure'
  | 'temporary_bank_issue';

/**
 * 9 observable cold-start features plus verified optional operational attributes.
 */
export interface PaymentFeatures {
  amount: number;
  attempt_number: number;
  dynamic_success_rate: number;
  cumulative_failures: number;
  consecutive_failed_cycles: number;
  notification_engagement_score: number;
  contact_response_score: number;
  payment_method: PaymentMethod;
  failure_reason: FailureReason;

  // Optional operational / guardrail fields explicitly verified in backend DTO
  consecutive_failures?: number;
  retry_count_current_cycle?: number;
  lifetime_escalations?: number;
  interventions_last_7_days?: number;
  status?: 'failed' | 'pending' | 'success' | 'recovered';
}

/**
 * Request DTO matching POST /decisions.
 */
export interface CreateDecisionRequest {
  payment_id: string;
  force_recompute?: boolean;
  features?: PaymentFeatures;
}

/**
 * Stable 11-field Response DTO from POST /decisions.
 */
export interface DecisionResponse {
  payment_id: string;
  model_decision: string;
  llm_decision: string;
  guardrail_overridden: boolean;
  guardrail_reason: string | null;
  final_action: 'RETRY' | 'RETRY_NUDGE' | 'WAIT' | 'ESCALATE' | 'STOP' | string;
  confidence: number | null;
  risk_level: string | null;
  reasoning: string | null;
  decision_source: 'FOUNDRY_REASONING' | 'cache' | 'error_path' | 'llm' | 'fallback_no_llm' | string;
  request_id: string;
}

/**
 * Standard backend error envelope shape.
 */
export interface ApiErrorEnvelope {
  error: {
    code: string;
    message: string;
    request_id: string;
  };
}

/**
 * Client-side session record for audit history tracking.
 */
export interface ClientAuditRecord {
  id: string;
  timestamp: string;
  payment_id: string;
  request_id: string;
  final_action: string;
  decision_source: string;
  confidence: number | null;
  guardrail_overridden: boolean;
  guardrail_reason: string | null;
  latency_ms: number;
  status: 'success' | 'error';
  error_message?: string;
  amount?: number;
  payment_method?: string;
}
