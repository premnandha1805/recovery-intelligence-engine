/**
 * Stable Response DTO for POST /decisions.
 *
 * Contract:
 * - payment_id: string
 * - model_decision: string
 * - llm_decision: string
 * - guardrail_overridden: boolean
 * - guardrail_reason: string | null
 * - final_action: string
 * - confidence: number | null
 * - risk_level: string | null
 * - reasoning: string | null
 * - decision_source: string
 * - request_id: string
 */
export class DecisionResponseDto {
  payment_id: string;
  model_decision: string;
  llm_decision: string;
  guardrail_overridden: boolean;
  guardrail_reason: string | null;
  final_action: string;
  confidence: number | null;
  risk_level: string | null;
  reasoning: string | null;
  decision_source: string;
  request_id: string;
}
