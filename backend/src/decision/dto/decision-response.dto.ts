import { ApiProperty } from '@nestjs/swagger';

/**
 * Stable Response DTO for POST /decisions.
 *
 * Guaranteed 11-field Contract:
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
  @ApiProperty({
    description: 'Unique identifier of the payment transaction.',
    example: 'pay_000001_a1',
  })
  payment_id: string;

  @ApiProperty({
    description:
      'Recommendation derived by the causal uplift model (argmax over expected net value).',
    enum: ['RETRY', 'RETRY_NUDGE', 'WAIT', 'ESCALATE', 'N/A — error path'],
    example: 'RETRY_NUDGE',
  })
  model_decision: string;

  @ApiProperty({
    description:
      'Recommended action proposed by the reasoning engine or deterministic fallback.',
    enum: ['RETRY', 'RETRY_NUDGE', 'WAIT', 'ESCALATE', 'N/A — error path'],
    example: 'RETRY_NUDGE',
  })
  llm_decision: string;

  @ApiProperty({
    description:
      'True if safety guardrail rules intervened to override the recommended action; false otherwise.',
    example: false,
  })
  guardrail_overridden: boolean;

  @ApiProperty({
    description:
      'Human-readable explanation of why a guardrail override took place, or null if unviolated.',
    nullable: true,
    example: null,
  })
  guardrail_reason: string | null;

  @ApiProperty({
    description:
      'Executable recovery action to execute for this payment cycle after guardrail validation.',
    enum: ['RETRY', 'RETRY_NUDGE', 'WAIT', 'ESCALATE'],
    example: 'RETRY_NUDGE',
  })
  final_action: string;

  @ApiProperty({
    description: 'Statistical confidence score of the decision [0.0 - 1.0].',
    nullable: true,
    example: 0.85,
  })
  confidence: number | null;

  @ApiProperty({
    description: 'Assessed customer churn / fatigue risk level.',
    nullable: true,
    enum: ['LOW', 'MEDIUM', 'HIGH', 'none', null],
    example: 'LOW',
  })
  risk_level: string | null;

  @ApiProperty({
    description: 'Executive reasoning narrative detailing the recovery rationale.',
    nullable: true,
    example: 'Evaluation succeeded with high confidence',
  })
  reasoning: string | null;

  @ApiProperty({
    description:
      'Provenance source for the decision. ' +
      '"FOUNDRY_REASONING": freshly computed via causal model & reasoning engine; ' +
      '"cache": retrieved from persistent SQLite cache for this payment_id without a new LLM invocation; ' +
      '"error_path": safe fallback generated when payment context was not found.',
    enum: ['FOUNDRY_REASONING', 'cache', 'error_path', 'llm'],
    example: 'FOUNDRY_REASONING',
  })
  decision_source: string;

  @ApiProperty({
    description:
      'Correlation identifier preserved end-to-end across NestJS and Python service components.',
    example: 'c4b8e21a-79a1-4621-8f52-7bfb809831a2',
  })
  request_id: string;
}
