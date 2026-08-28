import { Injectable } from '@nestjs/common';
import { randomUUID } from 'crypto';
import { CreateDecisionDto } from './dto/create-decision.dto';
import { DecisionResponseDto } from './dto/decision-response.dto';

@Injectable()
export class DecisionService {
  /**
   * Generates decision response conforming to stable 7C contract.
   * Full HTTP adapter calling Python decision engine service is deferred to Day 7E.
   */
  async createDecision(
    dto: CreateDecisionDto,
    requestId?: string,
  ): Promise<DecisionResponseDto> {
    const effectiveRequestId = requestId || `req-${randomUUID()}`;

    return {
      payment_id: dto.payment_id,
      model_decision: 'RETRY_NUDGE',
      llm_decision: 'RETRY_NUDGE',
      guardrail_overridden: false,
      guardrail_reason: null,
      final_action: 'RETRY_NUDGE',
      confidence: 0.85,
      risk_level: 'LOW',
      reasoning:
        'Evaluation skeleton placeholder conforming to Day 7C stable contract',
      decision_source: 'FOUNDRY_REASONING',
      request_id: effectiveRequestId,
    };
  }
}
