import { Injectable, Logger } from '@nestjs/common';
import { randomUUID } from 'crypto';
import { Request } from 'express';
import { DecisionEngineAdapter } from '../decision-engine/decision-engine.adapter';
import { CreateDecisionDto } from './dto/create-decision.dto';
import { DecisionResponseDto } from './dto/decision-response.dto';
import {
  DecisionEngineErrorException,
  DecisionEngineTimeoutException,
  DecisionEngineUnavailableException,
} from '../common/exceptions/decision-engine.exceptions';

@Injectable()
export class DecisionService {
  private readonly logger = new Logger(DecisionService.name);

  constructor(
    private readonly decisionEngineAdapter: DecisionEngineAdapter,
  ) {}

  /**
   * Process recovery decision request by delegating to Python DecisionEngineAdapter.
   *
   * Rules:
   * - Reuses caller-supplied X-Request-Id if present, otherwise generates UUID [FIX-10].
   * - Reuses request_id in any thrown decision engine exceptions for error correlation.
   * - Calls DecisionEngineAdapter.evaluate(payment_id, request_id, force_recompute).
   * - Maps Python result 1:1 into stable DecisionResponseDto.
   * - Preserves guardrail_overridden and guardrail_reason verbatim.
   * - Logs execution duration_ms and final_action.
   * - Never computes probabilities, calls LLM, evaluates guardrails, or accesses DB directly.
   */
  async createDecision(
    dto: CreateDecisionDto,
    clientRequestId?: string,
    req?: Request,
  ): Promise<DecisionResponseDto> {
    const startTime = Date.now();

    // Preserve incoming X-Request-Id or reuse/generate fresh UUID [FIX-10]
    const requestId =
      clientRequestId && clientRequestId.trim()
        ? clientRequestId.trim()
        : (req as any)?.requestId || randomUUID();

    if (req) {
      (req as any).requestId = requestId;
    }

    const forceRecompute = dto.force_recompute ?? false;

    let pythonResult;
    try {
      pythonResult = await this.decisionEngineAdapter.evaluate(
        dto.payment_id,
        requestId,
        forceRecompute,
      );
    } catch (err: any) {
      // Ensure decision engine exceptions carry the correlated requestId for exception filter
      if (err instanceof DecisionEngineUnavailableException && !err.requestId) {
        throw new DecisionEngineUnavailableException(err.message, requestId);
      }
      if (err instanceof DecisionEngineTimeoutException && !err.requestId) {
        throw new DecisionEngineTimeoutException(err.message, requestId);
      }
      if (err instanceof DecisionEngineErrorException && !err.requestId) {
        throw new DecisionEngineErrorException(err.message, requestId);
      }
      throw err;
    }

    // Map 1:1 into stable DecisionResponseDto, preserving guardrails verbatim
    const responseDto: DecisionResponseDto = {
      payment_id: pythonResult.payment_id,
      model_decision: pythonResult.model_decision,
      llm_decision: pythonResult.llm_decision,
      guardrail_overridden: Boolean(pythonResult.guardrail_overridden),
      guardrail_reason:
        pythonResult.guardrail_reason !== undefined
          ? pythonResult.guardrail_reason
          : null,
      final_action: pythonResult.final_action,
      confidence:
        pythonResult.confidence !== undefined ? pythonResult.confidence : null,
      risk_level:
        pythonResult.risk_level !== undefined ? pythonResult.risk_level : null,
      reasoning:
        pythonResult.reasoning !== undefined ? pythonResult.reasoning : null,
      decision_source: pythonResult.decision_source,
      request_id: pythonResult.request_id || requestId,
    };

    const durationMs = Date.now() - startTime;
    this.logger.log(
      `[${requestId}] Decision completed in ${durationMs}ms for ${dto.payment_id}: final_action=${responseDto.final_action}`,
    );

    return responseDto;
  }
}
