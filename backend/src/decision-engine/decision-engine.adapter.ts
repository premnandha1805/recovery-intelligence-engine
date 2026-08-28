import { Injectable, Logger } from '@nestjs/common';
import { DecisionEngineService } from './decision-engine.service';

export interface PythonDecisionResult {
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

@Injectable()
export class DecisionEngineAdapter {
  private readonly logger = new Logger(DecisionEngineAdapter.name);

  constructor(private readonly decisionEngineService: DecisionEngineService) {}

  /**
   * Delegate recovery decision evaluation to DecisionEngineService over HTTP.
   */
  async evaluate(
    paymentId: string,
    requestId: string,
    forceRecompute: boolean = false,
  ): Promise<PythonDecisionResult> {
    return this.decisionEngineService.evaluate(
      paymentId,
      requestId,
      forceRecompute,
    );
  }
}
