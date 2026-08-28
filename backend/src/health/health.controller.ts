import { Controller, Get, HttpStatus, Res } from '@nestjs/common';
import { Response } from 'express';
import {
  DecisionEngineDependencyStatus,
  DecisionEngineService,
} from '../decision-engine/decision-engine.service';

export interface HealthCheckResponse {
  status: 'ok' | 'degraded';
  service: 'recovery-intelligence-api';
  dependencies: {
    decision_engine: DecisionEngineDependencyStatus;
  };
}

@Controller('health')
export class HealthController {
  constructor(
    private readonly decisionEngineService: DecisionEngineService,
  ) {}

  @Get()
  async check(@Res({ passthrough: true }) res: Response): Promise<HealthCheckResponse> {
    const depStatus = await this.decisionEngineService.checkHealth();
    const isHealthy = depStatus === 'ok';

    // Never hide a failed Python decision engine behind HTTP 200 + "ok"
    if (!isHealthy) {
      res.status(HttpStatus.SERVICE_UNAVAILABLE);
    }

    return {
      status: isHealthy ? 'ok' : 'degraded',
      service: 'recovery-intelligence-api',
      dependencies: {
        decision_engine: depStatus,
      },
    };
  }
}
