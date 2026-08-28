import { Injectable, Logger, OnModuleDestroy } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class DecisionEngineService implements OnModuleDestroy {
  private readonly logger = new Logger(DecisionEngineService.name);
  private readonly decisionEngineUrl: string;
  private readonly timeoutMs: number;

  constructor(private readonly configService: ConfigService) {
    this.decisionEngineUrl = this.configService.get<string>(
      'DECISION_ENGINE_URL',
      'http://localhost:8000',
    );
    this.timeoutMs = this.configService.get<number>(
      'DECISION_ENGINE_TIMEOUT_MS',
      8000,
    );
    this.logger.log(
      `DecisionEngineService initialized with URL: ${this.decisionEngineUrl}, timeout: ${this.timeoutMs}ms (skeleton for Day 7E)`,
    );
  }

  getDecisionEngineUrl(): string {
    return this.decisionEngineUrl;
  }

  getTimeoutMs(): number {
    return this.timeoutMs;
  }

  onModuleDestroy() {
    // Shutdown hook for HTTP agent clean teardown (FIX-8, to be populated in Day 7E)
    this.logger.log('DecisionEngineService teardown hook executed');
  }
}
