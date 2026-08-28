import { Injectable, Logger, OnModuleDestroy } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as http from 'http';
import * as https from 'https';
import {
  DecisionEngineErrorException,
  DecisionEngineTimeoutException,
  DecisionEngineUnavailableException,
} from '../common/exceptions/decision-engine.exceptions';
import { PythonDecisionResult } from './decision-engine.adapter';

export type DecisionEngineDependencyStatus = 'ok' | 'unreachable' | 'timeout';

@Injectable()
export class DecisionEngineService implements OnModuleDestroy {
  private readonly logger = new Logger(DecisionEngineService.name);
  private readonly decisionEngineUrl: string;
  private readonly timeoutMs: number;
  private readonly healthCheckTimeoutMs: number;

  // Shared pooled HTTP agents for keep-alive connections (reused across requests, including 7E)
  private readonly httpAgent: http.Agent;
  private readonly httpsAgent: https.Agent;

  constructor(private readonly configService: ConfigService) {
    this.decisionEngineUrl = this.configService.get<string>(
      'DECISION_ENGINE_URL',
      'http://localhost:8000',
    );
    this.timeoutMs = this.configService.get<number>(
      'DECISION_ENGINE_TIMEOUT_MS',
      8000,
    );
    this.healthCheckTimeoutMs = this.configService.get<number>(
      'HEALTH_CHECK_TIMEOUT_MS',
      1000,
    );

    this.httpAgent = new http.Agent({
      keepAlive: true,
      maxSockets: 50,
      maxFreeSockets: 10,
      timeout: this.timeoutMs,
    });

    this.httpsAgent = new https.Agent({
      keepAlive: true,
      maxSockets: 50,
      maxFreeSockets: 10,
      timeout: this.timeoutMs,
    });

    this.logger.log(
      `DecisionEngineService initialized with URL: ${this.decisionEngineUrl}, timeout: ${this.timeoutMs}ms, healthTimeout: ${this.healthCheckTimeoutMs}ms`,
    );
  }

  getDecisionEngineUrl(): string {
    return this.decisionEngineUrl;
  }

  getTimeoutMs(): number {
    return this.timeoutMs;
  }

  getHealthCheckTimeoutMs(): number {
    return this.healthCheckTimeoutMs;
  }

  getHttpAgent(): http.Agent {
    return this.httpAgent;
  }

  getHttpsAgent(): https.Agent {
    return this.httpsAgent;
  }

  /**
   * Check health of Python Decision Engine dependency using shared pooled HTTP client.
   * Distinguishes 'ok', 'unreachable', and 'timeout' explicitly.
   */
  async checkHealth(): Promise<DecisionEngineDependencyStatus> {
    const url = new URL('/health', this.decisionEngineUrl);
    const isHttps = url.protocol === 'https:';
    const transport = isHttps ? https : http;
    const agent = isHttps ? this.httpsAgent : this.httpAgent;

    return new Promise<DecisionEngineDependencyStatus>((resolve) => {
      let settled = false;

      const finish = (result: DecisionEngineDependencyStatus) => {
        if (!settled) {
          settled = true;
          clearTimeout(safetyTimer);
          resolve(result);
        }
      };

      const req = transport.request(
        url,
        {
          method: 'GET',
          agent,
          timeout: this.healthCheckTimeoutMs,
        },
        (res) => {
          let data = '';
          res.setEncoding('utf8');
          res.on('data', (chunk) => {
            data += chunk;
          });
          res.on('end', () => {
            if (res.statusCode === 200) {
              try {
                const parsed = JSON.parse(data);
                if (parsed.status === 'ok') {
                  finish('ok');
                  return;
                }
              } catch {
                // malformed json
              }
            }
            finish('unreachable');
          });
        },
      );

      // Safety timeout timer matching healthCheckTimeoutMs to abort socket stalls
      const safetyTimer = setTimeout(() => {
        req.destroy(new Error('HEALTH_CHECK_TIMEOUT'));
        finish('timeout');
      }, this.healthCheckTimeoutMs);

      req.on('timeout', () => {
        req.destroy();
        finish('timeout');
      });

      req.on('error', (err: any) => {
        if (
          err?.message === 'HEALTH_CHECK_TIMEOUT' ||
          err?.code === 'ETIMEDOUT' ||
          (err?.code === 'ECONNRESET' && req.destroyed)
        ) {
          finish('timeout');
        } else {
          finish('unreachable');
        }
      });

      req.end();
    });
  }

  /**
   * Evaluate a recovery decision by invoking Python decision engine service over HTTP.
   * Maps connection failures to DecisionEngineUnavailableException (503),
   * timeouts to DecisionEngineTimeoutException (503),
   * and upstream 5xx errors to DecisionEngineErrorException (502).
   */
  async evaluate(
    paymentId: string,
    requestId: string,
    forceRecompute: boolean = false,
  ): Promise<PythonDecisionResult> {
    const url = new URL('/evaluate', this.decisionEngineUrl);
    const isHttps = url.protocol === 'https:';
    const transport = isHttps ? https : http;
    const agent = isHttps ? this.httpsAgent : this.httpAgent;

    const payload = JSON.stringify({
      payment_id: paymentId,
      request_id: requestId,
      force_recompute: forceRecompute,
    });

    return new Promise<PythonDecisionResult>((resolve, reject) => {
      let settled = false;

      const finishError = (err: any) => {
        if (!settled) {
          settled = true;
          clearTimeout(safetyTimer);
          reject(err);
        }
      };

      const finishSuccess = (res: PythonDecisionResult) => {
        if (!settled) {
          settled = true;
          clearTimeout(safetyTimer);
          resolve(res);
        }
      };

      const req = transport.request(
        url,
        {
          method: 'POST',
          agent,
          headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(payload),
            'x-request-id': requestId,
          },
          timeout: this.timeoutMs,
        },
        (res) => {
          let data = '';
          res.setEncoding('utf8');
          res.on('data', (chunk) => {
            data += chunk;
          });
          res.on('end', () => {
            if (res.statusCode === 200) {
              try {
                const parsed = JSON.parse(data) as PythonDecisionResult;
                finishSuccess(parsed);
              } catch (parseErr) {
                finishError(
                  new DecisionEngineErrorException(
                    'Invalid JSON received from Python decision engine',
                  ),
                );
              }
            } else if (res.statusCode && res.statusCode >= 500) {
              finishError(
                new DecisionEngineErrorException(
                  `Python decision engine returned upstream HTTP ${res.statusCode}`,
                ),
              );
            } else {
              finishError(
                new DecisionEngineErrorException(
                  `Python decision engine returned unexpected HTTP ${res.statusCode}`,
                ),
              );
            }
          });
        },
      );

      const safetyTimer = setTimeout(() => {
        req.destroy(new Error('DECISION_ENGINE_TIMEOUT'));
        finishError(new DecisionEngineTimeoutException());
      }, this.timeoutMs);

      req.on('timeout', () => {
        req.destroy();
        finishError(new DecisionEngineTimeoutException());
      });

      req.on('error', (err: any) => {
        if (
          err?.message === 'DECISION_ENGINE_TIMEOUT' ||
          err?.code === 'ETIMEDOUT' ||
          (err?.code === 'ECONNRESET' && req.destroyed)
        ) {
          finishError(new DecisionEngineTimeoutException());
        } else {
          finishError(
            new DecisionEngineUnavailableException(
              err?.message || 'Connection refused to decision engine',
            ),
          );
        }
      });

      req.write(payload);
      req.end();
    });
  }

  onModuleDestroy() {
    this.httpAgent.destroy();
    this.httpsAgent.destroy();
    this.logger.log('DecisionEngineService HTTP agents destroyed cleanly');
  }
}
