import { Injectable, Logger, OnModuleDestroy } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as http from 'http';
import * as https from 'https';
import {
  DecisionEngineErrorException,
  DecisionEngineTimeoutException,
  DecisionEngineUnavailableException,
} from '../common/exceptions/decision-engine.exceptions';

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

/**
 * Check whether an error is a network-level / DNS / connection failure eligible for retry.
 * Excludes timeouts and HTTP error responses.
 */
function isNetworkLevelFailure(err: any): boolean {
  if (!err) return false;
  if (
    err.message === 'DECISION_ENGINE_TIMEOUT' ||
    err.code === 'ETIMEDOUT' ||
    err.name === 'AbortError' ||
    err.isTimeout === true
  ) {
    return false;
  }
  const code = err.code;
  return (
    code === 'ECONNREFUSED' ||
    code === 'ENOTFOUND' ||
    code === 'EAI_AGAIN' ||
    code === 'EHOSTUNREACH' ||
    code === 'ENETUNREACH' ||
    code === 'ECONNRESET'
  );
}

@Injectable()
export class DecisionEngineAdapter implements OnModuleDestroy {
  private readonly logger = new Logger(DecisionEngineAdapter.name);
  private readonly decisionEngineUrl: string;
  private readonly timeoutMs: number;

  // Shared persistent keep-alive agents for the process lifetime [FIX-8]
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
      `DecisionEngineAdapter initialized with URL: ${this.decisionEngineUrl}, timeout: ${this.timeoutMs}ms (shared keep-alive pool)`,
    );
  }

  getDecisionEngineUrl(): string {
    return this.decisionEngineUrl;
  }

  getTimeoutMs(): number {
    return this.timeoutMs;
  }

  getHttpAgent(): http.Agent {
    return this.httpAgent;
  }

  getHttpsAgent(): https.Agent {
    return this.httpsAgent;
  }

  /**
   * Evaluate recovery decision with Python decision engine.
   *
   * Rules:
   * - Retries network-level connection/DNS/refused failures exactly once.
   * - Never retries HTTP errors (4xx/5xx).
   * - Never retries timeouts (Python manages internal retry/fallback).
   * - Sends X-Request-Id header on all calls.
   * - Maps non-2xx responses using Python's error envelope shape.
   */
  async evaluate(
    paymentId: string,
    requestId: string,
    forceRecompute: boolean = false,
  ): Promise<PythonDecisionResult> {
    const maxAttempts = 2; // Initial attempt + 1 retry on network-level failure
    let lastError: any;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        return await this.executePostEvaluate(
          paymentId,
          requestId,
          forceRecompute,
        );
      } catch (err: any) {
        lastError = err;

        // Only retry network-level failures on attempt 1
        if (attempt < maxAttempts && isNetworkLevelFailure(err)) {
          this.logger.warn(
            `[${requestId}] Network failure on attempt ${attempt}/${maxAttempts} (${err.code || err.message}). Retrying exactly once...`,
          );
          continue;
        }

        // Do NOT retry timeouts, HTTP 4xx/5xx, or exhausted attempts
        break;
      }
    }

    if (
      lastError instanceof DecisionEngineTimeoutException ||
      lastError instanceof DecisionEngineErrorException ||
      lastError instanceof DecisionEngineUnavailableException
    ) {
      throw lastError;
    }

    if (
      lastError?.isTimeout ||
      lastError?.code === 'ETIMEDOUT' ||
      lastError?.message === 'DECISION_ENGINE_TIMEOUT'
    ) {
      throw new DecisionEngineTimeoutException(
        'Python decision engine request timed out',
        requestId,
      );
    }

    throw new DecisionEngineUnavailableException(
      lastError?.message || 'Python decision engine service unavailable',
      requestId,
    );
  }

  private executePostEvaluate(
    paymentId: string,
    requestId: string,
    forceRecompute: boolean,
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
                    requestId,
                  ),
                );
              }
            } else {
              // Map non-2xx using Python's error envelope shape: { "error": { "code": "...", "message": "..." } }
              let errorMessage = `Python decision engine returned HTTP ${res.statusCode}`;
              try {
                const parsed = JSON.parse(data);
                if (parsed?.error?.message) {
                  errorMessage = parsed.error.message;
                } else if (parsed?.message) {
                  errorMessage = String(parsed.message);
                }
              } catch {
                // non-json response body
              }

              finishError(
                new DecisionEngineErrorException(errorMessage, requestId),
              );
            }
          });
        },
      );

      const safetyTimer = setTimeout(() => {
        const timeoutErr: any = new Error('DECISION_ENGINE_TIMEOUT');
        timeoutErr.isTimeout = true;
        req.destroy(timeoutErr);
        finishError(
          new DecisionEngineTimeoutException(
            'Python decision engine request timed out',
            requestId,
          ),
        );
      }, this.timeoutMs);

      req.on('timeout', () => {
        const timeoutErr: any = new Error('DECISION_ENGINE_TIMEOUT');
        timeoutErr.isTimeout = true;
        req.destroy(timeoutErr);
        finishError(
          new DecisionEngineTimeoutException(
            'Python decision engine request timed out',
            requestId,
          ),
        );
      });

      req.on('error', (err: any) => {
        if (
          err?.message === 'DECISION_ENGINE_TIMEOUT' ||
          err?.code === 'ETIMEDOUT' ||
          err?.isTimeout
        ) {
          finishError(
            new DecisionEngineTimeoutException(
              'Python decision engine request timed out',
              requestId,
            ),
          );
        } else {
          finishError(err);
        }
      });

      req.write(payload);
      req.end();
    });
  }

  onModuleDestroy() {
    this.httpAgent.destroy();
    this.httpsAgent.destroy();
    this.logger.log('DecisionEngineAdapter HTTP agents destroyed cleanly [FIX-8]');
  }
}
