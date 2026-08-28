import { Test, TestingModule } from '@nestjs/testing';
import { ConfigService } from '@nestjs/config';
import * as http from 'http';
import { EventEmitter } from 'events';
import {
  DecisionEngineAdapter,
  PythonDecisionResult,
} from './decision-engine.adapter';
import {
  DecisionEngineErrorException,
  DecisionEngineTimeoutException,
  DecisionEngineUnavailableException,
} from '../common/exceptions/decision-engine.exceptions';

describe('DecisionEngineAdapter (Day 7E HTTP Client)', () => {
  let adapter: DecisionEngineAdapter;
  let configService: ConfigService;

  const mockConfigValues: Record<string, any> = {
    DECISION_ENGINE_URL: 'http://localhost:8000',
    DECISION_ENGINE_TIMEOUT_MS: 8000,
  };

  const mockPythonDecision: PythonDecisionResult = {
    payment_id: 'pay_000001_a1',
    model_decision: 'RETRY_NUDGE',
    llm_decision: 'RETRY_NUDGE',
    guardrail_overridden: false,
    guardrail_reason: null,
    final_action: 'RETRY_NUDGE',
    confidence: 0.89,
    risk_level: 'LOW',
    reasoning: 'High recovery probability model recommendation',
    decision_source: 'FOUNDRY_REASONING',
    request_id: 'req-python-eval-123',
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        DecisionEngineAdapter,
        {
          provide: ConfigService,
          useValue: {
            get: jest.fn((key: string, defaultValue?: any) => {
              if (key in mockConfigValues) {
                return mockConfigValues[key];
              }
              return defaultValue;
            }),
          },
        },
      ],
    }).compile();

    adapter = module.get<DecisionEngineAdapter>(DecisionEngineAdapter);
    configService = module.get<ConfigService>(ConfigService);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should be defined', () => {
    expect(adapter).toBeDefined();
  });

  // ── 1. Normal successful response ──────────────────────────────────────────
  it('1. normal successful response: maps Python response 1:1', async () => {
    jest.spyOn(http, 'request').mockImplementation(((url: any, options: any, cb: any) => {
      const req = new EventEmitter() as any;
      req.write = jest.fn();
      req.end = jest.fn();
      req.destroy = jest.fn();

      process.nextTick(() => {
        const res = new EventEmitter() as any;
        res.statusCode = 200;
        res.setEncoding = jest.fn();
        cb(res);
        res.emit('data', JSON.stringify(mockPythonDecision));
        res.emit('end');
      });

      return req;
    }) as any);

    const result = await adapter.evaluate('pay_000001_a1', 'req-python-eval-123', false);

    expect(result).toEqual(mockPythonDecision);
    expect(result.payment_id).toBe('pay_000001_a1');
    expect(result.final_action).toBe('RETRY_NUDGE');
    expect(result.request_id).toBe('req-python-eval-123');
  });

  // ── 2. Timeout value matches config (not hardcoded) ────────────────────────
  it('2. timeout: assert timeout value used matches configured 8s from config, not hardcoded', async () => {
    let capturedTimeout: number | undefined;

    // Use a custom non-default config timeout (8500ms) to prove config adherence
    jest.spyOn(configService, 'get').mockImplementation(((key: string, def?: any) => {
      if (key === 'DECISION_ENGINE_TIMEOUT_MS') return 8500;
      if (key === 'DECISION_ENGINE_URL') return 'http://localhost:8000';
      return def;
    }) as any);

    const customModule = await Test.createTestingModule({
      providers: [
        DecisionEngineAdapter,
        { provide: ConfigService, useValue: configService },
      ],
    }).compile();
    const customAdapter = customModule.get<DecisionEngineAdapter>(DecisionEngineAdapter);

    jest.spyOn(http, 'request').mockImplementation(((url: any, options: any, cb: any) => {
      capturedTimeout = options.timeout;
      const req = new EventEmitter() as any;
      req.write = jest.fn();
      req.end = jest.fn();
      req.destroy = jest.fn();

      process.nextTick(() => {
        const res = new EventEmitter() as any;
        res.statusCode = 200;
        res.setEncoding = jest.fn();
        cb(res);
        res.emit('data', JSON.stringify(mockPythonDecision));
        res.emit('end');
      });

      return req;
    }) as any);

    await customAdapter.evaluate('pay_000001_a1', 'req-timeout-check', false);

    expect(capturedTimeout).toBe(8500);
    expect(capturedTimeout).toBe(customAdapter.getTimeoutMs());
  });

  // ── 3. Connection refused ──────────────────────────────────────────────────
  it('3. connection refused: maps to DecisionEngineUnavailableException after retry', async () => {
    let requestCount = 0;
    jest.spyOn(http, 'request').mockImplementation(((url: any, options: any, cb: any) => {
      requestCount++;
      const req = new EventEmitter() as any;
      req.write = jest.fn();
      req.end = jest.fn();
      req.destroy = jest.fn();

      process.nextTick(() => {
        const connErr: any = new Error('connect ECONNREFUSED 127.0.0.1:8000');
        connErr.code = 'ECONNREFUSED';
        req.emit('error', connErr);
      });

      return req;
    }) as any);

    let thrownErr: any;
    try {
      await adapter.evaluate('pay_000001_a1', 'req-conn-refused', false);
    } catch (err) {
      thrownErr = err;
    }

    expect(thrownErr).toBeInstanceOf(DecisionEngineUnavailableException);
    expect(thrownErr.getStatus()).toBe(503);
    expect(thrownErr.requestId).toBe('req-conn-refused');
    expect(requestCount).toBe(2); // Initial + 1 retry
  });

  // ── 4. DNS / network failure ───────────────────────────────────────────────
  it('4. DNS/network failure: maps to DecisionEngineUnavailableException after retry', async () => {
    let requestCount = 0;
    jest.spyOn(http, 'request').mockImplementation(((url: any, options: any, cb: any) => {
      requestCount++;
      const req = new EventEmitter() as any;
      req.write = jest.fn();
      req.end = jest.fn();
      req.destroy = jest.fn();

      process.nextTick(() => {
        const dnsErr: any = new Error('getaddrinfo ENOTFOUND fastapi-host');
        dnsErr.code = 'ENOTFOUND';
        req.emit('error', dnsErr);
      });

      return req;
    }) as any);

    let thrownErr: any;
    try {
      await adapter.evaluate('pay_000001_a1', 'req-dns-fail', false);
    } catch (err) {
      thrownErr = err;
    }

    expect(thrownErr).toBeInstanceOf(DecisionEngineUnavailableException);
    expect(thrownErr.getStatus()).toBe(503);
    expect(requestCount).toBe(2);
  });

  // ── 5. Python 500 → DECISION_ENGINE_ERROR, no retry ───────────────────────
  it('5. Python 500: mapped to DECISION_ENGINE_ERROR with envelope message and no retry', async () => {
    let requestCount = 0;
    jest.spyOn(http, 'request').mockImplementation(((url: any, options: any, cb: any) => {
      requestCount++;
      const req = new EventEmitter() as any;
      req.write = jest.fn();
      req.end = jest.fn();
      req.destroy = jest.fn();

      process.nextTick(() => {
        const res = new EventEmitter() as any;
        res.statusCode = 500;
        res.setEncoding = jest.fn();
        cb(res);
        res.emit(
          'data',
          JSON.stringify({
            error: {
              code: 'INTERNAL_ERROR',
              message: 'CausalUpliftPolicy failed during estimation',
            },
          }),
        );
        res.emit('end');
      });

      return req;
    }) as any);

    let thrownErr: any;
    try {
      await adapter.evaluate('pay_000001_a1', 'req-python-500', false);
    } catch (err) {
      thrownErr = err;
    }

    expect(thrownErr).toBeInstanceOf(DecisionEngineErrorException);
    expect(thrownErr.getStatus()).toBe(502);
    expect(thrownErr.message).toBe('CausalUpliftPolicy failed during estimation');
    expect(thrownErr.requestId).toBe('req-python-500');
    expect(requestCount).toBe(1); // Crucial: NO retry on HTTP 5xx
  });

  // ── 6. Retry exactly once on connection refusal ───────────────────────────
  it('6. retry exactly once on connection refusal: succeeds on 2nd attempt', async () => {
    let attempt = 0;
    jest.spyOn(http, 'request').mockImplementation(((url: any, options: any, cb: any) => {
      attempt++;
      const req = new EventEmitter() as any;
      req.write = jest.fn();
      req.end = jest.fn();
      req.destroy = jest.fn();

      if (attempt === 1) {
        process.nextTick(() => {
          const connErr: any = new Error('connect ECONNREFUSED 127.0.0.1:8000');
          connErr.code = 'ECONNREFUSED';
          req.emit('error', connErr);
        });
      } else {
        process.nextTick(() => {
          const res = new EventEmitter() as any;
          res.statusCode = 200;
          res.setEncoding = jest.fn();
          cb(res);
          res.emit('data', JSON.stringify(mockPythonDecision));
          res.emit('end');
        });
      }

      return req;
    }) as any);

    const result = await adapter.evaluate('pay_000001_a1', 'req-retry-success', false);

    expect(attempt).toBe(2); // Exactly 1 retry
    expect(result).toEqual(mockPythonDecision);
  });

  // ── 7. No retry on HTTP 5xx ────────────────────────────────────────────────
  it('7. no retry on HTTP 5xx: aborts immediately on 503 from Python', async () => {
    let attempts = 0;
    jest.spyOn(http, 'request').mockImplementation(((url: any, options: any, cb: any) => {
      attempts++;
      const req = new EventEmitter() as any;
      req.write = jest.fn();
      req.end = jest.fn();
      req.destroy = jest.fn();

      process.nextTick(() => {
        const res = new EventEmitter() as any;
        res.statusCode = 503;
        res.setEncoding = jest.fn();
        cb(res);
        res.emit(
          'data',
          JSON.stringify({
            error: {
              code: 'INTERNAL_ERROR',
              message: 'Engine components not fully initialized',
            },
          }),
        );
        res.emit('end');
      });

      return req;
    }) as any);

    await expect(
      adapter.evaluate('pay_000001_a1', 'req-no-retry-503', false),
    ).rejects.toThrow(DecisionEngineErrorException);

    expect(attempts).toBe(1); // Proves no retry on HTTP 5xx
  });

  // ── 8. 8-second timeout behavior (no retry on timeout) ─────────────────────
  it('8. 8-second timeout behavior: throws DecisionEngineTimeoutException with no retry', async () => {
    let attempts = 0;
    jest.spyOn(http, 'request').mockImplementation(((url: any, options: any, cb: any) => {
      attempts++;
      const req = new EventEmitter() as any;
      req.write = jest.fn();
      req.end = jest.fn();
      req.destroy = jest.fn();

      process.nextTick(() => {
        req.emit('timeout');
      });

      return req;
    }) as any);

    let thrownErr: any;
    try {
      await adapter.evaluate('pay_000001_a1', 'req-timeout-8s', false);
    } catch (err) {
      thrownErr = err;
    }

    expect(thrownErr).toBeInstanceOf(DecisionEngineTimeoutException);
    expect(thrownErr.getStatus()).toBe(503);
    expect(thrownErr.requestId).toBe('req-timeout-8s');
    expect(attempts).toBe(1); // Proves no retry on timeout
  });

  // ── 9. X-Request-Id header present and correct ─────────────────────────────
  it('9. X-Request-Id header present and correct in outgoing HTTP headers', async () => {
    let capturedHeaders: Record<string, any> = {};
    jest.spyOn(http, 'request').mockImplementation(((url: any, options: any, cb: any) => {
      capturedHeaders = options.headers;
      const req = new EventEmitter() as any;
      req.write = jest.fn();
      req.end = jest.fn();
      req.destroy = jest.fn();

      process.nextTick(() => {
        const res = new EventEmitter() as any;
        res.statusCode = 200;
        res.setEncoding = jest.fn();
        cb(res);
        res.emit('data', JSON.stringify(mockPythonDecision));
        res.emit('end');
      });

      return req;
    }) as any);

    await adapter.evaluate('pay_000001_a1', 'req-header-check-999', false);

    expect(capturedHeaders).toBeDefined();
    expect(capturedHeaders['x-request-id']).toBe('req-header-check-999');
    expect(capturedHeaders['Content-Type']).toBe('application/json');
  });

  // ── 10. Request body shape ─────────────────────────────────────────────────
  it('10. request body shape: contains payment_id, request_id, and force_recompute', async () => {
    let capturedBody = '';
    jest.spyOn(http, 'request').mockImplementation(((url: any, options: any, cb: any) => {
      const req = new EventEmitter() as any;
      req.write = jest.fn((chunk: string) => {
        capturedBody += chunk;
      });
      req.end = jest.fn();
      req.destroy = jest.fn();

      process.nextTick(() => {
        const res = new EventEmitter() as any;
        res.statusCode = 200;
        res.setEncoding = jest.fn();
        cb(res);
        res.emit('data', JSON.stringify(mockPythonDecision));
        res.emit('end');
      });

      return req;
    }) as any);

    await adapter.evaluate('pay_000001_a1', 'req-body-shape-test', false);

    const parsed = JSON.parse(capturedBody);
    expect(parsed).toEqual({
      payment_id: 'pay_000001_a1',
      request_id: 'req-body-shape-test',
      force_recompute: false,
    });
  });

  // ── 11. Cached decision behavior passed through unchanged ──────────────────
  it('11. cached decision behavior passed through from Python unchanged', async () => {
    const cachedDecision: PythonDecisionResult = {
      ...mockPythonDecision,
      payment_id: 'pay_000002_a2',
      decision_source: 'cache',
      reasoning: 'Loaded from SQLite decision_audit cache',
    };

    jest.spyOn(http, 'request').mockImplementation(((url: any, options: any, cb: any) => {
      const req = new EventEmitter() as any;
      req.write = jest.fn();
      req.end = jest.fn();
      req.destroy = jest.fn();

      process.nextTick(() => {
        const res = new EventEmitter() as any;
        res.statusCode = 200;
        res.setEncoding = jest.fn();
        cb(res);
        res.emit('data', JSON.stringify(cachedDecision));
        res.emit('end');
      });

      return req;
    }) as any);

    const result = await adapter.evaluate('pay_000002_a2', 'req-cache-check', false);

    expect(result).toEqual(cachedDecision);
    expect(result.decision_source).toBe('cache');
    expect(result.reasoning).toBe('Loaded from SQLite decision_audit cache');
  });

  // ── 12. force_recompute propagation ────────────────────────────────────────
  it('12. force_recompute propagation: correctly propagates true and false in POST body', async () => {
    const capturedPayloads: any[] = [];
    jest.spyOn(http, 'request').mockImplementation(((url: any, options: any, cb: any) => {
      const req = new EventEmitter() as any;
      req.write = jest.fn((chunk: string) => {
        capturedPayloads.push(JSON.parse(chunk));
      });
      req.end = jest.fn();
      req.destroy = jest.fn();

      process.nextTick(() => {
        const res = new EventEmitter() as any;
        res.statusCode = 200;
        res.setEncoding = jest.fn();
        cb(res);
        res.emit('data', JSON.stringify(mockPythonDecision));
        res.emit('end');
      });

      return req;
    }) as any);

    await adapter.evaluate('pay_000001_a1', 'req-force-false', false);
    await adapter.evaluate('pay_000001_a1', 'req-force-true', true);

    expect(capturedPayloads[0].force_recompute).toBe(false);
    expect(capturedPayloads[1].force_recompute).toBe(true);
  });

  // ── 13. Shared agent reused across multiple calls [FIX-8 setup] ────────────
  it('13. shared agent is reused across multiple calls: same agent instance, no new construction', async () => {
    const usedAgents: any[] = [];
    jest.spyOn(http, 'request').mockImplementation(((url: any, options: any, cb: any) => {
      usedAgents.push(options.agent);
      const req = new EventEmitter() as any;
      req.write = jest.fn();
      req.end = jest.fn();
      req.destroy = jest.fn();

      process.nextTick(() => {
        const res = new EventEmitter() as any;
        res.statusCode = 200;
        res.setEncoding = jest.fn();
        cb(res);
        res.emit('data', JSON.stringify(mockPythonDecision));
        res.emit('end');
      });

      return req;
    }) as any);

    await adapter.evaluate('pay_000001_a1', 'req-agent-1', false);
    await adapter.evaluate('pay_000002_a2', 'req-agent-2', false);
    await adapter.evaluate('pay_000003_a3', 'req-agent-3', false);

    expect(usedAgents.length).toBe(3);
    // Strict instance equality: exact same agent reused across all 3 calls
    expect(usedAgents[0]).toBe(adapter.getHttpAgent());
    expect(usedAgents[1]).toBe(adapter.getHttpAgent());
    expect(usedAgents[2]).toBe(adapter.getHttpAgent());

    const agentOptions = (adapter.getHttpAgent() as any).options;
    expect(agentOptions?.keepAlive ?? (adapter.getHttpAgent() as any).keepAlive).toBe(true);

    // Assert OnModuleDestroy cleans up the shared agents
    const destroySpy = jest.spyOn(adapter.getHttpAgent(), 'destroy');
    adapter.onModuleDestroy();
    expect(destroySpy).toHaveBeenCalled();
  });
});
