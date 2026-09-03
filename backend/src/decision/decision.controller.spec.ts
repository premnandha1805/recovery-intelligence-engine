import { Test, TestingModule } from '@nestjs/testing';
import { HttpStatus, INestApplication, Logger, ValidationPipe } from '@nestjs/common';
import * as request from 'supertest';
import * as http from 'http';
import { EventEmitter } from 'events';
import { ConfigService } from '@nestjs/config';
import { HealthModule } from '../health/health.module';
import { DecisionModule } from './decision.module';
import { DecisionResponseDto } from './dto/decision-response.dto';
import { HttpExceptionFilter } from '../common/filters/http-exception.filter';
import { RequestIdMiddleware } from '../common/middleware/request-id.middleware';
import {
  DecisionEngineAdapter,
  PythonDecisionResult,
} from '../decision-engine/decision-engine.adapter';
import {
  DecisionEngineService,
} from '../decision-engine/decision-engine.service';
import {
  DecisionEngineErrorException,
  DecisionEngineTimeoutException,
  DecisionEngineUnavailableException,
} from '../common/exceptions/decision-engine.exceptions';

describe('DecisionController & HealthController (DAY 7H Final Public API Regression Suite)', () => {
  let app: INestApplication;
  let adapter: DecisionEngineAdapter;
  let decisionEngineService: DecisionEngineService;

  const mockPythonDecision: PythonDecisionResult = {
    payment_id: 'pay_000001_a1',
    model_decision: 'RETRY_NUDGE',
    llm_decision: 'RETRY_NUDGE',
    guardrail_overridden: false,
    guardrail_reason: null,
    final_action: 'RETRY_NUDGE',
    confidence: 0.85,
    risk_level: 'LOW',
    reasoning: 'Evaluation succeeded with high confidence',
    decision_source: 'FOUNDRY_REASONING',
    request_id: 'req-mock-default-123',
  };

  beforeAll(async () => {
    const moduleRef: TestingModule = await Test.createTestingModule({
      imports: [HealthModule, DecisionModule],
    })
      .overrideProvider(DecisionEngineAdapter)
      .useValue({
        evaluate: jest
          .fn()
          .mockImplementation(
            async (pid: string, reqId: string, forceRecompute: boolean = false) => ({
              ...mockPythonDecision,
              payment_id: pid,
              request_id: reqId,
            }),
          ),
        getTimeoutMs: jest.fn().mockReturnValue(8000),
      })
      .overrideProvider(DecisionEngineService)
      .useValue({
        checkHealth: jest.fn().mockResolvedValue('ok'),
        getHealthCheckTimeoutMs: jest.fn().mockReturnValue(1000),
        getTimeoutMs: jest.fn().mockReturnValue(8000),
      })
      .compile();

    adapter = moduleRef.get<DecisionEngineAdapter>(DecisionEngineAdapter);
    decisionEngineService = moduleRef.get<DecisionEngineService>(DecisionEngineService);

    app = moduleRef.createNestApplication();
    app.use((req: any, res: any, next: any) => {
      new RequestIdMiddleware().use(req, res, next);
    });
    app.useGlobalPipes(
      new ValidationPipe({
        whitelist: true,
        forbidNonWhitelisted: true,
        transform: true,
      }),
    );
    app.useGlobalFilters(new HttpExceptionFilter());
    await app.init();
  });

  afterAll(async () => {
    await app.close();
  });

  beforeEach(() => {
    jest.clearAllMocks();
  });

  // ── A. HEALTH — HEALTHY DEPENDENCY ─────────────────────────────────────────
  describe('A. HEALTH — HEALTHY DEPENDENCY', () => {
    it('should return HTTP 200, status ok, and no credential or stack trace leakage', async () => {
      jest.spyOn(decisionEngineService, 'checkHealth').mockResolvedValueOnce('ok');

      const response = await request(app.getHttpServer())
        .get('/health')
        .expect(200);

      expect(response.body).toEqual({
        status: 'ok',
        service: 'recovery-intelligence-api',
        dependencies: {
          decision_engine: 'ok',
        },
      });

      const jsonStr = JSON.stringify(response.body);
      expect(jsonStr).not.toContain('stack');
      expect(jsonStr).not.toContain('trace');
      expect(jsonStr).not.toContain('password');
      expect(jsonStr).not.toContain('secret');
      expect(jsonStr).not.toContain('key');
    });
  });

  // ── B. HEALTH — DEPENDENCY UNAVAILABLE ──────────────────────────────────────
  describe('B. HEALTH — DEPENDENCY UNAVAILABLE', () => {
    it('should return HTTP 503 and status degraded with decision_engine unreachable', async () => {
      jest.spyOn(decisionEngineService, 'checkHealth').mockResolvedValueOnce('unreachable');

      const response = await request(app.getHttpServer())
        .get('/health')
        .expect(HttpStatus.SERVICE_UNAVAILABLE);

      expect(response.body).toEqual({
        status: 'degraded',
        service: 'recovery-intelligence-api',
        dependencies: {
          decision_engine: 'unreachable',
        },
      });
      expect(response.body.status).not.toBe('ok');
    });
  });

  // ── C. HEALTH — DEPENDENCY TIMEOUT ─────────────────────────────────────────
  describe('C. HEALTH — DEPENDENCY TIMEOUT', () => {
    it('should return HTTP 503 and status degraded with decision_engine timeout', async () => {
      jest.spyOn(decisionEngineService, 'checkHealth').mockResolvedValueOnce('timeout');

      const response = await request(app.getHttpServer())
        .get('/health')
        .expect(HttpStatus.SERVICE_UNAVAILABLE);

      expect(response.body).toEqual({
        status: 'degraded',
        service: 'recovery-intelligence-api',
        dependencies: {
          decision_engine: 'timeout',
        },
      });
      expect(response.body.status).not.toBe('ok');
    });
  });

  // ── C2. HEALTH TIMEOUT USES ITS OWN SHORT CONFIG, NOT DECISION TIMEOUT ──────
  describe('C2. HEALTH TIMEOUT USES ITS OWN SHORT CONFIG, NOT THE DECISION TIMEOUT', () => {
    it('should prove health check uses isolated short timeout (~1s), distinct from 8s decision timeout', async () => {
      const realModule = await Test.createTestingModule({
        providers: [
          DecisionEngineService,
          DecisionEngineAdapter,
          {
            provide: ConfigService,
            useValue: {
              get: jest.fn((key: string, def?: any) => {
                if (key === 'HEALTH_CHECK_TIMEOUT_MS') return 1000;
                if (key === 'DECISION_ENGINE_TIMEOUT_MS') return 8000;
                if (key === 'DECISION_ENGINE_URL') return 'http://localhost:8000';
                return def;
              }),
            },
          },
        ],
      }).compile();

      const realHealthService = realModule.get<DecisionEngineService>(DecisionEngineService);
      const realAdapter = realModule.get<DecisionEngineAdapter>(DecisionEngineAdapter);

      // 1. Configured timeout values are explicitly distinct
      const healthTimeout = realHealthService.getHealthCheckTimeoutMs();
      const decisionTimeout = realAdapter.getTimeoutMs();

      expect(healthTimeout).toBe(1000);
      expect(decisionTimeout).toBe(8000);
      expect(healthTimeout).not.toBe(decisionTimeout);

      // 2. Assert via http.request options that checkHealth() actually uses its own short timeout
      let capturedOptions: any;
      jest.spyOn(http, 'request').mockImplementationOnce(((url: any, options: any, cb: any) => {
        capturedOptions = options;
        const req = new EventEmitter() as any;
        req.write = jest.fn();
        req.end = jest.fn();
        req.destroy = jest.fn();

        process.nextTick(() => {
          const res = new EventEmitter() as any;
          res.statusCode = 200;
          res.setEncoding = jest.fn();
          cb(res);
          res.emit('data', JSON.stringify({ status: 'ok' }));
          res.emit('end');
        });

        return req;
      }) as any);

      await realHealthService.checkHealth();

      expect(capturedOptions).toBeDefined();
      expect(capturedOptions.timeout).toBe(1000);
      expect(capturedOptions.timeout).toBe(healthTimeout);
      expect(capturedOptions.timeout).not.toBe(decisionTimeout);

      realHealthService.onModuleDestroy();
      realAdapter.onModuleDestroy();
    });
  });

  // ── D. NORMAL DECISION ─────────────────────────────────────────────────────
  describe('D. NORMAL DECISION', () => {
    it('should return HTTP 200 and the exact stable 11-field response contract', async () => {
      jest.spyOn(adapter, 'evaluate').mockResolvedValueOnce({
        payment_id: 'pay_000001_a1',
        model_decision: 'RETRY_NUDGE',
        llm_decision: 'RETRY_NUDGE',
        guardrail_overridden: false,
        guardrail_reason: null,
        final_action: 'RETRY_NUDGE',
        confidence: 0.91,
        risk_level: 'LOW',
        reasoning: 'Evaluation succeeded with high confidence',
        decision_source: 'FOUNDRY_REASONING',
        request_id: 'req-contract-check-1',
      });

      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({
          payment_id: 'pay_000001_a1',
          force_recompute: false,
        })
        .expect(200);

      const body: DecisionResponseDto = response.body;

      expect(body.payment_id).toBe('pay_000001_a1');
      expect(body.model_decision).toBe('RETRY_NUDGE');
      expect(body.llm_decision).toBe('RETRY_NUDGE');
      expect(body.guardrail_overridden).toBe(false);
      expect(body.guardrail_reason).toBeNull();
      expect(body.final_action).toBe('RETRY_NUDGE');
      expect(body.confidence).toBe(0.91);
      expect(body.risk_level).toBe('LOW');
      expect(body.reasoning).toBe('Evaluation succeeded with high confidence');
      expect(body.decision_source).toBe('FOUNDRY_REASONING');
      expect(body.request_id).toBeDefined();

      // Strict contract check: exactly the 11 stable response fields
      expect(Object.keys(body).sort()).toEqual(
        [
          'payment_id',
          'model_decision',
          'llm_decision',
          'guardrail_overridden',
          'guardrail_reason',
          'final_action',
          'confidence',
          'risk_level',
          'reasoning',
          'decision_source',
          'request_id',
        ].sort(),
      );
    });
  });

  // ── E. DTO VALIDATION ──────────────────────────────────────────────────────
  describe('E. DTO VALIDATION', () => {
    it('should reject missing payment_id with 400 VALIDATION_ERROR and exact envelope', async () => {
      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({})
        .expect(400);

      expect(response.body).toEqual({
        error: {
          code: 'VALIDATION_ERROR',
          message: expect.stringContaining('payment_id should not be empty'),
          request_id: expect.any(String),
        },
      });
      expect(Object.keys(response.body)).toEqual(['error']);
      expect(Object.keys(response.body.error).sort()).toEqual(
        ['code', 'message', 'request_id'].sort(),
      );
    });

    it('should reject empty payment_id with 400 VALIDATION_ERROR and exact envelope', async () => {
      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({ payment_id: '' })
        .expect(400);

      expect(response.body.error.code).toBe('VALIDATION_ERROR');
      expect(response.body.error.request_id).toBeDefined();
    });

    it('should reject malformed payment_id with 400 VALIDATION_ERROR and exact envelope', async () => {
      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({ payment_id: 'invalid_format_xyz' })
        .expect(400);

      expect(response.body.error.code).toBe('VALIDATION_ERROR');
      expect(response.body.error.message).toContain('payment_id must match the format pay_XXXXXX_aY');
      expect(response.body.error.request_id).toBeDefined();
    });
  });

  // ── F. UNEXPECTED PAYLOAD FIELDS ───────────────────────────────────────────
  describe('F. UNEXPECTED PAYLOAD FIELDS', () => {
    it('should reject extra payload field net_value with 400 and not silently strip it', async () => {
      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({
          payment_id: 'pay_000001_a1',
          net_value: 100,
        })
        .expect(400);

      expect(response.body.error.code).toBe('VALIDATION_ERROR');
      expect(response.body.error.message).toContain('property net_value should not exist');
      expect(Object.keys(response.body)).toEqual(['error']);
    });

    it('should reject unexpected ML field model_decision with 400', async () => {
      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({
          payment_id: 'pay_000001_a1',
          model_decision: 'RETRY',
        })
        .expect(400);

      expect(response.body.error.code).toBe('VALIDATION_ERROR');
      expect(response.body.error.message).toContain('property model_decision should not exist');
    });

    it('should reject unexpected guardrail field guardrail_overridden with 400', async () => {
      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({
          payment_id: 'pay_000001_a1',
          guardrail_overridden: true,
        })
        .expect(400);

      expect(response.body.error.code).toBe('VALIDATION_ERROR');
      expect(response.body.error.message).toContain('property guardrail_overridden should not exist');
    });
  });

  // ── G. UNKNOWN PAYMENT ─────────────────────────────────────────────────────
  describe('G. UNKNOWN PAYMENT', () => {
    it('should return controlled HTTP 200 response with final_action WAIT and decision_source error_path', async () => {
      const unknownPaymentId = 'pay_999999_a1';
      const customReqId = 'req-unknown-pay-test-999';

      jest.spyOn(adapter, 'evaluate').mockResolvedValueOnce({
        payment_id: unknownPaymentId,
        model_decision: 'N/A — error path',
        llm_decision: 'N/A — error path',
        guardrail_overridden: false,
        guardrail_reason: 'Bypassed due to error: PAYMENT_NOT_FOUND',
        final_action: 'WAIT',
        confidence: 0.0,
        risk_level: 'none',
        reasoning: 'Error: PAYMENT_NOT_FOUND',
        decision_source: 'error_path',
        request_id: customReqId,
      });

      const response = await request(app.getHttpServer())
        .post('/decisions')
        .set('x-request-id', customReqId)
        .send({ payment_id: unknownPaymentId })
        .expect(200);

      expect(response.body.payment_id).toBe(unknownPaymentId);
      expect(response.body.final_action).toBe('WAIT');
      expect(response.body.decision_source).toBe('error_path');
      expect(response.body.request_id).toBe(customReqId);

      const jsonStr = JSON.stringify(response.body);
      expect(jsonStr).not.toContain('stack');
      expect(jsonStr).not.toContain('trace');
    });
  });

  // ── H. DECISION ENGINE UNAVAILABLE ─────────────────────────────────────────
  describe('H. DECISION ENGINE UNAVAILABLE', () => {
    it('should map to HTTP 503 DECISION_ENGINE_UNAVAILABLE and match X-Request-Id', async () => {
      const customReqId = 'req-unavail-test-503';
      jest
        .spyOn(adapter, 'evaluate')
        .mockRejectedValueOnce(
          new DecisionEngineUnavailableException('Connection refused 127.0.0.1:8000'),
        );

      const response = await request(app.getHttpServer())
        .post('/decisions')
        .set('x-request-id', customReqId)
        .send({ payment_id: 'pay_000001_a1' })
        .expect(503);

      expect(response.body.error).toEqual({
        code: 'DECISION_ENGINE_UNAVAILABLE',
        message: 'Connection refused 127.0.0.1:8000',
        request_id: customReqId,
      });
      expect(response.headers['x-request-id']).toBe(customReqId);
    });
  });

  // ── I. DECISION ENGINE TIMEOUT ─────────────────────────────────────────────
  describe('I. DECISION ENGINE TIMEOUT', () => {
    it('should map to HTTP 503 DECISION_ENGINE_TIMEOUT with preserved request_id correlation', async () => {
      const customReqId = 'req-timeout-test-503';
      jest
        .spyOn(adapter, 'evaluate')
        .mockRejectedValueOnce(new DecisionEngineTimeoutException());

      const response = await request(app.getHttpServer())
        .post('/decisions')
        .set('x-request-id', customReqId)
        .send({ payment_id: 'pay_000001_a1' })
        .expect(503);

      expect(response.body.error.code).toBe('DECISION_ENGINE_TIMEOUT');
      expect(response.body.error.message).toBe('Python decision engine request timed out');
      expect(response.body.error.request_id).toBe(customReqId);
      expect(response.headers['x-request-id']).toBe(customReqId);
    });
  });

  // ── J. UPSTREAM DECISION ENGINE ERROR ──────────────────────────────────────
  describe('J. UPSTREAM DECISION ENGINE ERROR', () => {
    it('should map to HTTP 502 DECISION_ENGINE_ERROR with preserved request_id correlation', async () => {
      const customReqId = 'req-upstream-502-trace';
      jest
        .spyOn(adapter, 'evaluate')
        .mockRejectedValueOnce(
          new DecisionEngineErrorException('Upstream HTTP 500 internal crash'),
        );

      const response = await request(app.getHttpServer())
        .post('/decisions')
        .set('x-request-id', customReqId)
        .send({ payment_id: 'pay_000001_a1' })
        .expect(502);

      expect(response.body.error).toEqual({
        code: 'DECISION_ENGINE_ERROR',
        message: 'Upstream HTTP 500 internal crash',
        request_id: customReqId,
      });
      expect(response.headers['x-request-id']).toBe(customReqId);
    });
  });

  // ── K. GUARDRAIL OVERRIDE ──────────────────────────────────────────────────
  describe('K. GUARDRAIL OVERRIDE', () => {
    it('should preserve guardrail_overridden, guardrail_reason, and final_action without reinterpreting', async () => {
      jest.spyOn(adapter, 'evaluate').mockResolvedValueOnce({
        payment_id: 'pay_000002_a2',
        model_decision: 'RETRY_NUDGE',
        llm_decision: 'RETRY_NUDGE',
        guardrail_overridden: true,
        guardrail_reason: 'Maximum interventions in 7-day window reached',
        final_action: 'WAIT',
        confidence: 0.9,
        risk_level: 'MEDIUM',
        reasoning: 'Model recommended retry nudge but customer reached weekly intervention cap',
        decision_source: 'FOUNDRY_REASONING',
        request_id: 'req-guardrail-test-1',
      });

      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({ payment_id: 'pay_000002_a2' })
        .expect(200);

      expect(response.body.guardrail_overridden).toBe(true);
      expect(response.body.guardrail_reason).toBe(
        'Maximum interventions in 7-day window reached',
      );
      expect(response.body.final_action).toBe('WAIT');
      expect(response.body.model_decision).toBe('RETRY_NUDGE');
      expect(response.body.llm_decision).toBe('RETRY_NUDGE');
    });
  });

  // ── L. REQUEST ID — CLIENT PROVIDED ────────────────────────────────────────
  describe('L. REQUEST ID — CLIENT PROVIDED', () => {
    it('should preserve client-supplied X-Request-Id end-to-end across headers, body, and adapter', async () => {
      const clientReqId = 'req-client-123';
      const evaluateSpy = jest.spyOn(adapter, 'evaluate').mockResolvedValueOnce({
        ...mockPythonDecision,
        payment_id: 'pay_000001_a1',
        request_id: clientReqId,
      });

      const response = await request(app.getHttpServer())
        .post('/decisions')
        .set('x-request-id', clientReqId)
        .send({ payment_id: 'pay_000001_a1' })
        .expect(200);

      expect(response.headers['x-request-id']).toBe(clientReqId);
      expect(response.body.request_id).toBe(clientReqId);
      expect(evaluateSpy).toHaveBeenCalledWith(
        'pay_000001_a1',
        clientReqId,
        false,
      );
    });
  });

  // ── M. REQUEST ID — GENERATED ──────────────────────────────────────────────
  describe('M. REQUEST ID — GENERATED', () => {
    it('should unconditionally generate and return a valid RFC 4122 UUID when header is absent', async () => {
      let capturedAdapterReqId: string | undefined;
      jest
        .spyOn(adapter, 'evaluate')
        .mockImplementationOnce(async (pid, reqId, force) => {
          capturedAdapterReqId = reqId;
          return {
            ...mockPythonDecision,
            payment_id: pid,
            request_id: reqId,
          };
        });

      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({ payment_id: 'pay_000001_a1' })
        .expect(200);

      const generatedId = response.body.request_id;
      expect(generatedId).toBeDefined();
      expect(response.headers['x-request-id']).toBe(generatedId);
      expect(capturedAdapterReqId).toBe(generatedId);

      // Unconditionally assert valid RFC 4122 UUID format
      const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
      expect(generatedId).toMatch(uuidRegex);
    });
  });

  // ── N. FORCE RECOMPUTE ─────────────────────────────────────────────────────
  describe('N. FORCE RECOMPUTE', () => {
    it('should forward force_recompute=false to adapter by default', async () => {
      const evaluateSpy = jest.spyOn(adapter, 'evaluate');

      await request(app.getHttpServer())
        .post('/decisions')
        .send({
          payment_id: 'pay_000001_a1',
          force_recompute: false,
        })
        .expect(200);

      expect(evaluateSpy).toHaveBeenCalledWith(
        'pay_000001_a1',
        expect.any(String),
        false,
      );
    });

    it('should forward force_recompute=true to adapter when explicitly specified', async () => {
      const evaluateSpy = jest.spyOn(adapter, 'evaluate');

      await request(app.getHttpServer())
        .post('/decisions')
        .send({
          payment_id: 'pay_000001_a1',
          force_recompute: true,
        })
        .expect(200);

      expect(evaluateSpy).toHaveBeenCalledWith(
        'pay_000001_a1',
        expect.any(String),
        true,
      );
    });
  });

  // ── O. ERROR ENVELOPE CONSISTENCY ──────────────────────────────────────────
  describe('O. ERROR ENVELOPE CONSISTENCY', () => {
    it('should return the exact same error envelope { error: { code, message, request_id } } across all failure classes', async () => {
      // 1. 400 VALIDATION_ERROR
      const res400 = await request(app.getHttpServer())
        .post('/decisions')
        .send({})
        .expect(400);
      expect(Object.keys(res400.body)).toEqual(['error']);
      expect(Object.keys(res400.body.error).sort()).toEqual(
        ['code', 'message', 'request_id'].sort(),
      );
      expect((res400.body as any).message).toBeUndefined();

      // 2. 503 DECISION_ENGINE_UNAVAILABLE
      jest
        .spyOn(adapter, 'evaluate')
        .mockRejectedValueOnce(new DecisionEngineUnavailableException());
      const res503Unavail = await request(app.getHttpServer())
        .post('/decisions')
        .send({ payment_id: 'pay_000001_a1' })
        .expect(503);
      expect(Object.keys(res503Unavail.body)).toEqual(['error']);
      expect(Object.keys(res503Unavail.body.error).sort()).toEqual(
        ['code', 'message', 'request_id'].sort(),
      );
      expect((res503Unavail.body as any).message).toBeUndefined();

      // 3. 503 DECISION_ENGINE_TIMEOUT
      jest
        .spyOn(adapter, 'evaluate')
        .mockRejectedValueOnce(new DecisionEngineTimeoutException());
      const res503Timeout = await request(app.getHttpServer())
        .post('/decisions')
        .send({ payment_id: 'pay_000001_a1' })
        .expect(503);
      expect(Object.keys(res503Timeout.body)).toEqual(['error']);
      expect(Object.keys(res503Timeout.body.error).sort()).toEqual(
        ['code', 'message', 'request_id'].sort(),
      );
      expect((res503Timeout.body as any).message).toBeUndefined();

      // 4. 502 DECISION_ENGINE_ERROR
      jest
        .spyOn(adapter, 'evaluate')
        .mockRejectedValueOnce(new DecisionEngineErrorException());
      const res502 = await request(app.getHttpServer())
        .post('/decisions')
        .send({ payment_id: 'pay_000001_a1' })
        .expect(502);
      expect(Object.keys(res502.body)).toEqual(['error']);
      expect(Object.keys(res502.body.error).sort()).toEqual(
        ['code', 'message', 'request_id'].sort(),
      );
      expect((res502.body as any).message).toBeUndefined();

      // 5. 500 INTERNAL_ERROR
      jest
        .spyOn(adapter, 'evaluate')
        .mockRejectedValueOnce(new Error('Unexpected unhandled database failure'));
      const res500 = await request(app.getHttpServer())
        .post('/decisions')
        .send({ payment_id: 'pay_000001_a1' })
        .expect(500);
      expect(Object.keys(res500.body)).toEqual(['error']);
      expect(Object.keys(res500.body.error).sort()).toEqual(
        ['code', 'message', 'request_id'].sort(),
      );
      expect((res500.body as any).message).toBeUndefined();
    });
  });

  // ── P. SECURITY / LEAKAGE ──────────────────────────────────────────────────
  describe('P. SECURITY / LEAKAGE', () => {
    it('should never leak API keys, bearer tokens, Azure strings, filesystem paths, or stack traces in responses or logs', async () => {
      const errorSpy = jest.spyOn(Logger.prototype, 'error');
      const customReqId = 'req-leak-test-999';
      const fakeSecretPayload =
        'Upstream crash sk-live-1234567890abcdef with Bearer ya29.mockToken12345 and DefaultEndpointsProtocol=https;AccountName=myacc;AccountKey=fakeKey987; at D:\\recovery-intelligence-engine\\secrets.env and /var/run/secrets/api.sock\n at Object.<anonymous> (D:\\app\\src\\index.ts:25:10)';

      jest
        .spyOn(adapter, 'evaluate')
        .mockRejectedValueOnce(new DecisionEngineErrorException(fakeSecretPayload));

      const response = await request(app.getHttpServer())
        .post('/decisions')
        .set('x-request-id', customReqId)
        .send({ payment_id: 'pay_000001_a1' })
        .expect(502);

      const responseBodyStr = JSON.stringify(response.body);

      // Verify response body does NOT leak secrets or paths
      expect(responseBodyStr).not.toContain('sk-live-1234567890abcdef');
      expect(responseBodyStr).not.toContain('ya29.mockToken12345');
      expect(responseBodyStr).not.toContain('fakeKey987');
      expect(responseBodyStr).not.toContain('D:\\recovery-intelligence-engine');
      expect(responseBodyStr).not.toContain('/var/run/secrets');
      expect(responseBodyStr).not.toContain('at Object.<anonymous>');

      // Verify logs do NOT leak secrets or paths
      const logCalls = errorSpy.mock.calls
        .map((c) => String(c[0]))
        .filter((str) => str.includes(customReqId));

      const combinedLogs = logCalls.join(' ');
      expect(combinedLogs).not.toContain('sk-live-1234567890abcdef');
      expect(combinedLogs).not.toContain('ya29.mockToken12345');
      expect(combinedLogs).not.toContain('fakeKey987');
      expect(combinedLogs).not.toContain('D:\\recovery-intelligence-engine');
      expect(combinedLogs).not.toContain('/var/run/secrets');
      expect(combinedLogs).not.toContain('at Object.<anonymous>');
    });
  });

  // ── Q. CACHED DECISION PASSTHROUGH ─────────────────────────────────────────
  describe('Q. CACHED DECISION PASSTHROUGH', () => {
    it('should pass through decision_source=cache unchanged with identical response shape and make exactly one adapter call', async () => {
      const evaluateSpy = jest.spyOn(adapter, 'evaluate').mockResolvedValueOnce({
        payment_id: 'pay_000001_a1',
        model_decision: 'RETRY_NUDGE',
        llm_decision: 'RETRY_NUDGE',
        guardrail_overridden: false,
        guardrail_reason: null,
        final_action: 'RETRY_NUDGE',
        confidence: 0.85,
        risk_level: 'LOW',
        reasoning: 'Loaded from SQLite decision_audit cache',
        decision_source: 'cache',
        request_id: 'req-cached-passthrough-1',
      });

      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({
          payment_id: 'pay_000001_a1',
          force_recompute: false,
        })
        .expect(200);

      // 1. HTTP 200 and identical 11-field response contract
      expect(Object.keys(response.body).sort()).toEqual(
        [
          'payment_id',
          'model_decision',
          'llm_decision',
          'guardrail_overridden',
          'guardrail_reason',
          'final_action',
          'confidence',
          'risk_level',
          'reasoning',
          'decision_source',
          'request_id',
        ].sort(),
      );

      // 2. decision_source passed through unchanged
      expect(response.body.decision_source).toBe('cache');
      expect(response.body.reasoning).toBe('Loaded from SQLite decision_audit cache');

      // 3. NestJS makes exactly one call to the adapter (no internal cache, retry, or dedup logic)
      expect(evaluateSpy).toHaveBeenCalledTimes(1);
    });

    it('Scenario 6: New payment with caller-supplied features is validated and forwarded to adapter', async () => {
      const evaluateSpy = jest.spyOn(adapter, 'evaluate').mockResolvedValueOnce({
        ...mockPythonDecision,
        payment_id: 'pay_999999_a1',
        decision_source: 'FOUNDRY_REASONING',
      });

      const validFeatures = {
        amount: 1499.0,
        attempt_number: 1,
        dynamic_success_rate: 0.65,
        cumulative_failures: 0,
        consecutive_failed_cycles: 0,
        notification_engagement_score: 0.8,
        contact_response_score: 0.5,
        payment_method: 'card',
        failure_reason: 'insufficient_funds',
      };

      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({
          payment_id: 'pay_999999_a1',
          features: validFeatures,
        })
        .expect(200);

      expect(response.body.payment_id).toBe('pay_999999_a1');
      expect(response.body.decision_source).toBe('FOUNDRY_REASONING');
      expect(evaluateSpy).toHaveBeenCalledWith(
        'pay_999999_a1',
        expect.any(String),
        false,
        expect.objectContaining({
          amount: 1499.0,
          payment_method: 'card',
          failure_reason: 'insufficient_funds',
        }),
      );
    });

    it('Scenario 7: Reject new payment request when features contains forbidden or undeclared fields', async () => {
      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({
          payment_id: 'pay_999999_a1',
          features: {
            amount: 1499.0,
            attempt_number: 1,
            dynamic_success_rate: 0.65,
            cumulative_failures: 0,
            consecutive_failed_cycles: 0,
            notification_engagement_score: 0.8,
            contact_response_score: 0.5,
            payment_method: 'card',
            failure_reason: 'insufficient_funds',
            p_success_retry: 0.85, // Forbidden simulator field
          },
        })
        .expect(400);

      expect(response.body.error.code).toBe('VALIDATION_ERROR');
    });
  });
});
