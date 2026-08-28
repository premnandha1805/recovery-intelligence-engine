import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication, ValidationPipe } from '@nestjs/common';
import * as request from 'supertest';
import { DecisionModule } from './decision.module';
import { DecisionResponseDto } from './dto/decision-response.dto';
import { HttpExceptionFilter } from '../common/filters/http-exception.filter';
import {
  DecisionEngineAdapter,
  PythonDecisionResult,
} from '../decision-engine/decision-engine.adapter';
import {
  DecisionEngineErrorException,
  DecisionEngineTimeoutException,
  DecisionEngineUnavailableException,
} from '../common/exceptions/decision-engine.exceptions';

describe('DecisionController (Day 7F e2e contract, error envelope & request correlation)', () => {
  let app: INestApplication;
  let adapter: DecisionEngineAdapter;

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
      imports: [DecisionModule],
    })
      .overrideProvider(DecisionEngineAdapter)
      .useValue({
        evaluate: jest.fn().mockImplementation(async (pid: string, reqId: string) => ({
          ...mockPythonDecision,
          payment_id: pid,
          request_id: reqId,
        })),
      })
      .compile();

    adapter = moduleRef.get<DecisionEngineAdapter>(DecisionEngineAdapter);

    app = moduleRef.createNestApplication();
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

  // ── Contract & Success Tests ────────────────────────────────────────────────

  it('should accept valid payload with required payment_id only and return stable DecisionResponseDto', async () => {
    const response = await request(app.getHttpServer())
      .post('/decisions')
      .send({
        payment_id: 'pay_000001_a1',
      })
      .expect(200);

    const body: DecisionResponseDto = response.body;

    expect(body.payment_id).toBe('pay_000001_a1');
    expect(typeof body.model_decision).toBe('string');
    expect(typeof body.llm_decision).toBe('string');
    expect(typeof body.guardrail_overridden).toBe('boolean');
    expect(
      body.guardrail_reason === null || typeof body.guardrail_reason === 'string',
    ).toBe(true);
    expect(typeof body.final_action).toBe('string');
    expect(
      body.confidence === null || typeof body.confidence === 'number',
    ).toBe(true);
    expect(body.risk_level === null || typeof body.risk_level === 'string').toBe(
      true,
    );
    expect(body.reasoning === null || typeof body.reasoning === 'string').toBe(
      true,
    );
    expect(typeof body.decision_source).toBe('string');
    expect(typeof body.request_id).toBe('string');
    expect(body.request_id.length).toBeGreaterThan(0);

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

  it('should accept valid payload with optional force_recompute', async () => {
    const response = await request(app.getHttpServer())
      .post('/decisions')
      .send({
        payment_id: 'pay_000002_a2',
        force_recompute: true,
      })
      .expect(200);

    expect(response.body.payment_id).toBe('pay_000002_a2');
    expect(response.body.final_action).toBeDefined();
    expect(response.body.request_id).toBeDefined();
  });

  // ── Day 7F Tests A through M ────────────────────────────────────────────────

  describe('Day 7F Tests A through M', () => {
    // Test A: 400 validation error
    it('Test A: 400 validation error maps to code VALIDATION_ERROR and status 400', async () => {
      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({})
        .expect(400);

      expect(response.body.error).toBeDefined();
      expect(response.body.error.code).toBe('VALIDATION_ERROR');
      expect(response.body.error.message).toContain('payment_id should not be empty');
      expect(typeof response.body.error.request_id).toBe('string');
      expect(response.body.error.request_id.length).toBeGreaterThan(10);
    });

    // Test B: custom X-Request-Id preserved
    it('Test B: custom X-Request-Id preserved on validation error responses [FIX-10]', async () => {
      const customReqId = 'req-custom-validation-fail-400';
      const response = await request(app.getHttpServer())
        .post('/decisions')
        .set('x-request-id', customReqId)
        .send({})
        .expect(400);

      expect(response.body.error.code).toBe('VALIDATION_ERROR');
      expect(response.body.error.request_id).toBe(customReqId);
      expect(response.headers['x-request-id']).toBe(customReqId);
    });

    // Test C: generated request ID when absent
    it('Test C: generated request ID when absent on error response', async () => {
      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({ payment_id: 'invalid_format' })
        .expect(400);

      expect(response.body.error.code).toBe('VALIDATION_ERROR');
      expect(typeof response.body.error.request_id).toBe('string');
      expect(response.body.error.request_id.length).toBeGreaterThan(10);
      expect(response.headers['x-request-id']).toBe(response.body.error.request_id);
    });

    // Test D: 503 unavailable
    it('Test D: dependency unavailable maps to HTTP 503 with DECISION_ENGINE_UNAVAILABLE and request_id', async () => {
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

    // Test E: 503 timeout
    it('Test E: dependency timeout maps to HTTP 503 with DECISION_ENGINE_TIMEOUT and generated request_id', async () => {
      jest
        .spyOn(adapter, 'evaluate')
        .mockRejectedValueOnce(new DecisionEngineTimeoutException());

      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({ payment_id: 'pay_000001_a1' })
        .expect(503);

      expect(response.body.error.code).toBe('DECISION_ENGINE_TIMEOUT');
      expect(response.body.error.message).toBe('Python decision engine request timed out');
      expect(typeof response.body.error.request_id).toBe('string');
      expect(response.body.error.request_id.length).toBeGreaterThan(10);
      expect(response.headers['x-request-id']).toBe(response.body.error.request_id);
    });

    // Test F: 502 upstream error
    it('Test F: upstream controlled 5xx maps to HTTP 502 with DECISION_ENGINE_ERROR and request_id', async () => {
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

    // Test G: 500 unexpected error
    it('Test G: unexpected 500 error maps to INTERNAL_ERROR and never leaks the underlying cause', async () => {
      const customReqId = 'req-unexpected-500-test';
      jest
        .spyOn(adapter, 'evaluate')
        .mockRejectedValueOnce(
          new Error('Critical DB connection failure at /var/run/secrets/sql.sock: access denied for user root'),
        );

      const response = await request(app.getHttpServer())
        .post('/decisions')
        .set('x-request-id', customReqId)
        .send({ payment_id: 'pay_000001_a1' })
        .expect(500);

      expect(response.body.error).toEqual({
        code: 'INTERNAL_ERROR',
        message: 'Internal server error',
        request_id: customReqId,
      });
      expect(response.headers['x-request-id']).toBe(customReqId);

      // Verify underlying cause is NOT leaked
      const bodyStr = JSON.stringify(response.body);
      expect(bodyStr).not.toContain('Critical DB connection failure');
      expect(bodyStr).not.toContain('/var/run/secrets');
      expect(bodyStr).not.toContain('access denied');
    });

    // Test H: exact envelope keys
    it('Test H: exact envelope keys across all failure responses: body contains ONLY "error"', async () => {
      // 1. 400 validation error
      const res400 = await request(app.getHttpServer())
        .post('/decisions')
        .send({})
        .expect(400);

      expect(Object.keys(res400.body)).toEqual(['error']);
      expect(Object.keys(res400.body.error).sort()).toEqual(
        ['code', 'message', 'request_id'].sort(),
      );

      // 2. 503 unavailable
      jest
        .spyOn(adapter, 'evaluate')
        .mockRejectedValueOnce(new DecisionEngineUnavailableException());
      const res503 = await request(app.getHttpServer())
        .post('/decisions')
        .send({ payment_id: 'pay_000001_a1' })
        .expect(503);

      expect(Object.keys(res503.body)).toEqual(['error']);
      expect(Object.keys(res503.body.error).sort()).toEqual(
        ['code', 'message', 'request_id'].sort(),
      );

      // 3. 502 upstream error
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

      // 4. 500 unexpected error
      jest
        .spyOn(adapter, 'evaluate')
        .mockRejectedValueOnce(new Error('Unexpected unhandled error'));
      const res500 = await request(app.getHttpServer())
        .post('/decisions')
        .send({ payment_id: 'pay_000001_a1' })
        .expect(500);

      expect(Object.keys(res500.body)).toEqual(['error']);
      expect(Object.keys(res500.body.error).sort()).toEqual(
        ['code', 'message', 'request_id'].sort(),
      );
    });

    // Test I: no top-level message
    it('Test I: no top-level message key exists on any failure response body', async () => {
      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({
          payment_id: 'pay_000001_a1',
          net_value: 100, // Forbidden extra property
        })
        .expect(400);

      expect((response.body as any).message).toBeUndefined();
      expect(response.body.error).toBeDefined();
      expect(response.body.error.code).toBe('VALIDATION_ERROR');
      expect(response.body.error.message).toContain('property net_value should not exist');
    });

    // Test J: no stack trace/secret/path leakage
    it('Test J: no stack trace, credentials, tokens, or filesystem paths leaked in error responses', async () => {
      // Upstream error containing Azure connection strings and Windows path
      jest.spyOn(adapter, 'evaluate').mockRejectedValueOnce(
        new DecisionEngineErrorException(
          'Upstream crash DefaultEndpointsProtocol=https;AccountName=myacc;AccountKey=secretKey12345; at D:\\recovery-intelligence-engine\\secrets.env with token: secret-token-xyz\n at Object.<anonymous> (D:\\app\\src\\index.ts:25:10)',
        ),
      );

      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({ payment_id: 'pay_000001_a1' })
        .expect(502);

      const bodyStr = JSON.stringify(response.body);
      expect(bodyStr).not.toContain('secretKey12345');
      expect(bodyStr).not.toContain('secret-token-xyz');
      expect(bodyStr).not.toContain('D:\\recovery-intelligence-engine');
      expect(bodyStr).not.toContain('D:\\app\\src');
      expect(bodyStr).not.toContain('\n at ');
    });

    // Test K: X-Request-Id response header present
    it('Test K: X-Request-Id response header is present on every error and success response', async () => {
      // Error response
      const errRes = await request(app.getHttpServer())
        .post('/decisions')
        .send({})
        .expect(400);
      expect(errRes.headers['x-request-id']).toBeDefined();
      expect(errRes.headers['x-request-id'].length).toBeGreaterThan(0);

      // Success response
      const successRes = await request(app.getHttpServer())
        .post('/decisions')
        .send({ payment_id: 'pay_000001_a1' })
        .expect(200);
      expect(successRes.headers['x-request-id']).toBeDefined();
      expect(successRes.headers['x-request-id'].length).toBeGreaterThan(0);
    });

    // Test L: error.request_id equals response header
    it('Test L: error.request_id strictly equals the X-Request-Id response header', async () => {
      const customReqId = 'req-test-l-correlation-match';
      const response = await request(app.getHttpServer())
        .post('/decisions')
        .set('x-request-id', customReqId)
        .send({ payment_id: 'invalid_format' })
        .expect(400);

      expect(response.body.error.request_id).toBe(customReqId);
      expect(response.headers['x-request-id']).toBe(customReqId);
      expect(response.body.error.request_id).toBe(response.headers['x-request-id']);
    });

    // Test M: success request_id equals response header
    it('Test M: success response.request_id strictly equals the X-Request-Id response header', async () => {
      const customReqId = 'req-test-m-success-correlation';
      jest.spyOn(adapter, 'evaluate').mockImplementationOnce(async (pid, reqId) => ({
        ...mockPythonDecision,
        payment_id: pid,
        request_id: reqId,
      }));

      const response = await request(app.getHttpServer())
        .post('/decisions')
        .set('x-request-id', customReqId)
        .send({ payment_id: 'pay_000001_a1' })
        .expect(200);

      expect(response.body.request_id).toBe(customReqId);
      expect(response.headers['x-request-id']).toBe(customReqId);
      expect(response.body.request_id).toBe(response.headers['x-request-id']);
    });
  });
});
