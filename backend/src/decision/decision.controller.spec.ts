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

describe('DecisionController (e2e contract & error envelope validation)', () => {
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

  it('should preserve and propagate x-request-id client header when provided [FIX-10]', async () => {
    const customRequestId = 'req-custom-client-test-uuid-999';
    jest.spyOn(adapter, 'evaluate').mockImplementationOnce(async (pid, reqId) => ({
      ...mockPythonDecision,
      payment_id: pid,
      request_id: reqId,
    }));

    const response = await request(app.getHttpServer())
      .post('/decisions')
      .set('x-request-id', customRequestId)
      .send({
        payment_id: 'pay_000003_a1',
      })
      .expect(200);

    expect(response.body.request_id).toBe(customRequestId);
  });

  describe('Error Envelope & Status Mappings', () => {
    it('should map invalid request payload to HTTP 400 with VALIDATION_ERROR code and request_id', async () => {
      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({})
        .expect(400);

      expect(response.body.error).toBeDefined();
      expect(response.body.error.code).toBe('VALIDATION_ERROR');
      expect(response.body.error.message).toContain('payment_id should not be empty');
      expect(response.body.error.request_id).toBeDefined();
      expect(typeof response.body.error.request_id).toBe('string');
      expect(response.body.error.request_id.length).toBeGreaterThan(10);
      expect(response.body.message).toEqual(
        expect.arrayContaining([
          expect.stringContaining('payment_id should not be empty'),
        ]),
      );
    });

    it('should preserve caller X-Request-Id on validation error responses [FIX-10]', async () => {
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

    it('should reject invalid payment_id format with 400 VALIDATION_ERROR and request_id', async () => {
      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({
          payment_id: 'invalid_payment_format',
        })
        .expect(400);

      expect(response.body.error.code).toBe('VALIDATION_ERROR');
      expect(response.body.error.message).toContain(
        'payment_id must match the format pay_XXXXXX_aY',
      );
      expect(response.body.error.request_id).toBeDefined();
    });

    it('should reject extra field "net_value" with 400 VALIDATION_ERROR and request_id [FIX-6 end-to-end]', async () => {
      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({
          payment_id: 'pay_000001_a1',
          net_value: 100,
        })
        .expect(400);

      expect(response.body.error.code).toBe('VALIDATION_ERROR');
      expect(response.body.error.message).toContain(
        'property net_value should not exist',
      );
      expect(response.body.error.request_id).toBeDefined();
      expect(response.body.message).toEqual(
        expect.arrayContaining(['property net_value should not exist']),
      );
    });

    it('should map Python unavailable to HTTP 503 with DECISION_ENGINE_UNAVAILABLE and request_id', async () => {
      const customReqId = 'req-unavail-test-503';
      jest
        .spyOn(adapter, 'evaluate')
        .mockRejectedValueOnce(
          new DecisionEngineUnavailableException('Connection refused 127.0.0.1:8000'),
        );

      const response = await request(app.getHttpServer())
        .post('/decisions')
        .set('x-request-id', customReqId)
        .send({
          payment_id: 'pay_000001_a1',
        })
        .expect(503);

      expect(response.body.error).toEqual({
        code: 'DECISION_ENGINE_UNAVAILABLE',
        message: 'Connection refused 127.0.0.1:8000',
        request_id: customReqId,
      });
      expect(response.headers['x-request-id']).toBe(customReqId);
    });

    it('should map Python timeout to HTTP 503 with DECISION_ENGINE_TIMEOUT and generated request_id', async () => {
      jest
        .spyOn(adapter, 'evaluate')
        .mockRejectedValueOnce(new DecisionEngineTimeoutException());

      const response = await request(app.getHttpServer())
        .post('/decisions')
        .send({
          payment_id: 'pay_000001_a1',
        })
        .expect(503);

      expect(response.body.error).toEqual({
        code: 'DECISION_ENGINE_TIMEOUT',
        message: 'Python decision engine request timed out',
        request_id: expect.any(String),
      });
      expect(response.body.error.request_id.length).toBeGreaterThan(10);
    });

    it('should map upstream real 5xx to HTTP 502 with DECISION_ENGINE_ERROR and request_id', async () => {
      const customReqId = 'req-upstream-502-trace';
      jest
        .spyOn(adapter, 'evaluate')
        .mockRejectedValueOnce(
          new DecisionEngineErrorException('Upstream HTTP 500 internal crash'),
        );

      const response = await request(app.getHttpServer())
        .post('/decisions')
        .set('x-request-id', customReqId)
        .send({
          payment_id: 'pay_000001_a1',
        })
        .expect(502);

      expect(response.body.error).toEqual({
        code: 'DECISION_ENGINE_ERROR',
        message: 'Upstream HTTP 500 internal crash',
        request_id: customReqId,
      });
      expect(response.headers['x-request-id']).toBe(customReqId);
    });

    it('should never leak credentials, secrets, or stack traces in error or success responses', async () => {
      const successRes = await request(app.getHttpServer())
        .post('/decisions')
        .send({ payment_id: 'pay_000001_a1' })
        .expect(200);

      const successStr = JSON.stringify(successRes.body);
      expect(successStr).not.toContain('password');
      expect(successStr).not.toContain('secret');
      expect(successStr).not.toContain('token');
      expect(successStr).not.toContain('stack');
      expect(successStr).not.toContain('Trace');

      const errRes = await request(app.getHttpServer())
        .post('/decisions')
        .send({ payment_id: 'pay_000001_a1', net_value: 100 })
        .expect(400);

      const errStr = JSON.stringify(errRes.body);
      expect(errStr).not.toContain('password');
      expect(errStr).not.toContain('secret');
      expect(errStr).not.toContain('token');
      expect(errStr).not.toContain('stack');
    });
  });
});
