import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication, ValidationPipe } from '@nestjs/common';
import * as request from 'supertest';
import { DecisionModule } from './decision.module';
import { DecisionResponseDto } from './dto/decision-response.dto';

describe('DecisionController (e2e contract validation)', () => {
  let app: INestApplication;

  beforeAll(async () => {
    const moduleRef: TestingModule = await Test.createTestingModule({
      imports: [DecisionModule],
    }).compile();

    app = moduleRef.createNestApplication();
    app.useGlobalPipes(
      new ValidationPipe({
        whitelist: true,
        forbidNonWhitelisted: true,
        transform: true,
      }),
    );
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
    expect(body.guardrail_reason === null || typeof body.guardrail_reason === 'string').toBe(true);
    expect(typeof body.final_action).toBe('string');
    expect(body.confidence === null || typeof body.confidence === 'number').toBe(true);
    expect(body.risk_level === null || typeof body.risk_level === 'string').toBe(true);
    expect(body.reasoning === null || typeof body.reasoning === 'string').toBe(true);
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

  it('should preserve and propagate x-request-id client header when provided', async () => {
    const customRequestId = 'req-custom-client-test-uuid-999';
    const response = await request(app.getHttpServer())
      .post('/decisions')
      .set('x-request-id', customRequestId)
      .send({
        payment_id: 'pay_000003_a1',
      })
      .expect(200);

    expect(response.body.request_id).toBe(customRequestId);
  });

  it('should reject payload missing payment_id with HTTP 400', async () => {
    const response = await request(app.getHttpServer())
      .post('/decisions')
      .send({})
      .expect(400);

    expect(response.body.message).toEqual(
      expect.arrayContaining([
        expect.stringContaining('payment_id should not be empty'),
      ]),
    );
  });

  it('should reject payload with invalid payment_id format with HTTP 400', async () => {
    const response = await request(app.getHttpServer())
      .post('/decisions')
      .send({
        payment_id: 'invalid_payment_format',
      })
      .expect(400);

    expect(response.body.message).toEqual(
      expect.arrayContaining([
        expect.stringContaining('payment_id must match the format pay_XXXXXX_aY'),
      ]),
    );
  });

  it('should reject payload with undeclared net_value with HTTP 400 [tests FIX-6 end-to-end]', async () => {
    const response = await request(app.getHttpServer())
      .post('/decisions')
      .send({
        payment_id: 'pay_000001_a1',
        net_value: 100,
      })
      .expect(400);

    expect(response.body.message).toEqual(
      expect.arrayContaining(['property net_value should not exist']),
    );
  });

  it('should reject client-injected probabilities, model decisions, or guardrails with HTTP 400', async () => {
    const response = await request(app.getHttpServer())
      .post('/decisions')
      .send({
        payment_id: 'pay_000001_a1',
        probabilities: { WAIT: 0.9 },
        tau: 0.88,
        model_action: 'RETRY_NUDGE',
        llm_decision: 'WAIT',
        guardrail_decision: 'STOP',
      })
      .expect(400);

    expect(response.body.message).toEqual(
      expect.arrayContaining([
        'property probabilities should not exist',
        'property tau should not exist',
        'property model_action should not exist',
        'property llm_decision should not exist',
        'property guardrail_decision should not exist',
      ]),
    );
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
