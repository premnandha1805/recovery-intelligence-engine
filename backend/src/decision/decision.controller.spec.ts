import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication, ValidationPipe } from '@nestjs/common';
import * as request from 'supertest';
import { DecisionModule } from './decision.module';

describe('DecisionController (e2e / validation)', () => {
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

  it('should accept valid payload with required payment_id only', async () => {
    const response = await request(app.getHttpServer())
      .post('/decisions')
      .send({
        payment_id: 'pay_000001_a1',
      })
      .expect(200);

    expect(response.body).toEqual({
      message: 'Decision request accepted (skeleton)',
      payment_id: 'pay_000001_a1',
      force_recompute: false,
    });
  });

  it('should accept valid payload with optional force_recompute', async () => {
    const response = await request(app.getHttpServer())
      .post('/decisions')
      .send({
        payment_id: 'pay_000001_a1',
        force_recompute: true,
      })
      .expect(200);

    expect(response.body).toEqual({
      message: 'Decision request accepted (skeleton)',
      payment_id: 'pay_000001_a1',
      force_recompute: true,
    });
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

  it('should reject payload with undeclared net_value with HTTP 400 [FIX-6]', async () => {
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
        guardrail_decision: 'STOP',
      })
      .expect(400);

    expect(response.body.message).toEqual(
      expect.arrayContaining([
        'property probabilities should not exist',
        'property guardrail_decision should not exist',
      ]),
    );
  });
});
