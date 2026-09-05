import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication, ValidationPipe } from '@nestjs/common';
import * as request from 'supertest';
import { getCorsConfig } from './cors.config';
import { HealthModule } from '../health/health.module';
import { DecisionModule } from '../decision/decision.module';
import { ConfigModule } from '@nestjs/config';
import { HttpExceptionFilter } from './filters/http-exception.filter';
import {
  DecisionEngineAdapter,
  PythonDecisionResult,
} from '../decision-engine/decision-engine.adapter';
import { DecisionEngineService } from '../decision-engine/decision-engine.service';

describe('CORS Configuration', () => {
  describe('getCorsConfig Unit Tests', () => {
    it('should include localhost development origins and production origin by default', () => {
      const config = getCorsConfig();
      const origins = config.origin as string[];

      expect(origins).toContain('http://localhost:5173');
      expect(origins).toContain('http://127.0.0.1:5173');
      expect(origins).toContain('http://localhost:3000');
      expect(origins).toContain('https://recovery-intelligence-engine-1.onrender.com');
      expect(origins).not.toContain('*');
    });

    it('should append custom production frontend origin when provided', () => {
      const customOrigin = 'https://custom-frontend.onrender.com';
      const config = getCorsConfig(customOrigin);
      const origins = config.origin as string[];

      expect(origins).toContain(customOrigin);
      expect(origins).toContain('https://recovery-intelligence-engine-1.onrender.com');
      expect(origins).toContain('http://localhost:5173');
    });

    it('should strip trailing slashes from production frontend origin', () => {
      const prodOrigin = 'https://custom-frontend.onrender.com/';
      const config = getCorsConfig(prodOrigin);
      const origins = config.origin as string[];

      expect(origins).toContain('https://custom-frontend.onrender.com');
      expect(origins).not.toContain('https://custom-frontend.onrender.com/');
    });

    it('should allow only GET, POST, OPTIONS methods', () => {
      const config = getCorsConfig();
      expect(config.methods).toEqual(['GET', 'POST', 'OPTIONS']);
    });

    it('should allow Content-Type and X-Request-Id headers', () => {
      const config = getCorsConfig();
      expect(config.allowedHeaders).toEqual(['Content-Type', 'X-Request-Id']);
    });
  });

  describe('CORS Integration Tests via HTTP', () => {
    let app: INestApplication;
    const prodOrigin = 'https://recovery-intelligence-engine-1.onrender.com';

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
      request_id: 'req-cors-test-123',
    };

    beforeAll(async () => {
      const moduleFixture: TestingModule = await Test.createTestingModule({
        imports: [
          ConfigModule.forRoot({ isGlobal: true }),
          HealthModule,
          DecisionModule,
        ],
      })
        .overrideProvider(DecisionEngineAdapter)
        .useValue({
          evaluate: jest.fn().mockResolvedValue(mockPythonDecision),
          getTimeoutMs: jest.fn().mockReturnValue(8000),
        })
        .overrideProvider(DecisionEngineService)
        .useValue({
          checkHealth: jest.fn().mockResolvedValue('ok'),
          getHealthCheckTimeoutMs: jest.fn().mockReturnValue(1000),
          getTimeoutMs: jest.fn().mockReturnValue(8000),
        })
        .compile();

      app = moduleFixture.createNestApplication();
      app.enableCors(getCorsConfig(prodOrigin));
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

    it('should permit OPTIONS preflight on /decisions from production origin with 204 and correct CORS headers', async () => {
      const res = await request(app.getHttpServer())
        .options('/decisions')
        .set('Origin', prodOrigin)
        .set('Access-Control-Request-Method', 'POST')
        .set('Access-Control-Request-Headers', 'Content-Type, X-Request-Id');

      expect(res.status).toBe(204);
      expect(res.headers['access-control-allow-origin']).toBe(prodOrigin);
      expect(res.headers['access-control-allow-methods']).toContain('POST');
      expect(res.headers['access-control-allow-methods']).toContain('OPTIONS');
      expect(res.headers['access-control-allow-headers']).toContain('Content-Type');
      expect(res.headers['access-control-allow-headers']).toContain('X-Request-Id');
      expect(res.headers['access-control-allow-credentials']).toBe('true');
    });

    it('should permit OPTIONS preflight on /decisions from localhost development origin (5173)', async () => {
      const res = await request(app.getHttpServer())
        .options('/decisions')
        .set('Origin', 'http://localhost:5173')
        .set('Access-Control-Request-Method', 'POST')
        .set('Access-Control-Request-Headers', 'Content-Type, X-Request-Id');

      expect(res.status).toBe(204);
      expect(res.headers['access-control-allow-origin']).toBe('http://localhost:5173');
    });

    it('should permit OPTIONS preflight on /decisions from 127.0.0.1 development origin (5173)', async () => {
      const res = await request(app.getHttpServer())
        .options('/decisions')
        .set('Origin', 'http://127.0.0.1:5173')
        .set('Access-Control-Request-Method', 'POST')
        .set('Access-Control-Request-Headers', 'Content-Type, X-Request-Id');

      expect(res.status).toBe(204);
      expect(res.headers['access-control-allow-origin']).toBe('http://127.0.0.1:5173');
    });

    it('should NOT attach Access-Control-Allow-Origin for an unrelated origin on preflight', async () => {
      const unrelatedOrigin = 'https://malicious-third-party.com';
      const res = await request(app.getHttpServer())
        .options('/decisions')
        .set('Origin', unrelatedOrigin)
        .set('Access-Control-Request-Method', 'POST')
        .set('Access-Control-Request-Headers', 'Content-Type, X-Request-Id');

      expect(res.headers['access-control-allow-origin']).toBeUndefined();
    });

    it('should attach Access-Control-Allow-Origin on normal GET /health from allowed production origin', async () => {
      const res = await request(app.getHttpServer())
        .get('/health')
        .set('Origin', prodOrigin);

      expect(res.status).toBe(200);
      expect(res.headers['access-control-allow-origin']).toBe(prodOrigin);
      expect(res.body.service).toBe('recovery-intelligence-api');
    });

    it('should attach Access-Control-Allow-Origin on normal POST /decisions from allowed production origin', async () => {
      const res = await request(app.getHttpServer())
        .post('/decisions')
        .set('Origin', prodOrigin)
        .send({
          payment_id: 'pay_000001_a1',
          features: {
            amount: 2500,
            attempt_number: 1,
            dynamic_success_rate: 0.7,
            cumulative_failures: 0,
            consecutive_failed_cycles: 0,
            notification_engagement_score: 0.85,
            contact_response_score: 0.6,
            payment_method: 'upi',
            failure_reason: 'temporary_bank_issue',
          },
        });

      expect(res.status).toBe(200);
      expect(res.headers['access-control-allow-origin']).toBe(prodOrigin);
      expect(res.body.payment_id).toBe('pay_000001_a1');
      expect(res.body.final_action).toBe('RETRY_NUDGE');
    });

    it('should NOT attach Access-Control-Allow-Origin on GET /health from unrelated origin', async () => {
      const unrelatedOrigin = 'https://malicious-third-party.com';
      const res = await request(app.getHttpServer())
        .get('/health')
        .set('Origin', unrelatedOrigin);

      expect(res.headers['access-control-allow-origin']).toBeUndefined();
    });

    it('should NOT attach Access-Control-Allow-Origin on POST /decisions from unrelated origin', async () => {
      const unrelatedOrigin = 'https://malicious-third-party.com';
      const res = await request(app.getHttpServer())
        .post('/decisions')
        .set('Origin', unrelatedOrigin)
        .send({
          payment_id: 'pay_000001_a1',
          features: {
            amount: 2500,
            attempt_number: 1,
            dynamic_success_rate: 0.7,
            cumulative_failures: 0,
            consecutive_failed_cycles: 0,
            notification_engagement_score: 0.85,
            contact_response_score: 0.6,
            payment_method: 'upi',
            failure_reason: 'temporary_bank_issue',
          },
        });

      expect(res.status).toBe(200);
      expect(res.headers['access-control-allow-origin']).toBeUndefined();
    });
  });
});
