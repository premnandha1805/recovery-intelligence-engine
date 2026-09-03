import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication } from '@nestjs/common';
import * as request from 'supertest';
import { getCorsConfig } from './cors.config';
import { HealthController } from '../health/health.controller';
import { HealthModule } from '../health/health.module';
import { ConfigModule } from '@nestjs/config';

describe('CORS Configuration', () => {
  describe('getCorsConfig Unit Tests', () => {
    it('should include localhost development origins by default', () => {
      const config = getCorsConfig();
      const origins = config.origin as string[];

      expect(origins).toContain('http://localhost:5173');
      expect(origins).toContain('http://127.0.0.1:5173');
      expect(origins).toContain('http://localhost:3000');
      expect(origins).not.toContain('*');
    });

    it('should append production frontend origin when provided', () => {
      const prodOrigin = 'https://recovery-frontend.onrender.com';
      const config = getCorsConfig(prodOrigin);
      const origins = config.origin as string[];

      expect(origins).toContain(prodOrigin);
      expect(origins).toContain('http://localhost:5173');
    });

    it('should strip trailing slashes from production frontend origin', () => {
      const prodOrigin = 'https://recovery-frontend.onrender.com/';
      const config = getCorsConfig(prodOrigin);
      const origins = config.origin as string[];

      expect(origins).toContain('https://recovery-frontend.onrender.com');
      expect(origins).not.toContain('https://recovery-frontend.onrender.com/');
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
    const testProdOrigin = 'https://recovery-frontend.onrender.com';

    beforeAll(async () => {
      const moduleFixture: TestingModule = await Test.createTestingModule({
        imports: [
          ConfigModule.forRoot({ isGlobal: true }),
          HealthModule,
        ],
      }).compile();

      app = moduleFixture.createNestApplication();
      app.enableCors(getCorsConfig(testProdOrigin));
      await app.init();
    });

    afterAll(async () => {
      await app.close();
    });

    it('should permit OPTIONS preflight from configured production frontend origin', async () => {
      const res = await request(app.getHttpServer())
        .options('/health')
        .set('Origin', testProdOrigin)
        .set('Access-Control-Request-Method', 'GET')
        .set('Access-Control-Request-Headers', 'Content-Type, X-Request-Id');

      expect(res.status).toBe(204);
      expect(res.headers['access-control-allow-origin']).toBe(testProdOrigin);
      expect(res.headers['access-control-allow-methods']).toContain('GET');
      expect(res.headers['access-control-allow-methods']).toContain('POST');
      expect(res.headers['access-control-allow-methods']).toContain('OPTIONS');
    });

    it('should permit OPTIONS preflight from localhost development origin (5173)', async () => {
      const res = await request(app.getHttpServer())
        .options('/health')
        .set('Origin', 'http://localhost:5173')
        .set('Access-Control-Request-Method', 'POST')
        .set('Access-Control-Request-Headers', 'Content-Type');

      expect(res.status).toBe(204);
      expect(res.headers['access-control-allow-origin']).toBe('http://localhost:5173');
    });

    it('should permit OPTIONS preflight from 127.0.0.1 development origin (5173)', async () => {
      const res = await request(app.getHttpServer())
        .options('/health')
        .set('Origin', 'http://127.0.0.1:5173')
        .set('Access-Control-Request-Method', 'POST')
        .set('Access-Control-Request-Headers', 'Content-Type');

      expect(res.status).toBe(204);
      expect(res.headers['access-control-allow-origin']).toBe('http://127.0.0.1:5173');
    });

    it('should NOT permit CORS for an unrelated origin', async () => {
      const unrelatedOrigin = 'https://malicious-third-party.com';
      const res = await request(app.getHttpServer())
        .options('/health')
        .set('Origin', unrelatedOrigin)
        .set('Access-Control-Request-Method', 'POST');

      // Express cors middleware omits the access-control-allow-origin header for disallowed origins
      expect(res.headers['access-control-allow-origin']).toBeUndefined();
    });

    it('should attach Access-Control-Allow-Origin on standard GET request from allowed origin', async () => {
      const res = await request(app.getHttpServer())
        .get('/health')
        .set('Origin', testProdOrigin);

      expect(res.headers['access-control-allow-origin']).toBe(testProdOrigin);
    });

    it('should NOT attach Access-Control-Allow-Origin on standard GET request from unrelated origin', async () => {
      const unrelatedOrigin = 'https://malicious-third-party.com';
      const res = await request(app.getHttpServer())
        .get('/health')
        .set('Origin', unrelatedOrigin);

      expect(res.headers['access-control-allow-origin']).toBeUndefined();
    });
  });
});
