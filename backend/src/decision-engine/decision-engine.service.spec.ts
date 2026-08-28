import { Test, TestingModule } from '@nestjs/testing';
import { ConfigService } from '@nestjs/config';
import * as http from 'http';
import { EventEmitter } from 'events';
import { DecisionEngineService } from './decision-engine.service';

describe('DecisionEngineService', () => {
  let service: DecisionEngineService;
  let configService: ConfigService;

  const mockConfigValues: Record<string, any> = {
    DECISION_ENGINE_URL: 'http://localhost:8000',
    DECISION_ENGINE_TIMEOUT_MS: 8000,
    HEALTH_CHECK_TIMEOUT_MS: 1250, // Non-magic custom number to prove config adherence
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        DecisionEngineService,
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

    service = module.get<DecisionEngineService>(DecisionEngineService);
    configService = module.get<ConfigService>(ConfigService);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  it('should read config values for URL and timeouts', () => {
    expect(service.getDecisionEngineUrl()).toBe('http://localhost:8000');
    expect(service.getTimeoutMs()).toBe(8000);
    expect(service.getHealthCheckTimeoutMs()).toBe(1250);
  });

  it('should maintain a shared pooled HTTP agent with keepAlive', () => {
    const agent = service.getHttpAgent();
    expect(agent).toBeInstanceOf(http.Agent);
    const agentAny = agent as any;
    expect(agentAny.options?.keepAlive ?? agentAny.keepAlive).toBe(true);
    expect(agentAny.options?.maxSockets ?? agentAny.maxSockets).toBe(50);
    expect(agentAny.options?.maxFreeSockets ?? agentAny.maxFreeSockets).toBe(10);
  });

  it('should clean up shared agents onModuleDestroy [FIX-8]', () => {
    const httpAgent = service.getHttpAgent();
    const httpsAgent = service.getHttpsAgent();
    const httpSpy = jest.spyOn(httpAgent, 'destroy');
    const httpsSpy = jest.spyOn(httpsAgent, 'destroy');

    service.onModuleDestroy();

    expect(httpSpy).toHaveBeenCalled();
    expect(httpsSpy).toHaveBeenCalled();
  });

  describe('checkHealth()', () => {
    it('should assert the request timeout matches HEALTH_CHECK_TIMEOUT_MS from config, not a magic number', async () => {
      let capturedTimeout: number | undefined;

      jest.spyOn(http, 'request').mockImplementation(((url: any, options: any, cb: any) => {
        capturedTimeout = options.timeout;
        const req = new EventEmitter() as any;
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

      const status = await service.checkHealth();

      expect(status).toBe('ok');
      // Proves timeout is strictly 1250ms from configured mockConfigValues, not 1000 or a hardcoded literal
      expect(capturedTimeout).toBe(1250);
      expect(capturedTimeout).toBe(service.getHealthCheckTimeoutMs());
    });

    it('should resolve "ok" when Python decision engine responds 200 {"status": "ok"}', async () => {
      jest.spyOn(http, 'request').mockImplementation(((url: any, options: any, cb: any) => {
        const req = new EventEmitter() as any;
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

      const result = await service.checkHealth();
      expect(result).toBe('ok');
    });

    it('should resolve "unreachable" when Python decision engine is down (connection error)', async () => {
      jest.spyOn(http, 'request').mockImplementation(((url: any, options: any, cb: any) => {
        const req = new EventEmitter() as any;
        req.end = jest.fn();
        req.destroy = jest.fn();

        process.nextTick(() => {
          const error: any = new Error('connect ECONNREFUSED 127.0.0.1:8000');
          error.code = 'ECONNREFUSED';
          req.emit('error', error);
        });

        return req;
      }) as any);

      const result = await service.checkHealth();
      expect(result).toBe('unreachable');
    });

    it('should resolve "unreachable" when Python decision engine returns HTTP 503 uninitialized', async () => {
      jest.spyOn(http, 'request').mockImplementation(((url: any, options: any, cb: any) => {
        const req = new EventEmitter() as any;
        req.end = jest.fn();
        req.destroy = jest.fn();

        process.nextTick(() => {
          const res = new EventEmitter() as any;
          res.statusCode = 503;
          res.setEncoding = jest.fn();
          cb(res);
          res.emit('data', JSON.stringify({ error: { code: 'INTERNAL_ERROR' } }));
          res.emit('end');
        });

        return req;
      }) as any);

      const result = await service.checkHealth();
      expect(result).toBe('unreachable');
    });

    it('should resolve "timeout" when request exceeds health check timeout', async () => {
      jest.spyOn(http, 'request').mockImplementation(((url: any, options: any, cb: any) => {
        const req = new EventEmitter() as any;
        req.end = jest.fn();
        req.destroy = jest.fn(() => {
          process.nextTick(() => {
            req.emit('error', new Error('HEALTH_CHECK_TIMEOUT'));
          });
        });

        // Simulate timeout event firing before any response
        process.nextTick(() => {
          req.emit('timeout');
        });

        return req;
      }) as any);

      const result = await service.checkHealth();
      expect(result).toBe('timeout');
    });
  });
});
