import { Test, TestingModule } from '@nestjs/testing';
import { HttpStatus } from '@nestjs/common';
import { HealthController } from './health.controller';
import { DecisionEngineService } from '../decision-engine/decision-engine.service';

describe('HealthController', () => {
  let controller: HealthController;
  let decisionEngineService: DecisionEngineService;

  const mockResponse = () => {
    const res: any = {};
    res.status = jest.fn().mockReturnValue(res);
    return res;
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [HealthController],
      providers: [
        {
          provide: DecisionEngineService,
          useValue: {
            checkHealth: jest.fn(),
          },
        },
      ],
    }).compile();

    controller = module.get<HealthController>(HealthController);
    decisionEngineService = module.get<DecisionEngineService>(DecisionEngineService);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  describe('GET /health composite checks', () => {
    it('should return 200 and status ok when both NestJS and Python engine are healthy', async () => {
      jest.spyOn(decisionEngineService, 'checkHealth').mockResolvedValue('ok');
      const res = mockResponse();

      const result = await controller.check(res);

      expect(res.status).not.toHaveBeenCalled();
      expect(result).toEqual({
        status: 'ok',
        service: 'recovery-intelligence-api',
        dependencies: {
          decision_engine: 'ok',
        },
      });
    });

    it('should return 503 and status degraded when Python engine is unavailable (unreachable)', async () => {
      jest.spyOn(decisionEngineService, 'checkHealth').mockResolvedValue('unreachable');
      const res = mockResponse();

      const result = await controller.check(res);

      expect(res.status).toHaveBeenCalledWith(HttpStatus.SERVICE_UNAVAILABLE);
      expect(result).toEqual({
        status: 'degraded',
        service: 'recovery-intelligence-api',
        dependencies: {
          decision_engine: 'unreachable',
        },
      });
      // Assert no failed Python decision engine hidden behind HTTP 200 or "ok"
      expect(result.status).not.toBe('ok');
    });

    it('should return 503 and status degraded when Python engine request times out', async () => {
      jest.spyOn(decisionEngineService, 'checkHealth').mockResolvedValue('timeout');
      const res = mockResponse();

      const result = await controller.check(res);

      expect(res.status).toHaveBeenCalledWith(HttpStatus.SERVICE_UNAVAILABLE);
      expect(result).toEqual({
        status: 'degraded',
        service: 'recovery-intelligence-api',
        dependencies: {
          decision_engine: 'timeout',
        },
      });
      expect(result.status).not.toBe('ok');
    });

    it('should contain no stack traces, passwords, or credentials in response', async () => {
      jest.spyOn(decisionEngineService, 'checkHealth').mockResolvedValue('unreachable');
      const res = mockResponse();

      const result = await controller.check(res);
      const jsonStr = JSON.stringify(result);

      expect(jsonStr).not.toContain('stack');
      expect(jsonStr).not.toContain('trace');
      expect(jsonStr).not.toContain('password');
      expect(jsonStr).not.toContain('secret');
      expect(jsonStr).not.toContain('key');
      expect(Object.keys(result)).toEqual(['status', 'service', 'dependencies']);
      expect(Object.keys(result.dependencies)).toEqual(['decision_engine']);
    });
  });
});
