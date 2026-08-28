import { Test, TestingModule } from '@nestjs/testing';
import { ConfigService } from '@nestjs/config';
import { DecisionEngineService } from './decision-engine.service';

describe('DecisionEngineService', () => {
  let service: DecisionEngineService;
  let configService: ConfigService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        DecisionEngineService,
        {
          provide: ConfigService,
          useValue: {
            get: jest.fn((key: string, defaultValue?: any) => {
              if (key === 'DECISION_ENGINE_URL') return 'http://localhost:8000';
              if (key === 'DECISION_ENGINE_TIMEOUT_MS') return 8000;
              return defaultValue;
            }),
          },
        },
      ],
    }).compile();

    service = module.get<DecisionEngineService>(DecisionEngineService);
    configService = module.get<ConfigService>(ConfigService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  it('should read config values for URL and timeout', () => {
    expect(service.getDecisionEngineUrl()).toBe('http://localhost:8000');
    expect(service.getTimeoutMs()).toBe(8000);
  });

  it('should support onModuleDestroy hook cleanly [FIX-8]', () => {
    expect(() => service.onModuleDestroy()).not.toThrow();
  });
});
