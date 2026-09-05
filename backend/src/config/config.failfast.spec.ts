import { ConfigModule } from '@nestjs/config';
import { configValidationSchema } from './config.schema';

describe('ConfigModule Boot Fail-Fast [FIX-7]', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  it('should fail fast at boot when PORT is malformed', async () => {
    process.env.PORT = 'not-a-number';

    await expect(
      ConfigModule.forRoot({
        validationSchema: configValidationSchema,
        validationOptions: { abortEarly: false },
      }),
    ).rejects.toThrow(/Config validation error.*PORT/i);
  });

  it('should fail fast at boot when DECISION_ENGINE_URL is malformed', async () => {
    process.env.DECISION_ENGINE_URL = 'malformed-url';

    await expect(
      ConfigModule.forRoot({
        validationSchema: configValidationSchema,
        validationOptions: { abortEarly: false },
      }),
    ).rejects.toThrow(/Config validation error.*DECISION_ENGINE_URL/i);
  });

  it('should fail fast at boot when DECISION_ENGINE_TIMEOUT_MS is negative', async () => {
    process.env.DECISION_ENGINE_TIMEOUT_MS = '-50';

    await expect(
      ConfigModule.forRoot({
        validationSchema: configValidationSchema,
        validationOptions: { abortEarly: false },
      }),
    ).rejects.toThrow(/Config validation error.*DECISION_ENGINE_TIMEOUT_MS/i);
  });

  it('should fail fast at boot when HEALTH_CHECK_TIMEOUT_MS is negative', async () => {
    process.env.HEALTH_CHECK_TIMEOUT_MS = '-10';

    await expect(
      ConfigModule.forRoot({
        validationSchema: configValidationSchema,
        validationOptions: { abortEarly: false },
      }),
    ).rejects.toThrow(/Config validation error.*HEALTH_CHECK_TIMEOUT_MS/i);
  });

  it('should boot successfully when env vars are valid or omitted (defaults)', async () => {
    delete process.env.PORT;
    delete process.env.DECISION_ENGINE_URL;
    delete process.env.DECISION_ENGINE_TIMEOUT_MS;
    delete process.env.HEALTH_CHECK_TIMEOUT_MS;

    await expect(
      ConfigModule.forRoot({
        validationSchema: configValidationSchema,
        validationOptions: { abortEarly: false },
      }),
    ).resolves.toBeDefined();
  });
});
