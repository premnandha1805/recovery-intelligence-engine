import { configValidationSchema } from './config.schema';

describe('ConfigValidationSchema', () => {
  it('should apply defaults when env vars are omitted', () => {
    const { error, value } = configValidationSchema.validate({}, { abortEarly: false });
    expect(error).toBeUndefined();
    expect(value.PORT).toBe(3000);
    expect(value.DECISION_ENGINE_URL).toBe('http://localhost:8000');
    expect(value.DECISION_ENGINE_TIMEOUT_MS).toBe(8000);
    expect(value.HEALTH_CHECK_TIMEOUT_MS).toBe(1000);
  });

  it('should accept valid custom values', () => {
    const input = {
      PORT: 4000,
      DECISION_ENGINE_URL: 'https://decision-engine.internal:8000',
      DECISION_ENGINE_TIMEOUT_MS: 5000,
      HEALTH_CHECK_TIMEOUT_MS: 2000,
    };
    const { error, value } = configValidationSchema.validate(input, { abortEarly: false });
    expect(error).toBeUndefined();
    expect(value.PORT).toBe(4000);
    expect(value.DECISION_ENGINE_URL).toBe('https://decision-engine.internal:8000');
    expect(value.DECISION_ENGINE_TIMEOUT_MS).toBe(5000);
    expect(value.HEALTH_CHECK_TIMEOUT_MS).toBe(2000);
  });

  it('should fail fast on invalid PORT', () => {
    const input = { PORT: 'not-a-port' };
    const { error } = configValidationSchema.validate(input, { abortEarly: false });
    expect(error).toBeDefined();
    expect(error?.message).toContain('"PORT" must be a number');
  });

  it('should fail fast on out-of-range PORT', () => {
    const input = { PORT: 70000 };
    const { error } = configValidationSchema.validate(input, { abortEarly: false });
    expect(error).toBeDefined();
    expect(error?.message).toContain('"PORT" must be a valid port');
  });

  it('should fail fast on malformed DECISION_ENGINE_URL', () => {
    const input = { DECISION_ENGINE_URL: 'not-a-valid-url' };
    const { error } = configValidationSchema.validate(input, { abortEarly: false });
    expect(error).toBeDefined();
    expect(error?.message).toContain('"DECISION_ENGINE_URL" must be a valid uri');
  });

  it('should fail fast on negative DECISION_ENGINE_TIMEOUT_MS', () => {
    const input = { DECISION_ENGINE_TIMEOUT_MS: -100 };
    const { error } = configValidationSchema.validate(input, { abortEarly: false });
    expect(error).toBeDefined();
    expect(error?.message).toContain('"DECISION_ENGINE_TIMEOUT_MS" must be a positive number');
  });

  it('should fail fast on non-integer HEALTH_CHECK_TIMEOUT_MS', () => {
    const input = { HEALTH_CHECK_TIMEOUT_MS: 'fast' };
    const { error } = configValidationSchema.validate(input, { abortEarly: false });
    expect(error).toBeDefined();
    expect(error?.message).toContain('"HEALTH_CHECK_TIMEOUT_MS" must be a number');
  });

  it('should accept valid FRONTEND_ORIGIN uri', () => {
    const input = { FRONTEND_ORIGIN: 'https://recovery-frontend.onrender.com' };
    const { error, value } = configValidationSchema.validate(input, { abortEarly: false });
    expect(error).toBeUndefined();
    expect(value.FRONTEND_ORIGIN).toBe('https://recovery-frontend.onrender.com');
  });

  it('should fail fast on malformed FRONTEND_ORIGIN', () => {
    const input = { FRONTEND_ORIGIN: 'not-a-valid-uri' };
    const { error } = configValidationSchema.validate(input, { abortEarly: false });
    expect(error).toBeDefined();
    expect(error?.message).toContain('"FRONTEND_ORIGIN" must be a valid uri');
  });
});
