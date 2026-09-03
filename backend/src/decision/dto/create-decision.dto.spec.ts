import { validate } from 'class-validator';
import { plainToInstance } from 'class-transformer';
import { CreateDecisionDto } from './create-decision.dto';

describe('CreateDecisionDto Validation', () => {
  const transformAndValidate = async (plain: Record<string, any>) => {
    const instance = plainToInstance(CreateDecisionDto, plain);
    const errors = await validate(instance, {
      whitelist: true,
      forbidNonWhitelisted: true,
    });
    return { instance, errors };
  };

  it('should pass validation with valid payment_id format (pay_XXXXXX_aY)', async () => {
    const { errors, instance } = await transformAndValidate({
      payment_id: 'pay_000001_a1',
    });

    expect(errors.length).toBe(0);
    expect(instance.payment_id).toBe('pay_000001_a1');
    expect(instance.force_recompute).toBe(false);
  });

  it('should accept valid optional force_recompute boolean', async () => {
    const { errors, instance } = await transformAndValidate({
      payment_id: 'pay_000042_a2',
      force_recompute: true,
    });

    expect(errors.length).toBe(0);
    expect(instance.force_recompute).toBe(true);
  });

  it('should reject missing payment_id', async () => {
    const { errors } = await transformAndValidate({});

    expect(errors.length).toBeGreaterThan(0);
    const paymentError = errors.find((e) => e.property === 'payment_id');
    expect(paymentError).toBeDefined();
    expect(paymentError?.constraints).toHaveProperty('isNotEmpty');
  });

  it('should reject empty payment_id string', async () => {
    const { errors } = await transformAndValidate({
      payment_id: '',
    });

    expect(errors.length).toBeGreaterThan(0);
    const paymentError = errors.find((e) => e.property === 'payment_id');
    expect(paymentError).toBeDefined();
    expect(paymentError?.constraints).toHaveProperty('isNotEmpty');
  });

  it.each([
    'invalid_payment_id',
    'pay_123',
    'pay_abcdef_a1',
    'PAY_000001_A1',
    'pay_00001_a1', // 5 digits instead of 6
    'pay_0000001_a1', // 7 digits
    'pay_000001', // missing attempt suffix
    '000001_a1', // missing pay_ prefix
  ])('should reject invalid payment_id format: %s', async (invalidId) => {
    const { errors } = await transformAndValidate({
      payment_id: invalidId,
    });

    expect(errors.length).toBeGreaterThan(0);
    const paymentError = errors.find((e) => e.property === 'payment_id');
    expect(paymentError).toBeDefined();
    expect(paymentError?.constraints).toHaveProperty('matches');
    expect(paymentError?.constraints?.matches).toContain('pay_XXXXXX_aY');
  });

  it('should reject extra field "net_value" with forbidNonWhitelisted [FIX-6]', async () => {
    const { errors } = await transformAndValidate({
      payment_id: 'pay_000001_a1',
      net_value: 100,
    });

    expect(errors.length).toBeGreaterThan(0);
    const extraError = errors.find((e) => e.property === 'net_value');
    expect(extraError).toBeDefined();
    expect(extraError?.constraints).toHaveProperty('whitelistValidation');
    expect(extraError?.constraints?.whitelistValidation).toContain(
      'property net_value should not exist',
    );
  });

  it.each([
    ['probabilities', { WAIT: 0.8, RETRY_NUDGE: 0.2 }],
    ['tau', 0.45],
    ['model_action', 'RETRY_NUDGE'],
    ['llm_decision', 'WAIT'],
    ['guardrail_verdict', 'ALLOW'],
    ['expected_incremental_value', 12.5],
  ])('should reject client-injected ML/engine field "%s"', async (field, val) => {
    const { errors } = await transformAndValidate({
      payment_id: 'pay_000001_a1',
      [field]: val,
    });

    expect(errors.length).toBeGreaterThan(0);
    const extraError = errors.find((e) => e.property === field);
    expect(extraError).toBeDefined();
    expect(extraError?.constraints).toHaveProperty('whitelistValidation');
    expect(extraError?.constraints?.whitelistValidation).toContain(
      `property ${field} should not exist`,
    );
  });

  // ── FEATURE-BASED EVALUATION TESTS ──────────────────────────────────────────

  const validFeatures = {
    amount: 1499.0,
    attempt_number: 1,
    dynamic_success_rate: 0.65,
    cumulative_failures: 0,
    consecutive_failed_cycles: 0,
    notification_engagement_score: 0.8,
    contact_response_score: 0.5,
    payment_method: 'card',
    failure_reason: 'insufficient_funds',
  };

  it('should pass validation with valid payment_id and complete 9 observable features', async () => {
    const { errors, instance } = await transformAndValidate({
      payment_id: 'pay_999999_a1',
      features: validFeatures,
    });

    expect(errors.length).toBe(0);
    expect(instance.features).toBeDefined();
    expect(instance.features?.amount).toBe(1499.0);
    expect(instance.features?.payment_method).toBe('card');
  });

  it('should accept optional operational/guardrail fields with valid values', async () => {
    const { errors } = await transformAndValidate({
      payment_id: 'pay_999999_a1',
      features: {
        ...validFeatures,
        status: 'failed',
        consecutive_failures: 1,
        retry_count_current_cycle: 1,
        lifetime_escalations: 0,
        interventions_last_7_days: 1,
      },
    });

    expect(errors.length).toBe(0);
  });

  it.each([
    'amount',
    'attempt_number',
    'dynamic_success_rate',
    'cumulative_failures',
    'consecutive_failed_cycles',
    'notification_engagement_score',
    'contact_response_score',
    'payment_method',
    'failure_reason',
  ])('should reject features when required field "%s" is missing', async (missingField) => {
    const incomplete = { ...validFeatures };
    delete (incomplete as any)[missingField];

    const { errors } = await transformAndValidate({
      payment_id: 'pay_999999_a1',
      features: incomplete,
    });

    expect(errors.length).toBeGreaterThan(0);
    const featuresError = errors.find((e) => e.property === 'features');
    expect(featuresError).toBeDefined();
    const nestedErrors = featuresError?.children || [];
    expect(nestedErrors.some((e) => e.property === missingField)).toBe(true);
  });

  it.each([
    ['amount', 0, 'amount must be positive'],
    ['amount', -50, 'amount must be positive'],
    ['attempt_number', 0, 'attempt_number must be >= 1'],
    ['attempt_number', -1, 'attempt_number must be >= 1'],
    ['dynamic_success_rate', -0.1, 'dynamic_success_rate must be in [0, 1]'],
    ['dynamic_success_rate', 1.1, 'dynamic_success_rate must be in [0, 1]'],
    ['cumulative_failures', -1, 'cumulative_failures must be >= 0'],
    ['consecutive_failed_cycles', -1, 'consecutive_failed_cycles must be >= 0'],
    ['notification_engagement_score', 1.5, 'notification_engagement_score in [0, 1]'],
    ['contact_response_score', -0.01, 'contact_response_score in [0, 1]'],
  ])('should reject invalid numeric range for "%s": %s', async (field, invalidVal) => {
    const { errors } = await transformAndValidate({
      payment_id: 'pay_999999_a1',
      features: { ...validFeatures, [field]: invalidVal },
    });

    expect(errors.length).toBeGreaterThan(0);
    const featuresError = errors.find((e) => e.property === 'features');
    expect(featuresError).toBeDefined();
    const nested = featuresError?.children || [];
    expect(nested.some((e) => e.property === field)).toBe(true);
  });

  it.each([
    ['payment_method', 'bitcoin'],
    ['payment_method', 'cheque'],
    ['failure_reason', 'dog_ate_my_card'],
    ['failure_reason', 'unknown_glitch'],
  ])('should reject invalid categorical vocabulary for "%s": %s', async (field, invalidVal) => {
    const { errors } = await transformAndValidate({
      payment_id: 'pay_999999_a1',
      features: { ...validFeatures, [field]: invalidVal },
    });

    expect(errors.length).toBeGreaterThan(0);
    const featuresError = errors.find((e) => e.property === 'features');
    expect(featuresError).toBeDefined();
    const nested = featuresError?.children || [];
    expect(nested.some((e) => e.property === field)).toBe(true);
  });

  it.each([
    ['amount', NaN],
    ['amount', Infinity],
    ['amount', -Infinity],
    ['dynamic_success_rate', NaN],
    ['dynamic_success_rate', Infinity],
  ])('should reject NaN or Infinity for numeric field "%s"', async (field, val) => {
    const { errors } = await transformAndValidate({
      payment_id: 'pay_999999_a1',
      features: { ...validFeatures, [field]: val },
    });

    expect(errors.length).toBeGreaterThan(0);
    const featuresError = errors.find((e) => e.property === 'features');
    expect(featuresError).toBeDefined();
  });

  it.each([
    'assigned_T',
    'realized_Y',
    'p_success_retry',
    'hidden_type',
    'natural_recovery_probability',
    'expected_net_value',
    'arbitrary_extra_field',
  ])('should reject undeclared or forbidden feature field "%s"', async (forbiddenField) => {
    const { errors } = await transformAndValidate({
      payment_id: 'pay_999999_a1',
      features: { ...validFeatures, [forbiddenField]: 123 },
    });

    expect(errors.length).toBeGreaterThan(0);
    const featuresError = errors.find((e) => e.property === 'features');
    expect(featuresError).toBeDefined();
    const nested = featuresError?.children || [];
    const extra = nested.find((e) => e.property === forbiddenField);
    expect(extra).toBeDefined();
    expect(extra?.constraints).toHaveProperty('whitelistValidation');
  });

  it.each([
    ['status', 'unsupported_status'],
    ['consecutive_failures', -5],
    ['retry_count_current_cycle', -2],
    ['lifetime_escalations', -1],
    ['interventions_last_7_days', -1],
  ])('should reject invalid optional guardrail state for "%s": %s', async (field, invalidVal) => {
    const { errors } = await transformAndValidate({
      payment_id: 'pay_999999_a1',
      features: { ...validFeatures, [field]: invalidVal },
    });

    expect(errors.length).toBeGreaterThan(0);
    const featuresError = errors.find((e) => e.property === 'features');
    expect(featuresError).toBeDefined();
    const nested = featuresError?.children || [];
    expect(nested.some((e) => e.property === field)).toBe(true);
  });
});
