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
});
