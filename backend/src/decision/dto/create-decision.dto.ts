import {
  IsBoolean,
  IsEnum,
  IsIn,
  IsInt,
  IsNotEmpty,
  IsNumber,
  IsOptional,
  IsPositive,
  IsString,
  Matches,
  Max,
  Min,
  ValidateNested,
} from 'class-validator';
import { Type } from 'class-transformer';
import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';

/**
 * Canonical payment methods sourced from models.schemas.PaymentMethod.
 */
export enum PaymentMethod {
  CARD = 'card',
  UPI = 'upi',
  NETBANKING = 'netbanking',
  WALLET = 'wallet',
}

/**
 * Canonical failure reasons sourced from models.schemas.FailureReason and simulator.config.FAILURE_REASON_WEIGHTS.
 */
export enum FailureReason {
  INSUFFICIENT_FUNDS = 'insufficient_funds',
  BANK_DECLINE = 'bank_decline',
  NETWORK_ERROR = 'network_error',
  EXPIRED_CARD = 'expired_card',
  AUTHENTICATION_FAILURE = 'authentication_failure',
  TEMPORARY_BANK_ISSUE = 'temporary_bank_issue',
}

/**
 * Observable payment features DTO for evaluating new/unseen payments.
 * Contains the 9 observable features consumed by the causal ML model,
 * plus optional operational fields with strict validation.
 */
export class PaymentFeaturesDto {
  @ApiProperty({
    description: 'Payment amount in INR (must be strictly positive).',
    example: 1499.0,
  })
  @IsNumber({ allowNaN: false, allowInfinity: false })
  @IsPositive()
  amount: number;

  @ApiProperty({
    description: 'Current attempt number within the billing cycle (>= 1).',
    example: 1,
  })
  @IsInt()
  @Min(1)
  attempt_number: number;

  @ApiProperty({
    description: 'Customer historical dynamic recovery/success rate [0.0, 1.0].',
    example: 0.65,
  })
  @IsNumber({ allowNaN: false, allowInfinity: false })
  @Min(0)
  @Max(1)
  dynamic_success_rate: number;

  @ApiProperty({
    description: 'Total cumulative payment failures across customer lifetime (>= 0).',
    example: 0,
  })
  @IsInt()
  @Min(0)
  cumulative_failures: number;

  @ApiProperty({
    description: 'Consecutive billing cycles with failed payments (>= 0).',
    example: 0,
  })
  @IsInt()
  @Min(0)
  consecutive_failed_cycles: number;

  @ApiProperty({
    description: 'Customer notification engagement propensity score [0.0, 1.0].',
    example: 0.8,
  })
  @IsNumber({ allowNaN: false, allowInfinity: false })
  @Min(0)
  @Max(1)
  notification_engagement_score: number;

  @ApiProperty({
    description: 'Customer contact response score [0.0, 1.0].',
    example: 0.5,
  })
  @IsNumber({ allowNaN: false, allowInfinity: false })
  @Min(0)
  @Max(1)
  contact_response_score: number;

  @ApiProperty({
    description: 'Payment rail / instrument method.',
    enum: PaymentMethod,
    example: PaymentMethod.CARD,
  })
  @IsString()
  @IsEnum(PaymentMethod, {
    message: 'payment_method must be a valid payment method (card, upi, netbanking, wallet)',
  })
  payment_method: string;

  @ApiProperty({
    description: 'Specific reason code for payment transaction failure.',
    enum: FailureReason,
    example: FailureReason.INSUFFICIENT_FUNDS,
  })
  @IsString()
  @IsEnum(FailureReason, {
    message:
      'failure_reason must be a valid failure reason (network_error, insufficient_funds, temporary_bank_issue, bank_decline, expired_card, authentication_failure)',
  })
  failure_reason: string;

  // ── Optional Operational / Guardrail Overrides ──────────────────────────────

  @ApiPropertyOptional({
    description: 'Optional operational payment status (cold-start default: "failed").',
    enum: ['failed', 'pending', 'success', 'recovered'],
    example: 'failed',
  })
  @IsOptional()
  @IsString()
  @IsIn(['failed', 'pending', 'success', 'recovered'])
  status?: string;

  @ApiPropertyOptional({
    description:
      'Optional consecutive failures count (cold-start default: consecutive_failed_cycles).',
    example: 0,
  })
  @IsOptional()
  @IsInt()
  @Min(0)
  consecutive_failures?: number;

  @ApiPropertyOptional({
    description:
      'Optional retries in current cycle (cold-start default: max(0, attempt_number - 1)).',
    example: 0,
  })
  @IsOptional()
  @IsInt()
  @Min(0)
  retry_count_current_cycle?: number;

  @ApiPropertyOptional({
    description:
      'Optional lifetime escalations count (cold-start default: 0).',
    example: 0,
  })
  @IsOptional()
  @IsInt()
  @Min(0)
  lifetime_escalations?: number;

  @ApiPropertyOptional({
    description:
      'Optional active interventions in last 7 days (cold-start default: 0).',
    example: 0,
  })
  @IsOptional()
  @IsInt()
  @Min(0)
  interventions_last_7_days?: number;
}

/**
 * Request DTO for POST /decisions.
 *
 * Requirements & Contract:
 * - payment_id: required, non-empty, matches existing repository payment format (pay_XXXXXX_aY).
 * - force_recompute: optional boolean, defaults to false.
 * - features: optional nested PaymentFeaturesDto for direct feature-based evaluation of new payments.
 * - Strict Whitelisting: Any additional or unrecognized fields are rejected with HTTP 400 (VALIDATION_ERROR).
 */
export class CreateDecisionDto {
  @ApiProperty({
    description:
      'Unique identifier of the failed payment transaction. Must strictly adhere to pattern pay_XXXXXX_aY.',
    example: 'pay_000001_a1',
    pattern: '^pay_\\d{6}_a\\d+$',
  })
  @IsString()
  @IsNotEmpty()
  @Matches(/^pay_\d{6}_a\d+$/, {
    message:
      'payment_id must match the format pay_XXXXXX_aY (e.g. pay_000001_a1)',
  })
  payment_id: string;

  @ApiPropertyOptional({
    description:
      'When false (default), the call is idempotent and returns the cached decision if one exists for payment_id. ' +
      'When true, bypasses the cache and forces fresh causal inference and reasoning evaluation.',
    example: false,
    default: false,
  })
  @IsOptional()
  @IsBoolean()
  force_recompute?: boolean = false;

  @ApiPropertyOptional({
    description:
      'Optional observable payment features for direct evaluation of a new/unseen payment without database/CSV lookup.',
    type: () => PaymentFeaturesDto,
  })
  @IsOptional()
  @ValidateNested()
  @Type(() => PaymentFeaturesDto)
  features?: PaymentFeaturesDto;
}
