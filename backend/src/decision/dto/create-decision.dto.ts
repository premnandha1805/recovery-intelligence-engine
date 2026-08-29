import {
  IsBoolean,
  IsNotEmpty,
  IsOptional,
  IsString,
  Matches,
} from 'class-validator';
import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';

/**
 * Request DTO for POST /decisions.
 *
 * Requirements & Contract:
 * - payment_id: required, non-empty, matches existing repository payment format (pay_XXXXXX_aY).
 * - force_recompute: optional boolean, defaults to false.
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
}
