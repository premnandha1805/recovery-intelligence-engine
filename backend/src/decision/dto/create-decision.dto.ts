import {
  IsBoolean,
  IsNotEmpty,
  IsOptional,
  IsString,
  Matches,
} from 'class-validator';

/**
 * Request DTO for POST /decisions.
 *
 * Requirements:
 * - payment_id: required, non-empty, matches existing repository payment format (pay_XXXXXX_aY).
 * - force_recompute: optional boolean, defaults to false.
 * - Global ValidationPipe (whitelist: true, forbidNonWhitelisted: true) rejects any extra fields.
 */
export class CreateDecisionDto {
  @IsString()
  @IsNotEmpty()
  @Matches(/^pay_\d{6}_a\d+$/, {
    message:
      'payment_id must match the format pay_XXXXXX_aY (e.g. pay_000001_a1)',
  })
  payment_id: string;

  @IsOptional()
  @IsBoolean()
  force_recompute?: boolean = false;
}
