import { ApiProperty } from '@nestjs/swagger';
import { ErrorCode } from '../filters/http-exception.filter';

export class ErrorDetailDto {
  @ApiProperty({
    description: 'Standardized machine-readable error code.',
    enum: [
      'VALIDATION_ERROR',
      'DECISION_ENGINE_UNAVAILABLE',
      'DECISION_ENGINE_TIMEOUT',
      'DECISION_ENGINE_ERROR',
      'INTERNAL_ERROR',
    ],
    example: 'VALIDATION_ERROR',
  })
  code: ErrorCode;

  @ApiProperty({
    description: 'Sanitized human-readable error description.',
    example:
      'payment_id must match the format pay_XXXXXX_aY (e.g. pay_000001_a1)',
  })
  message: string;

  @ApiProperty({
    description:
      'Correlation identifier traceable end-to-end across NestJS and Python service logs.',
    example: 'c4b8e21a-79a1-4621-8f52-7bfb809831a2',
  })
  request_id: string;
}

export class ErrorEnvelopeDto {
  @ApiProperty({
    description:
      'Standardized error envelope object. Notice: no top-level "message" exists outside "error".',
    type: ErrorDetailDto,
  })
  error: ErrorDetailDto;
}
