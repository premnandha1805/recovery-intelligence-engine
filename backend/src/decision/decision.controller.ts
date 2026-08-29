import {
  Body,
  Controller,
  Headers,
  HttpCode,
  HttpStatus,
  Post,
  Req,
  Res,
} from '@nestjs/common';
import { Request, Response } from 'express';
import {
  ApiBadRequestResponse,
  ApiBadGatewayResponse,
  ApiBody,
  ApiExtraModels,
  ApiHeader,
  ApiInternalServerErrorResponse,
  ApiOkResponse,
  ApiOperation,
  ApiServiceUnavailableResponse,
  ApiTags,
  getSchemaPath,
} from '@nestjs/swagger';
import { DecisionService } from './decision.service';
import { CreateDecisionDto } from './dto/create-decision.dto';
import { DecisionResponseDto } from './dto/decision-response.dto';
import { ErrorEnvelopeDto } from '../common/dto/error-envelope.dto';

@ApiTags('Decisions')
@ApiExtraModels(CreateDecisionDto, DecisionResponseDto, ErrorEnvelopeDto)
@Controller('decisions')
export class DecisionController {
  constructor(private readonly decisionService: DecisionService) {}

  @Post()
  @HttpCode(HttpStatus.OK)
  @ApiOperation({
    summary: 'Evaluate recovery decision for a failed payment',
    description:
      'Evaluates optimal payment recovery action using causal uplift modeling, contextual feature synthesis, ' +
      'policy optimization, safety guardrail checks, and Azure OpenAI reasoning.\n\n' +
      '### Idempotency Guarantee\n' +
      'This endpoint is **idempotent per `payment_id` by default**. A repeated call with the same `payment_id` ' +
      'returns the previously computed decision (surfaced via `decision_source = "cache"`) without re-invoking the LLM. ' +
      'To force fresh re-evaluation, set `force_recompute: true`. Idempotency is maintained in the decision engine SQLite cache.\n\n' +
      '### Strict Payload Validation\n' +
      'Any additional, undeclared, or unrecognized request fields (e.g. extra ML attributes or guardrail flags) are ' +
      '**strictly rejected with HTTP 400 `VALIDATION_ERROR`**. No silent property stripping occurs.\n\n' +
      '### Request Correlation (X-Request-Id)\n' +
      '- If client supplies `X-Request-Id`, it is preserved end-to-end without regeneration and returned in both header and body.\n' +
      '- If omitted, the gateway generates a canonical RFC 4122 v4 UUID and returns it identically.\n' +
      '- All success responses, error envelopes, and structured server logs are traceable by this identifier.\n\n' +
      '### Error Envelope Guarantee\n' +
      'All failures return the exact envelope `{ "error": { "code", "message", "request_id" } }` with no top-level "message" field.',
  })
  @ApiHeader({
    name: 'x-request-id',
    description:
      'Optional correlation ID. If provided, honored verbatim and echoed in response header and body. If omitted, a UUID v4 is generated.',
    required: false,
    schema: {
      type: 'string',
      example: 'req-client-transaction-001',
    },
  })
  @ApiBody({
    type: CreateDecisionDto,
    description: 'Payment decision evaluation request payload.',
    examples: {
      'standard-request': {
        summary: 'Standard Evaluation Request',
        description: 'Evaluate payment recovery idempotently using default caching.',
        value: {
          payment_id: 'pay_000001_a1',
          force_recompute: false,
        },
      },
      'forced-recompute': {
        summary: 'Forced Recomputation Request',
        description: 'Bypass persistent cache and force fresh inference and reasoning.',
        value: {
          payment_id: 'pay_000001_a1',
          force_recompute: true,
        },
      },
    },
  })
  @ApiOkResponse({
    description:
      'Payment recovery decision successfully resolved. Response adheres strictly to the 11-field contract.',
    content: {
      'application/json': {
        schema: { $ref: getSchemaPath(DecisionResponseDto) },
        examples: {
          'normal-decision': {
            summary: 'Normal Decision (Model & LLM Agree)',
            description:
              'Freshly evaluated decision where causal policy and reasoning engine recommend RETRY_NUDGE with low churn risk.',
            value: {
              payment_id: 'pay_000001_a1',
              model_decision: 'RETRY_NUDGE',
              llm_decision: 'RETRY_NUDGE',
              guardrail_overridden: false,
              guardrail_reason: null,
              final_action: 'RETRY_NUDGE',
              confidence: 0.85,
              risk_level: 'LOW',
              reasoning: 'Evaluation succeeded with high confidence',
              decision_source: 'FOUNDRY_REASONING',
              request_id: 'c4b8e21a-79a1-4621-8f52-7bfb809831a2',
            },
          },
          'guardrail-override': {
            summary: 'Guardrail-Overridden Decision',
            description:
              'Safety rule intervened: causal model recommended retry nudge, but weekly customer intervention cap triggered an override to WAIT.',
            value: {
              payment_id: 'pay_000002_a2',
              model_decision: 'RETRY_NUDGE',
              llm_decision: 'RETRY_NUDGE',
              guardrail_overridden: true,
              guardrail_reason: 'Maximum interventions in 7-day window reached',
              final_action: 'WAIT',
              confidence: 0.9,
              risk_level: 'MEDIUM',
              reasoning:
                'Model recommended retry nudge but customer reached weekly intervention cap',
              decision_source: 'FOUNDRY_REASONING',
              request_id: 'd5a9f31b-80b2-4732-9a63-8cfc910942b3',
            },
          },
          'cached-decision': {
            summary: 'Cached Decision (Idempotent Passthrough)',
            description:
              'Repeated request for same payment_id: returned directly from persistent SQLite audit cache without a second LLM call (observable via decision_source = "cache").',
            value: {
              payment_id: 'pay_000001_a1',
              model_decision: 'RETRY_NUDGE',
              llm_decision: 'RETRY_NUDGE',
              guardrail_overridden: false,
              guardrail_reason: null,
              final_action: 'RETRY_NUDGE',
              confidence: 0.85,
              risk_level: 'LOW',
              reasoning: 'Loaded from SQLite decision_audit cache',
              decision_source: 'cache',
              request_id: 'e6ba042c-91c3-4843-ab74-9ded021053c4',
            },
          },
          'unknown-payment-fallback': {
            summary: 'Unknown Payment Fallback (Safe Error Path)',
            description:
              'Payment ID was not found in dataset context: Python engine safely routed to WAIT without crashing or converting into infrastructure error.',
            value: {
              payment_id: 'pay_999999_a1',
              model_decision: 'N/A — error path',
              llm_decision: 'N/A — error path',
              guardrail_overridden: false,
              guardrail_reason: 'Bypassed due to error: PAYMENT_NOT_FOUND',
              final_action: 'WAIT',
              confidence: 0.0,
              risk_level: 'none',
              reasoning: 'Error: PAYMENT_NOT_FOUND',
              decision_source: 'error_path',
              request_id: 'f7cb153d-02d4-4954-bc85-0efe132164d5',
            },
          },
        },
      },
    },
  })
  @ApiBadRequestResponse({
    description:
      'HTTP 400 VALIDATION_ERROR: Client payload is missing required fields, has invalid format, or includes extra unrecognized fields.',
    content: {
      'application/json': {
        schema: { $ref: getSchemaPath(ErrorEnvelopeDto) },
        examples: {
          'invalid-format': {
            summary: '400 VALIDATION_ERROR — Malformed payment_id',
            description: 'Validation failed due to non-matching regex pattern.',
            value: {
              error: {
                code: 'VALIDATION_ERROR',
                message:
                  'payment_id must match the format pay_XXXXXX_aY (e.g. pay_000001_a1)',
                request_id: 'c4b8e21a-79a1-4621-8f52-7bfb809831a2',
              },
            },
          },
          'unexpected-field': {
            summary: '400 VALIDATION_ERROR — Unexpected Extra Field',
            description: 'Strict whitelisting rejected unrecognized client property.',
            value: {
              error: {
                code: 'VALIDATION_ERROR',
                message: 'property net_value should not exist',
                request_id: 'c4b8e21a-79a1-4621-8f52-7bfb809831a2',
              },
            },
          },
        },
      },
    },
  })
  @ApiBadGatewayResponse({
    description:
      'HTTP 502 DECISION_ENGINE_ERROR: Upstream Python decision engine returned a 5xx error or invalid payload.',
    content: {
      'application/json': {
        schema: { $ref: getSchemaPath(ErrorEnvelopeDto) },
        examples: {
          'upstream-error': {
            summary: '502 DECISION_ENGINE_ERROR',
            description: 'Downstream inference process encountered an internal failure.',
            value: {
              error: {
                code: 'DECISION_ENGINE_ERROR',
                message: 'Decision engine returned upstream error',
                request_id: 'a1da264e-13e5-4065-cd96-1fff243275e6',
              },
            },
          },
        },
      },
    },
  })
  @ApiServiceUnavailableResponse({
    description:
      'HTTP 503 Dependency Failure: Decision engine is unavailable or request exceeded timeout deadline (8s).',
    content: {
      'application/json': {
        schema: { $ref: getSchemaPath(ErrorEnvelopeDto) },
        examples: {
          'engine-unavailable': {
            summary: '503 DECISION_ENGINE_UNAVAILABLE',
            description: 'Connection refused or downstream network failure.',
            value: {
              error: {
                code: 'DECISION_ENGINE_UNAVAILABLE',
                message: 'Python decision engine service is unavailable',
                request_id: 'b2eb375f-24f6-4176-de07-2aaa354386f7',
              },
            },
          },
          'engine-timeout': {
            summary: '503 DECISION_ENGINE_TIMEOUT',
            description:
              'Decision engine request exceeded the configured 8000ms deadline.',
            value: {
              error: {
                code: 'DECISION_ENGINE_TIMEOUT',
                message: 'Python decision engine request timed out',
                request_id: 'c3fc4860-3507-4287-ef18-3bbb46549708',
              },
            },
          },
        },
      },
    },
  })
  @ApiInternalServerErrorResponse({
    description:
      'HTTP 500 INTERNAL_ERROR: Unexpected unhandled server exception. Implementation details and stack traces are suppressed.',
    content: {
      'application/json': {
        schema: { $ref: getSchemaPath(ErrorEnvelopeDto) },
        examples: {
          'internal-error': {
            summary: '500 INTERNAL_ERROR',
            description: 'Unhandled gateway error. Raw error details are sanitized and masked.',
            value: {
              error: {
                code: 'INTERNAL_ERROR',
                message: 'Internal server error',
                request_id: 'd40d5971-4618-4398-f029-4ccc57650819',
              },
            },
          },
        },
      },
    },
  })
  async createDecision(
    @Body() createDecisionDto: CreateDecisionDto,
    @Headers('x-request-id') requestId?: string,
    @Req() req?: Request,
    @Res({ passthrough: true }) res?: Response,
  ): Promise<DecisionResponseDto> {
    const result = await this.decisionService.createDecision(
      createDecisionDto,
      requestId,
      req,
    );
    if (res && result.request_id) {
      res.setHeader('x-request-id', result.request_id);
    }
    return result;
  }
}
