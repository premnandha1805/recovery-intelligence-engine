import { HttpException, HttpStatus } from '@nestjs/common';

/**
 * Thrown when Python Decision Engine service cannot be reached (e.g. connection refused, network down).
 * Maps to HTTP 503 with error code DECISION_ENGINE_UNAVAILABLE.
 */
export class DecisionEngineUnavailableException extends HttpException {
  public readonly requestId?: string;

  constructor(
    message = 'Python decision engine service is unavailable',
    requestId?: string,
  ) {
    super(
      {
        code: 'DECISION_ENGINE_UNAVAILABLE',
        message,
        request_id: requestId,
      },
      HttpStatus.SERVICE_UNAVAILABLE, // 503
    );
    this.requestId = requestId;
  }
}

/**
 * Thrown when Python Decision Engine service request times out.
 * Maps to HTTP 503 with error code DECISION_ENGINE_TIMEOUT.
 */
export class DecisionEngineTimeoutException extends HttpException {
  public readonly requestId?: string;

  constructor(
    message = 'Python decision engine request timed out',
    requestId?: string,
  ) {
    super(
      {
        code: 'DECISION_ENGINE_TIMEOUT',
        message,
        request_id: requestId,
      },
      HttpStatus.SERVICE_UNAVAILABLE, // 503
    );
    this.requestId = requestId;
  }
}

/**
 * Thrown when Python Decision Engine service returns an upstream 5xx or unhandled error.
 * Maps to HTTP 502 with error code DECISION_ENGINE_ERROR.
 */
export class DecisionEngineErrorException extends HttpException {
  public readonly requestId?: string;

  constructor(
    message = 'Decision engine returned upstream error',
    requestId?: string,
  ) {
    super(
      {
        code: 'DECISION_ENGINE_ERROR',
        message,
        request_id: requestId,
      },
      HttpStatus.BAD_GATEWAY, // 502
    );
    this.requestId = requestId;
  }
}
