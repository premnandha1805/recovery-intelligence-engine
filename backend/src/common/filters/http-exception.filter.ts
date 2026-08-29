import {
  ExceptionFilter,
  Catch,
  ArgumentsHost,
  HttpException,
  HttpStatus,
  Logger,
} from '@nestjs/common';
import { Request, Response } from 'express';
import { randomUUID } from 'crypto';
import {
  DecisionEngineErrorException,
  DecisionEngineTimeoutException,
  DecisionEngineUnavailableException,
} from '../exceptions/decision-engine.exceptions';
import { StructuredLogger } from '../logging/structured-logger';

export type ErrorCode =
  | 'VALIDATION_ERROR'
  | 'DECISION_ENGINE_UNAVAILABLE'
  | 'DECISION_ENGINE_TIMEOUT'
  | 'DECISION_ENGINE_ERROR'
  | 'INTERNAL_ERROR';

export interface ErrorEnvelope {
  error: {
    code: ErrorCode;
    message: string;
    request_id: string;
  };
}

/**
 * Sanitize error message to prevent leakage of secrets, keys, paths, or stack traces.
 */
function sanitizeErrorMessage(msg: string): string {
  if (!msg || typeof msg !== 'string') {
    return 'An error occurred';
  }
  let clean = msg.split(/\n\s*at /)[0].trim();
  clean = clean.split(/Traceback \(most recent call last\):/)[0].trim();

  // Redact Azure credentials and connection strings
  clean = clean.replace(/DefaultEndpointsProtocol=[^;]+;?/gi, '[REDACTED];');
  clean = clean.replace(/AccountKey=[^;\s"']+;?/gi, 'AccountKey=[REDACTED];');
  clean = clean.replace(/AccountName=[^;\s"']+;?/gi, 'AccountName=[REDACTED];');
  clean = clean.replace(/SharedAccessSignature=[^;\s"']+;?/gi, 'SharedAccessSignature=[REDACTED];');

  // Redact Bearer tokens, API keys, secrets, passwords
  clean = clean.replace(/(?:api[_-]?key|password|secret|token)\s*[:=]\s*['"]?[a-zA-Z0-9_\-\.]+['"]?/gi, '[REDACTED]');
  clean = clean.replace(/\b(bearer\s+)[a-zA-Z0-9_\-\.]+/gi, '$1[REDACTED]');
  clean = clean.replace(/\bAIza[0-9A-Za-z-_]{35}\b/g, '[REDACTED]');
  clean = clean.replace(/\bsk-[a-zA-Z0-9_\-]{16,}\b/g, '[REDACTED]');

  // Redact Windows and Unix filesystem paths
  clean = clean.replace(/[a-zA-Z]:\\[^:\s"'<>|?*]+/g, '[REDACTED_PATH]');
  clean = clean.replace(/(?:\/[a-zA-Z0-9_\.\-]+){2,}/g, '[REDACTED_PATH]');

  return clean;
}

@Catch()
export class HttpExceptionFilter implements ExceptionFilter {
  private readonly structuredLogger = new StructuredLogger(HttpExceptionFilter.name);

  catch(exception: unknown, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const request = ctx.getRequest<Request>();
    const response = ctx.getResponse<Response>();

    let status = HttpStatus.INTERNAL_SERVER_ERROR;
    let code: ErrorCode = 'INTERNAL_ERROR';
    let message = 'Internal server error';
    let exceptionRequestId: string | undefined = undefined;

    if (exception instanceof DecisionEngineUnavailableException) {
      status = HttpStatus.SERVICE_UNAVAILABLE; // 503
      code = 'DECISION_ENGINE_UNAVAILABLE';
      message = exception.message || 'Python decision engine service is unavailable';
      exceptionRequestId = exception.requestId;
    } else if (exception instanceof DecisionEngineTimeoutException) {
      status = HttpStatus.SERVICE_UNAVAILABLE; // 503
      code = 'DECISION_ENGINE_TIMEOUT';
      message = exception.message || 'Python decision engine request timed out';
      exceptionRequestId = exception.requestId;
    } else if (exception instanceof DecisionEngineErrorException) {
      status = HttpStatus.BAD_GATEWAY; // 502
      code = 'DECISION_ENGINE_ERROR';
      message = exception.message || 'Decision engine returned upstream error';
      exceptionRequestId = exception.requestId;
    } else if (exception instanceof HttpException) {
      status = exception.getStatus();
      const res = exception.getResponse();

      if (typeof res === 'object' && res !== null) {
        const resObj = res as Record<string, any>;
        exceptionRequestId = resObj.request_id || resObj.error?.request_id;

        const candidateCode = resObj.code || resObj.error?.code;
        if (
          candidateCode === 'VALIDATION_ERROR' ||
          candidateCode === 'DECISION_ENGINE_UNAVAILABLE' ||
          candidateCode === 'DECISION_ENGINE_TIMEOUT' ||
          candidateCode === 'DECISION_ENGINE_ERROR'
        ) {
          code = candidateCode;
        } else if (status === HttpStatus.BAD_REQUEST) {
          code = 'VALIDATION_ERROR';
        } else if (status === HttpStatus.SERVICE_UNAVAILABLE) {
          code = 'DECISION_ENGINE_UNAVAILABLE';
        } else if (status === HttpStatus.BAD_GATEWAY) {
          code = 'DECISION_ENGINE_ERROR';
        } else {
          status = HttpStatus.INTERNAL_SERVER_ERROR;
          code = 'INTERNAL_ERROR';
        }

        if (code === 'INTERNAL_ERROR') {
          message = 'Internal server error';
        } else if (resObj.message) {
          if (Array.isArray(resObj.message)) {
            message = resObj.message.join('; ');
          } else {
            message = String(resObj.message);
          }
        } else if (resObj.error?.message) {
          message = String(resObj.error.message);
        } else {
          message = exception.message;
        }
      } else if (typeof res === 'string') {
        if (status === HttpStatus.BAD_REQUEST) {
          code = 'VALIDATION_ERROR';
          message = res;
        } else if (status === HttpStatus.SERVICE_UNAVAILABLE) {
          code = 'DECISION_ENGINE_UNAVAILABLE';
          message = res;
        } else if (status === HttpStatus.BAD_GATEWAY) {
          code = 'DECISION_ENGINE_ERROR';
          message = res;
        } else {
          status = HttpStatus.INTERNAL_SERVER_ERROR;
          code = 'INTERNAL_ERROR';
          message = 'Internal server error';
        }
      }
    } else {
      status = HttpStatus.INTERNAL_SERVER_ERROR;
      code = 'INTERNAL_ERROR';
      message = 'Internal server error';
    }

    // Conservative sanitization: never leak sensitive information
    if (code === 'INTERNAL_ERROR') {
      message = 'Internal server error';
    } else {
      message = sanitizeErrorMessage(message);
    }

    // Resolve correlation request_id:
    // 1. incoming X-Request-Id
    // 2. existing request-scoped requestId
    // 3. generated UUID
    const rawHeader = request?.headers?.['x-request-id'];
    const clientHeaderId =
      typeof rawHeader === 'string' && rawHeader.trim()
        ? rawHeader.trim()
        : undefined;

    const existingReqId =
      typeof (request as any)?.requestId === 'string' &&
      (request as any).requestId.trim()
        ? (request as any).requestId.trim()
        : undefined;

    const requestId =
      clientHeaderId ||
      existingReqId ||
      exceptionRequestId ||
      (request ? ((request as any).requestId = randomUUID()) : randomUUID());

    if (request && !(request as any).requestId) {
      (request as any).requestId = requestId;
    }

    // Return X-Request-Id: <resolved request_id> on every response
    response.setHeader('x-request-id', requestId);

    const paymentId =
      typeof (request?.body as any)?.payment_id === 'string'
        ? (request?.body as any).payment_id
        : undefined;

    // Event E: request_error (structured JSON logging)
    this.structuredLogger.error('request_error', requestId, {
      payment_id: paymentId,
      code,
      status,
      message,
    });

    // Strict error envelope: contains ONLY the "error" top-level key
    const errorBody: ErrorEnvelope = {
      error: {
        code,
        message,
        request_id: requestId,
      },
    };

    response.status(status).json(errorBody);
  }
}


