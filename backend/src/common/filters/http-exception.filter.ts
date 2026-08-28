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

export interface ErrorEnvelope {
  error: {
    code: string;
    message: string;
    request_id: string;
  };
}

@Catch()
export class HttpExceptionFilter implements ExceptionFilter {
  private readonly logger = new Logger(HttpExceptionFilter.name);

  catch(exception: unknown, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const request = ctx.getRequest<Request>();
    const response = ctx.getResponse<Response>();

    let status = HttpStatus.INTERNAL_SERVER_ERROR;
    let code = 'INTERNAL_ERROR';
    let message = 'Internal server error';
    let rawMessage: any = undefined;
    let exceptionRequestId: string | undefined = undefined;

    if (exception instanceof HttpException) {
      status = exception.getStatus();
      const res = exception.getResponse();

      if (typeof res === 'object' && res !== null) {
        const resObj = res as Record<string, any>;
        rawMessage = resObj.message;
        exceptionRequestId = resObj.request_id;

        if (resObj.code) {
          code = resObj.code;
        } else if (resObj.error?.code) {
          code = resObj.error.code;
        } else if (status === HttpStatus.BAD_REQUEST) {
          code = 'VALIDATION_ERROR';
        } else if (status === HttpStatus.SERVICE_UNAVAILABLE) {
          code = 'DECISION_ENGINE_UNAVAILABLE';
        } else if (status === HttpStatus.BAD_GATEWAY) {
          code = 'DECISION_ENGINE_ERROR';
        }

        if (resObj.message) {
          if (Array.isArray(resObj.message)) {
            message = resObj.message.join('; ');
          } else {
            message = String(resObj.message);
          }
        } else if (resObj.error?.message) {
          message = String(resObj.error.message);
        }
      } else if (typeof res === 'string') {
        message = res;
      }
    } else {
      this.logger.error(`Unhandled non-HTTP exception: ${exception}`);
    }

    // Resolve correlation request_id without generating an unrelated second ID:
    // 1. Exception-provided request_id (from DecisionService / exceptions)
    // 2. Request property requestId (from RequestIdMiddleware or DecisionService)
    // 3. Client header x-request-id
    // 4. Fallback generated UUID (cached on request to avoid duplicate generation)
    const rawHeader = request?.headers?.['x-request-id'];
    const clientHeaderId =
      typeof rawHeader === 'string' && rawHeader.trim()
        ? rawHeader.trim()
        : undefined;

    const existingReqId = (request as any)?.requestId;

    const requestId =
      clientHeaderId ||
      exceptionRequestId ||
      existingReqId ||
      (request ? ((request as any).requestId = randomUUID()) : randomUUID());

    // Ensure response header carries request_id as well
    response.setHeader('x-request-id', requestId);

    const errorBody: any = {
      error: {
        code,
        message,
        request_id: requestId,
      },
    };

    // Preserve array message at top level for compatibility with validation tests
    if (Array.isArray(rawMessage)) {
      errorBody.message = rawMessage;
    }

    response.status(status).json(errorBody);
  }
}
