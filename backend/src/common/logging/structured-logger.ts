import { Logger } from '@nestjs/common';

export interface StructuredLogPayload {
  timestamp: string;
  service: 'nestjs';
  event: string;
  request_id: string;
  [key: string]: any;
}

export class StructuredLogger {
  private readonly logger: Logger;

  constructor(context: string) {
    this.logger = new Logger(context);
  }

  log(event: string, requestId: string, details: Record<string, any> = {}) {
    const payload: StructuredLogPayload = {
      timestamp: new Date().toISOString(),
      service: 'nestjs',
      event,
      request_id: requestId,
      ...details,
    };
    this.logger.log(JSON.stringify(payload));
  }

  warn(event: string, requestId: string, details: Record<string, any> = {}) {
    const payload: StructuredLogPayload = {
      timestamp: new Date().toISOString(),
      service: 'nestjs',
      event,
      request_id: requestId,
      ...details,
    };
    this.logger.warn(JSON.stringify(payload));
  }

  error(event: string, requestId: string, details: Record<string, any> = {}) {
    const payload: StructuredLogPayload = {
      timestamp: new Date().toISOString(),
      service: 'nestjs',
      event,
      request_id: requestId,
      ...details,
    };
    this.logger.error(JSON.stringify(payload));
  }
}
