import { Injectable, NestMiddleware } from '@nestjs/common';
import { Request, Response, NextFunction } from 'express';
import { randomUUID } from 'crypto';

@Injectable()
export class RequestIdMiddleware implements NestMiddleware {
  use(req: Request, res: Response, next: NextFunction) {
    const rawHeader = req.headers['x-request-id'];
    const requestId =
      typeof rawHeader === 'string' && rawHeader.trim()
        ? rawHeader.trim()
        : randomUUID();

    (req as any).requestId = requestId;
    res.setHeader('x-request-id', requestId);
    next();
  }
}
