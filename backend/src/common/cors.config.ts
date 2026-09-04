import { CorsOptions } from '@nestjs/common/interfaces/external/cors-options.interface';

/**
 * Narrowly-scoped CORS configuration for Recovery Intelligence Gateway.
 * Strictly limits allowed origins to:
 * - configured production frontend origin (via FRONTEND_ORIGIN)
 * - local development origins (http://localhost:5173, http://127.0.0.1:5173, http://localhost:3000)
 *
 * Disallows wildcard origins (*) and limits methods to GET, POST, OPTIONS.
 */
export function getCorsConfig(frontendOrigin?: string): CorsOptions {
  const allowedOrigins: string[] = [
    'http://localhost:5173',
    'http://127.0.0.1:5173',
    'http://localhost:3000',
    'https://recovery-intelligence-engine-1.onrender.com',
  ];

  if (frontendOrigin && frontendOrigin.trim() !== '') {
    const trimmed = frontendOrigin.trim().replace(/\/$/, '');
    if (!allowedOrigins.includes(trimmed)) {
      allowedOrigins.push(trimmed);
    }
  }

  return {
    origin: allowedOrigins,
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'X-Request-Id'],
    credentials: true,
  };
}
