import { NestFactory } from '@nestjs/core';
import { Logger, ValidationPipe } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';
import { AppModule } from './app.module';
import { HttpExceptionFilter } from './common/filters/http-exception.filter';
import { getCorsConfig } from './common/cors.config';

async function bootstrap() {
  const logger = new Logger('Bootstrap');
  const app = await NestFactory.create(AppModule);

  const configService = app.get(ConfigService);
  const frontendOrigin = configService.get<string>('FRONTEND_ORIGIN');

  // Narrowly-scoped CORS: allow configured production frontend origin & local dev origins only
  app.enableCors(getCorsConfig(frontendOrigin));
  logger.log(
    `CORS configured for development origins and production origin: ${
      frontendOrigin || 'none (FRONTEND_ORIGIN not set)'
    }`,
  );

  // Global ValidationPipe: strict DTO whitelisting to reject undeclared client fields [FIX-6]
  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      forbidNonWhitelisted: true,
      transform: true,
    }),
  );

  // Global Exception Filter: maps errors to shared envelope (400, 502, 503, 500)
  app.useGlobalFilters(new HttpExceptionFilter());

  // Enable shutdown hooks for graceful termination [FIX-8]
  app.enableShutdownHooks();

  // ── OpenAPI / Swagger Documentation Setup [Day 7I] ─────────────────────────
  const swaggerConfig = new DocumentBuilder()
    .setTitle('Recovery Intelligence Engine API')
    .setDescription(
      'Autonomous Payment Recovery Intelligence & Decisioning Gateway.\n\n' +
      '### Architecture & Guarantees\n' +
      '- **Idempotency**: POST `/decisions` is idempotent per `payment_id` by default. Repeated calls return the previously computed decision (observable via `decision_source = "cache"`) without re-invoking the LLM. Supply `force_recompute=true` to bypass cache and force fresh inference.\n' +
      '- **Strict Whitelisting**: Any unexpected or extra request attributes are rejected with HTTP 400 `VALIDATION_ERROR`.\n' +
      '- **Request Correlation**: The `X-Request-Id` header is preserved end-to-end. If omitted by the caller, the gateway generates an RFC 4122 v4 UUID and returns it in both header and body.\n' +
      '- **Unified Error Envelope**: All failures return `{ "error": { "code", "message", "request_id" } }` with no top-level "message" field.\n' +
      '- **Downstream Health Isolation**: The `GET /health` endpoint probes dependencies via an isolated 1-second timeout, preventing dead services from stalling health probes for the standard 8-second decision deadline.',
    )
    .setVersion('1.0.0')
    .addTag('Decisions', 'Payment recovery decision evaluation')
    .addTag('Health', 'System and dependency health diagnostics')
    .build();

  const document = SwaggerModule.createDocument(app, swaggerConfig);
  SwaggerModule.setup('api/docs', app, document, {
    customSiteTitle: 'Recovery Intelligence Engine API Documentation',
  });

  const port = configService.get<number>('PORT', 3000);

  await app.listen(port);
  logger.log(`Recovery Intelligence Backend listening on port ${port}`);
  logger.log(`Swagger UI available at: http://localhost:${port}/api/docs`);
}

bootstrap();
