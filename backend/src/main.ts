import { NestFactory } from '@nestjs/core';
import { Logger, ValidationPipe } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { AppModule } from './app.module';
import { HttpExceptionFilter } from './common/filters/http-exception.filter';

async function bootstrap() {
  const logger = new Logger('Bootstrap');
  const app = await NestFactory.create(AppModule);

  // Global ValidationPipe: strict DTO whitelisting to reject undeclared client fields [FIX-6]
  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      forbidNonWhitelisted: true,
      transform: true,
    }),
  );

  // Global Exception Filter: maps errors to shared envelope (400, 502, 503)
  app.useGlobalFilters(new HttpExceptionFilter());

  // Enable shutdown hooks for graceful termination [FIX-8]
  app.enableShutdownHooks();

  // CORS left disabled in Day 7A

  const configService = app.get(ConfigService);
  const port = configService.get<number>('PORT', 3000);

  await app.listen(port);
  logger.log(`Recovery Intelligence Backend listening on port ${port}`);
}

bootstrap();
