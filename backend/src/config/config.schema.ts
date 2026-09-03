import * as Joi from 'joi';

export interface AppConfig {
  PORT: number;
  DECISION_ENGINE_URL: string;
  DECISION_ENGINE_TIMEOUT_MS: number;
  HEALTH_CHECK_TIMEOUT_MS: number;
  FRONTEND_ORIGIN?: string;
}

export const configValidationSchema = Joi.object({
  PORT: Joi.number().port().default(3000),
  DECISION_ENGINE_URL: Joi.string()
    .uri({ scheme: ['http', 'https'] })
    .default('http://localhost:8000'),
  DECISION_ENGINE_TIMEOUT_MS: Joi.number()
    .integer()
    .positive()
    .default(8000),
  HEALTH_CHECK_TIMEOUT_MS: Joi.number()
    .integer()
    .positive()
    .default(1000),
  FRONTEND_ORIGIN: Joi.string()
    .uri({ scheme: ['http', 'https'] })
    .optional(),
});
