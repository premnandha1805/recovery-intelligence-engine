import { Controller, Get, HttpStatus, Res } from '@nestjs/common';
import { Response } from 'express';
import {
  ApiExtraModels,
  ApiOkResponse,
  ApiOperation,
  ApiServiceUnavailableResponse,
  ApiTags,
  getSchemaPath,
} from '@nestjs/swagger';
import { DecisionEngineService } from '../decision-engine/decision-engine.service';
import { HealthResponseDto } from './dto/health-response.dto';

@ApiTags('Health')
@ApiExtraModels(HealthResponseDto)
@Controller('health')
export class HealthController {
  constructor(
    private readonly decisionEngineService: DecisionEngineService,
  ) {}

  @Get()
  @ApiOperation({
    summary: 'Check API and dependency health status',
    description:
      'Performs an active diagnostic check of the API gateway and probes the downstream ' +
      'Python Decision Engine service via an isolated fast health timeout (~1s). ' +
      'Returns 200 OK if all subsystems are operational, or 503 Service Unavailable if any downstream dependency fails.',
  })
  @ApiOkResponse({
    description: 'System and dependencies are fully operational.',
    content: {
      'application/json': {
        schema: { $ref: getSchemaPath(HealthResponseDto) },
        examples: {
          'healthy-system': {
            summary: 'Healthy Dependency (HTTP 200)',
            description: 'Gateway is alive and Python decision engine responded with status "ok".',
            value: {
              status: 'ok',
              service: 'recovery-intelligence-api',
              dependencies: {
                decision_engine: 'ok',
              },
            },
          },
        },
      },
    },
  })
  @ApiServiceUnavailableResponse({
    description:
      'System is degraded due to downstream dependency failure. Documented variants:\n' +
      '- **unreachable**: Downstream Python service refused connection or network failed.\n' +
      '- **timeout**: Downstream Python service did not respond within the 1-second health timeout probe.',
    content: {
      'application/json': {
        schema: { $ref: getSchemaPath(HealthResponseDto) },
        examples: {
          'dependency-unreachable': {
            summary: 'Degraded — Dependency Unreachable',
            description:
              'Python service connection was refused or host unreachable. Gateway returned 503 without hanging.',
            value: {
              status: 'degraded',
              service: 'recovery-intelligence-api',
              dependencies: {
                decision_engine: 'unreachable',
              },
            },
          },
          'dependency-timeout': {
            summary: 'Degraded — Dependency Timeout (~1s)',
            description:
              'Python service health probe exceeded the fast 1000ms health check timeout deadline.',
            value: {
              status: 'degraded',
              service: 'recovery-intelligence-api',
              dependencies: {
                decision_engine: 'timeout',
              },
            },
          },
        },
      },
    },
  })
  async check(@Res({ passthrough: true }) res: Response): Promise<HealthResponseDto> {
    const depStatus = await this.decisionEngineService.checkHealth();
    const isHealthy = depStatus === 'ok';

    // Never hide a failed Python decision engine behind HTTP 200 + "ok"
    if (!isHealthy) {
      res.status(HttpStatus.SERVICE_UNAVAILABLE);
    }

    return {
      status: isHealthy ? 'ok' : 'degraded',
      service: 'recovery-intelligence-api',
      dependencies: {
        decision_engine: depStatus,
      },
    };
  }
}
