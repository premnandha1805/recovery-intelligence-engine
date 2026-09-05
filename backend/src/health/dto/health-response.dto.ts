import { ApiProperty } from '@nestjs/swagger';
import { DecisionEngineDependencyStatus } from '../../decision-engine/decision-engine.service';

export class DependenciesStatusDto {
  @ApiProperty({
    description:
      'Operational status of the downstream Python Decision Engine dependency. ' +
      '"ok": healthy; "unreachable": network/connection failure; "timeout": dependency response exceeded timeout deadline (~1s).',
    enum: ['ok', 'unreachable', 'timeout'],
    example: 'ok',
  })
  decision_engine: DecisionEngineDependencyStatus;
}

export class HealthResponseDto {
  @ApiProperty({
    description:
      'Overall system availability status. "ok": all dependencies available; "degraded": downstream dependency unreachable or timed out.',
    enum: ['ok', 'degraded'],
    example: 'ok',
  })
  status: 'ok' | 'degraded';

  @ApiProperty({
    description: 'API service name.',
    example: 'recovery-intelligence-api',
  })
  service: 'recovery-intelligence-api';

  @ApiProperty({
    description: 'Diagnostic map of subsystem dependencies.',
    type: DependenciesStatusDto,
  })
  dependencies: DependenciesStatusDto;
}
