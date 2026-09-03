import { Module } from '@nestjs/common';
import { HealthController } from './health.controller';
import { DecisionEngineModule } from '../decision-engine/decision-engine.module';

@Module({
  imports: [DecisionEngineModule],
  controllers: [HealthController],
})
export class HealthModule {}
