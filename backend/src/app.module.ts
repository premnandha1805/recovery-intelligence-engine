import { Module } from '@nestjs/common';
import { AppConfigModule } from './config/config.module';
import { HealthModule } from './health/health.module';
import { DecisionModule } from './decision/decision.module';
import { DecisionEngineModule } from './decision-engine/decision-engine.module';

@Module({
  imports: [
    AppConfigModule,
    HealthModule,
    DecisionModule,
    DecisionEngineModule,
  ],
})
export class AppModule {}
