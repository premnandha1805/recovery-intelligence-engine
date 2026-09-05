import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { DecisionEngineService } from './decision-engine.service';
import { DecisionEngineAdapter } from './decision-engine.adapter';

@Module({
  imports: [ConfigModule],
  providers: [DecisionEngineService, DecisionEngineAdapter],
  exports: [DecisionEngineService, DecisionEngineAdapter],
})
export class DecisionEngineModule {}
