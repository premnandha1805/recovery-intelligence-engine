import { Module } from '@nestjs/common';
import { DecisionEngineService } from './decision-engine.service';

@Module({
  providers: [DecisionEngineService],
  exports: [DecisionEngineService],
})
export class DecisionEngineModule {}
