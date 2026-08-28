import { Module } from '@nestjs/common';
import { DecisionController } from './decision.controller';
import { DecisionService } from './decision.service';
import { DecisionEngineModule } from '../decision-engine/decision-engine.module';

@Module({
  imports: [DecisionEngineModule],
  controllers: [DecisionController],
  providers: [DecisionService],
  exports: [DecisionService],
})
export class DecisionModule {}
