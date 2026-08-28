import { Injectable } from '@nestjs/common';
import { CreateDecisionDto } from './dto/create-decision.dto';

@Injectable()
export class DecisionService {
  async createDecision(dto: CreateDecisionDto) {
    // Skeleton implementation — full decision logic deferred to Day 7C/7D
    return {
      message: 'Decision request accepted (skeleton)',
      payment_id: dto.payment_id,
      force_recompute: dto.force_recompute ?? false,
    };
  }
}
