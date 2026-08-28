import {
  Body,
  Controller,
  Headers,
  HttpCode,
  HttpStatus,
  Post,
  Req,
} from '@nestjs/common';
import { Request } from 'express';
import { DecisionService } from './decision.service';
import { CreateDecisionDto } from './dto/create-decision.dto';
import { DecisionResponseDto } from './dto/decision-response.dto';

@Controller('decisions')
export class DecisionController {
  constructor(private readonly decisionService: DecisionService) {}

  @Post()
  @HttpCode(HttpStatus.OK)
  async createDecision(
    @Body() createDecisionDto: CreateDecisionDto,
    @Headers('x-request-id') requestId?: string,
    @Req() req?: Request,
  ): Promise<DecisionResponseDto> {
    return this.decisionService.createDecision(
      createDecisionDto,
      requestId,
      req,
    );
  }
}
