import {
  Body,
  Controller,
  Headers,
  HttpCode,
  HttpStatus,
  Post,
  Req,
  Res,
} from '@nestjs/common';
import { Request, Response } from 'express';
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
    @Res({ passthrough: true }) res?: Response,
  ): Promise<DecisionResponseDto> {
    const result = await this.decisionService.createDecision(
      createDecisionDto,
      requestId,
      req,
    );
    if (res && result.request_id) {
      res.setHeader('x-request-id', result.request_id);
    }
    return result;
  }
}

