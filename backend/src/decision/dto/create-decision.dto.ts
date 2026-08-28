import { IsBoolean, IsNotEmpty, IsOptional, IsString } from 'class-validator';

export class CreateDecisionDto {
  @IsString()
  @IsNotEmpty()
  payment_id: string;

  @IsOptional()
  @IsBoolean()
  force_recompute?: boolean = false;
}
