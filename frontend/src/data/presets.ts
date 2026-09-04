import { CreateDecisionRequest } from '../types/api';

export interface PresetScenario {
  id: string;
  name: string;
  badge: string;
  description: string;
  verifiedSource: string;
  data: CreateDecisionRequest;
}

export const PRESET_SCENARIOS: PresetScenario[] = [
  {
    id: 'baseline-reliable',
    name: 'Reliable customer / one-off failure',
    badge: 'Baseline Verified',
    description: 'High success rate customer with zero prior cycle failures experiencing a transient UPI network failure.',
    verifiedSource: 'Verified live demo baseline scenario (pay_910099_a1)',
    data: {
      payment_id: 'pay_910099_a1',
      force_recompute: false,
      features: {
        amount: 2500,
        attempt_number: 1,
        dynamic_success_rate: 0.70,
        cumulative_failures: 0,
        consecutive_failed_cycles: 0,
        notification_engagement_score: 0.85,
        contact_response_score: 0.60,
        payment_method: 'upi',
        failure_reason: 'temporary_bank_issue',
      },
    },
  },
  {
    id: 'chronic-risk',
    name: 'Chronic failure / high risk',
    badge: 'High Fatigue Risk',
    description: 'Low-propensity customer with multiple cumulative failures, low contact engagement, and recurring insufficient funds.',
    verifiedSource: 'Verified high-risk pattern in decision engine test suite',
    data: {
      payment_id: 'pay_910099_a3',
      force_recompute: false,
      features: {
        amount: 4500,
        attempt_number: 3,
        dynamic_success_rate: 0.15,
        cumulative_failures: 5,
        consecutive_failed_cycles: 2,
        notification_engagement_score: 0.20,
        contact_response_score: 0.10,
        payment_method: 'card',
        failure_reason: 'insufficient_funds',
      },
    },
  },
  {
    id: 'guardrail-override',
    name: 'Guardrail override',
    badge: 'Verified Hard Safety Rule',
    description: 'Consecutive cycle failure limit reached (3 >= 3). Even if client passes consecutive_failures = 0, safety engine bounds failures to 3 and halts recovery (Action.STOP).',
    verifiedSource: 'decision_engine/test_service.py: line 1138 (Case A: consecutive_failed_cycles = 3, consecutive_failures = 0)',
    data: {
      payment_id: 'pay_999999_a6',
      force_recompute: false,
      features: {
        amount: 1499,
        attempt_number: 1,
        dynamic_success_rate: 0.65,
        cumulative_failures: 3,
        consecutive_failed_cycles: 3,
        consecutive_failures: 0,
        notification_engagement_score: 0.80,
        contact_response_score: 0.50,
        payment_method: 'card',
        failure_reason: 'insufficient_funds',
      },
    },
  },
];
