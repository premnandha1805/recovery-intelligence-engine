import { Test, TestingModule } from '@nestjs/testing';
import { Logger } from '@nestjs/common';
import { DecisionService } from './decision.service';
import {
  DecisionEngineAdapter,
  PythonDecisionResult,
} from '../decision-engine/decision-engine.adapter';
import {
  DecisionEngineErrorException,
  DecisionEngineTimeoutException,
  DecisionEngineUnavailableException,
} from '../common/exceptions/decision-engine.exceptions';

describe('DecisionService', () => {
  let service: DecisionService;
  let adapter: DecisionEngineAdapter;

  const mockPythonResult: PythonDecisionResult = {
    payment_id: 'pay_000001_a1',
    model_decision: 'RETRY_NUDGE',
    llm_decision: 'RETRY_NUDGE',
    guardrail_overridden: false,
    guardrail_reason: null,
    final_action: 'RETRY_NUDGE',
    confidence: 0.92,
    risk_level: 'low',
    reasoning: 'Customer has high recovery likelihood on retry nudge',
    decision_source: 'FOUNDRY_REASONING',
    request_id: 'req-test-uuid-123',
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        DecisionService,
        {
          provide: DecisionEngineAdapter,
          useValue: {
            evaluate: jest.fn().mockResolvedValue(mockPythonResult),
          },
        },
      ],
    }).compile();

    service = module.get<DecisionService>(DecisionService);
    adapter = module.get<DecisionEngineAdapter>(DecisionEngineAdapter);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  describe('createDecision', () => {
    it('should reuse caller-supplied X-Request-Id as request_id [FIX-10]', async () => {
      const clientReqId = 'req-caller-provided-777';
      const evaluateSpy = jest
        .spyOn(adapter, 'evaluate')
        .mockResolvedValueOnce({
          ...mockPythonResult,
          request_id: clientReqId,
        });

      const result = await service.createDecision(
        { payment_id: 'pay_000001_a1' },
        clientReqId,
      );

      expect(evaluateSpy).toHaveBeenCalledWith(
        'pay_000001_a1',
        clientReqId,
        false,
      );
      expect(result.request_id).toBe(clientReqId);
    });

    it('should generate a UUID request_id if none provided by client [FIX-10]', async () => {
      let capturedReqId: string | undefined;
      jest
        .spyOn(adapter, 'evaluate')
        .mockImplementation(async (pid, reqId, force) => {
          capturedReqId = reqId;
          return { ...mockPythonResult, request_id: reqId };
        });

      const result = await service.createDecision({
        payment_id: 'pay_000001_a1',
      });

      expect(capturedReqId).toBeDefined();
      expect(typeof capturedReqId).toBe('string');
      expect(capturedReqId?.length).toBeGreaterThan(10);
      expect(result.request_id).toBe(capturedReqId);
    });

    it('should map Python result 1:1 into stable DecisionResponseDto and delegate exactly once to adapter', async () => {
      const evaluateSpy = jest.spyOn(adapter, 'evaluate');

      const result = await service.createDecision({
        payment_id: 'pay_000001_a1',
      });

      expect(evaluateSpy).toHaveBeenCalledTimes(1);
      expect(result).toEqual({
        payment_id: 'pay_000001_a1',
        model_decision: 'RETRY_NUDGE',
        llm_decision: 'RETRY_NUDGE',
        guardrail_overridden: false,
        guardrail_reason: null,
        final_action: 'RETRY_NUDGE',
        confidence: 0.92,
        risk_level: 'low',
        reasoning: 'Customer has high recovery likelihood on retry nudge',
        decision_source: 'FOUNDRY_REASONING',
        request_id: 'req-test-uuid-123',
      });
    });

    it('should preserve guardrail_overridden and guardrail_reason verbatim when overridden', async () => {
      jest.spyOn(adapter, 'evaluate').mockResolvedValueOnce({
        ...mockPythonResult,
        guardrail_overridden: true,
        guardrail_reason: 'Max retry attempts exceeded for payment scenario',
        final_action: 'WAIT',
      });

      const result = await service.createDecision({
        payment_id: 'pay_000001_a1',
      });

      expect(result.guardrail_overridden).toBe(true);
      expect(result.guardrail_reason).toBe(
        'Max retry attempts exceeded for payment scenario',
      );
      expect(result.final_action).toBe('WAIT');
    });

    it('should emit structured JSON logs for request lifecycle with consistent request_id', async () => {
      const loggerSpy = jest.spyOn(Logger.prototype, 'log');

      const customReqId = 'req-test-uuid-123';
      const result = await service.createDecision(
        {
          payment_id: 'pay_000001_a1',
          force_recompute: true,
        },
        customReqId,
      );

      // Extract all JSON payloads logged by this request
      const jsonCalls = loggerSpy.mock.calls
        .map((call) => {
          try {
            return JSON.parse(call[0]);
          } catch {
            return null;
          }
        })
        .filter((payload) => payload && payload.request_id === customReqId);

      expect(jsonCalls.length).toBeGreaterThanOrEqual(4);

      // 1. Every event contains required contract fields
      for (const logItem of jsonCalls) {
        expect(logItem.service).toBe('nestjs');
        expect(logItem.request_id).toBe(customReqId);
        expect(typeof logItem.timestamp).toBe('string');
        expect(typeof logItem.event).toBe('string');
      }

      // 2. Events match required sequence: request_received, decision_engine_request_started, decision_engine_request_completed, decision_completed
      const events = jsonCalls.map((c) => c.event);
      expect(events).toContain('request_received');
      expect(events).toContain('decision_engine_request_started');
      expect(events).toContain('decision_engine_request_completed');
      expect(events).toContain('decision_completed');

      // 3. Verify event fields
      const received = jsonCalls.find((c) => c.event === 'request_received');
      expect(received.payment_id).toBe('pay_000001_a1');
      expect(received.force_recompute).toBe(true);

      const deStarted = jsonCalls.find((c) => c.event === 'decision_engine_request_started');
      expect(deStarted.payment_id).toBe('pay_000001_a1');

      const deCompleted = jsonCalls.find((c) => c.event === 'decision_engine_request_completed');
      expect(deCompleted.payment_id).toBe('pay_000001_a1');
      expect(typeof deCompleted.duration_ms).toBe('number');

      const completed = jsonCalls.find((c) => c.event === 'decision_completed');
      expect(completed.payment_id).toBe('pay_000001_a1');
      expect(completed.final_action).toBe('RETRY_NUDGE');
      expect(completed.guardrail_overridden).toBe(false);
      expect(typeof completed.duration_ms).toBe('number');
      expect(completed.request_id).toBe(result.request_id);
    });


    it('should propagate DecisionEngineUnavailableException (503) and attach correlated requestId', async () => {
      jest
        .spyOn(adapter, 'evaluate')
        .mockRejectedValueOnce(
          new DecisionEngineUnavailableException('ECONNREFUSED 127.0.0.1:8000'),
        );

      let thrownErr: any;
      try {
        await service.createDecision(
          { payment_id: 'pay_000001_a1' },
          'req-custom-client-error-test',
        );
      } catch (err) {
        thrownErr = err;
      }

      expect(thrownErr).toBeInstanceOf(DecisionEngineUnavailableException);
      expect(thrownErr.requestId).toBe('req-custom-client-error-test');
      expect((thrownErr.getResponse() as any).request_id).toBe(
        'req-custom-client-error-test',
      );
    });

    it('should propagate DecisionEngineTimeoutException (503) with generated requestId', async () => {
      jest
        .spyOn(adapter, 'evaluate')
        .mockRejectedValueOnce(new DecisionEngineTimeoutException());

      let thrownErr: any;
      try {
        await service.createDecision({ payment_id: 'pay_000001_a1' });
      } catch (err) {
        thrownErr = err;
      }

      expect(thrownErr).toBeInstanceOf(DecisionEngineTimeoutException);
      expect(thrownErr.requestId).toBeDefined();
      expect(typeof thrownErr.requestId).toBe('string');
      expect(thrownErr.requestId.length).toBeGreaterThan(10);
    });

    it('should propagate DecisionEngineErrorException (502) on upstream 5xx error', async () => {
      jest
        .spyOn(adapter, 'evaluate')
        .mockRejectedValueOnce(
          new DecisionEngineErrorException('Upstream HTTP 500'),
        );

      let thrownErr: any;
      try {
        await service.createDecision(
          { payment_id: 'pay_000001_a1' },
          'req-upstream-err-test',
        );
      } catch (err) {
        thrownErr = err;
      }

      expect(thrownErr).toBeInstanceOf(DecisionEngineErrorException);
      expect(thrownErr.requestId).toBe('req-upstream-err-test');
    });
  });
});
