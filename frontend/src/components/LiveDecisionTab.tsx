import React, { useState } from 'react';
import {
  CreateDecisionRequest,
  DecisionResponse,
  ClientAuditRecord,
} from '../types/api';
import { evaluatePaymentDecision, ApiEvaluationResult } from '../services/api';

interface LiveDecisionTabProps {
  onAddAuditRecord: (record: ClientAuditRecord) => void;
  onLatencyUpdate?: (ms: number) => void;
  evaluating: boolean;
  setEvaluating: (val: boolean) => void;
  currentResult: DecisionResponse | null;
  setCurrentResult: (val: DecisionResponse | null) => void;
}

export const LiveDecisionTab: React.FC<LiveDecisionTabProps> = ({
  onAddAuditRecord,
  onLatencyUpdate,
  evaluating,
  setEvaluating,
  currentResult,
  setCurrentResult,
}) => {
  // Scenario state
  const [activeScenario, setActiveScenario] = useState<string>('reliable');

  // Form input state initialized with verified default
  const [paymentId, setPaymentId] = useState<string>('pay_910099_a1');
  const [amount, setAmount] = useState<number>(2500);
  const [attemptNumber, setAttemptNumber] = useState<number>(1);
  const [dynamicSuccessRate, setDynamicSuccessRate] = useState<number>(0.70);
  const [cumulativeFailures, setCumulativeFailures] = useState<number>(0);
  const [consecutiveFailedCycles, setConsecutiveFailedCycles] = useState<number>(0);
  const [notificationScore, setNotificationScore] = useState<number>(0.85);
  const [contactScore, setContactScore] = useState<number>(0.60);
  const [paymentMethod, setPaymentMethod] = useState<'upi' | 'card' | 'netbanking' | 'wallet'>('upi');
  const [failureReason, setFailureReason] = useState<string>('temporary_bank_issue');

  const [copied, setCopied] = useState<boolean>(false);
  const [apiError, setApiError] = useState<string | null>(null);

  // Apply scenario presets
  const applyPreset = (preset: 'reliable' | 'chronic' | 'guardrail') => {
    setActiveScenario(preset);
    setApiError(null);
    if (preset === 'reliable') {
      setPaymentId('pay_910099_a1');
      setAmount(2500);
      setAttemptNumber(1);
      setDynamicSuccessRate(0.70);
      setCumulativeFailures(0);
      setConsecutiveFailedCycles(0);
      setNotificationScore(0.85);
      setContactScore(0.60);
      setPaymentMethod('upi');
      setFailureReason('temporary_bank_issue');
    } else if (preset === 'chronic') {
      setPaymentId('pay_331002_a2');
      setAmount(4200);
      setAttemptNumber(2);
      setDynamicSuccessRate(0.25);
      setCumulativeFailures(2);
      setConsecutiveFailedCycles(1);
      setNotificationScore(0.35);
      setContactScore(0.30);
      setPaymentMethod('card');
      setFailureReason('bank_decline');
    } else if (preset === 'guardrail') {
      setPaymentId('pay_999999_a6');
      setAmount(1499);
      setAttemptNumber(3);
      setDynamicSuccessRate(0.65);
      setCumulativeFailures(3);
      setConsecutiveFailedCycles(3);
      setNotificationScore(0.80);
      setContactScore(0.50);
      setPaymentMethod('card');
      setFailureReason('insufficient_funds');
    }
  };

  const handleCopyPaymentId = () => {
    navigator.clipboard.writeText(paymentId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Main evaluation calling the real deployed API
  const handleEvaluate = async (customPayload?: CreateDecisionRequest) => {
    setEvaluating(true);
    setApiError(null);

    const payload: CreateDecisionRequest = customPayload || {
      payment_id: paymentId,
      features: {
        amount,
        attempt_number: attemptNumber,
        dynamic_success_rate: dynamicSuccessRate,
        cumulative_failures: cumulativeFailures,
        consecutive_failed_cycles: consecutiveFailedCycles,
        consecutive_failures: consecutiveFailedCycles >= 3 ? 0 : cumulativeFailures,
        notification_engagement_score: notificationScore,
        contact_response_score: contactScore,
        payment_method: paymentMethod,
        failure_reason: failureReason as any,
      },
      force_recompute: false,
    };

    const res: ApiEvaluationResult = await evaluatePaymentDecision(payload);
    setEvaluating(false);

    if (res.latencyMs && onLatencyUpdate) {
      onLatencyUpdate(res.latencyMs);
    }

    if (res.data) {
      setCurrentResult(res.data);
      onAddAuditRecord({
        id: crypto.randomUUID(),
        timestamp: new Date().toLocaleTimeString(),
        payment_id: res.data.payment_id,
        request_id: res.data.request_id,
        final_action: res.data.final_action,
        decision_source: res.data.decision_source,
        confidence: res.data.confidence,
        guardrail_overridden: res.data.guardrail_overridden,
        guardrail_reason: res.data.guardrail_reason,
        latency_ms: res.latencyMs,
        status: 'success',
        amount: payload.features?.amount,
        payment_method: payload.features?.payment_method,
      });
    } else if (res.error) {
      setApiError(res.error.message || 'API request failed');
    }
  };

  const handleRepeatSameRequest = () => {
    handleEvaluate();
  };

  const handleChangeAmountTo7500 = () => {
    setAmount(7500);
    handleEvaluate({
      payment_id: paymentId,
      features: {
        amount: 7500,
        attempt_number: attemptNumber,
        dynamic_success_rate: dynamicSuccessRate,
        cumulative_failures: cumulativeFailures,
        consecutive_failed_cycles: consecutiveFailedCycles,
        consecutive_failures: consecutiveFailedCycles >= 3 ? 0 : cumulativeFailures,
        notification_engagement_score: notificationScore,
        contact_response_score: contactScore,
        payment_method: paymentMethod,
        failure_reason: failureReason as any,
      },
      force_recompute: false,
    });
  };

  // Resolved result presentation
  const isOverridden = currentResult?.guardrail_overridden;
  const isCache = currentResult?.decision_source === 'cache';
  const confidencePct = Math.round((currentResult?.confidence ?? 0.95) * 100);

  // Extract expected net value from LLM reasoning if present (real server returned value)
  const reasoningText = currentResult?.reasoning || '';
  const netMatch = reasoningText.match(/(?:net expected value|net value)\s*(?:of)?\s*(?:\()?\s*(?:INR\s*)?([\d,]+\.?\d*)\s*(?:INR)?/i) || reasoningText.match(/INR\s*([\d,]+\.?\d*)/i);
  const displayedNetValue = netMatch ? `₹${parseFloat(netMatch[1].replace(/,/g, '')).toLocaleString('en-IN')}` : currentResult ? '—' : '₹2,018';

  return (
    <div className="flex flex-col w-full text-on-surface">
      {/* ── Page Header & Architecture Line ──────────────────────────────── */}
      <div className="flex flex-col mb-space-lg">
        <h1 className="font-display-lg text-[22px] sm:text-[28px] text-on-surface font-semibold tracking-tight">
          Recovery Intelligence Engine
        </h1>
        <p className="font-body-md text-on-surface-variant mt-1 text-[13px] sm:text-[14px]">
          Causal recovery decisions for failed payments
        </p>
        <div className="flex items-center gap-space-xs font-mono-code-sm text-[11px] sm:text-[12px] text-on-surface-variant/80 mt-2 flex-wrap">
          <span className="text-secondary font-medium">Observable features</span>
          <span>→</span>
          <span className="text-on-surface">causal policy</span>
          <span>→</span>
          <span className="text-primary font-medium">recovery economics</span>
          <span>→</span>
          <span className="text-primary font-semibold">guarded action</span>
        </div>
      </div>

      {/* ── Compact Pipeline Visualizer ──────────────────────────────────── */}
      <div className="bg-surface-container-low rounded-lg p-space-sm mb-space-xl border border-surface-container-high/60 overflow-x-auto">
        <div className="flex items-center min-w-[700px] justify-between text-[11px] font-mono-code-sm">
          <div className="flex items-center gap-space-xs flex-1 px-space-sm py-space-xs bg-surface-container rounded">
            <span className="w-1.5 h-1.5 rounded-full bg-primary"></span>
            <span className="text-on-surface-variant font-medium">1. Context</span>
            <span className="text-primary ml-auto">Features Checked</span>
          </div>
          <span className="px-space-xs text-outline">→</span>
          <div className="flex items-center gap-space-xs flex-1 px-space-sm py-space-xs bg-surface-container rounded">
            <span className="w-1.5 h-1.5 rounded-full bg-primary"></span>
            <span className="text-on-surface-variant font-medium">2. Estimation</span>
            <span className="text-primary ml-auto">Causal Loss</span>
          </div>
          <span className="px-space-xs text-outline">→</span>
          <div className="flex items-center gap-space-xs flex-1 px-space-sm py-space-xs bg-surface-container rounded">
            <span className="w-1.5 h-1.5 rounded-full bg-secondary"></span>
            <span className="text-on-surface-variant font-medium">3. Reasoning</span>
            <span className="text-secondary ml-auto">LLM Synthesized</span>
          </div>
          <span className="px-space-xs text-outline">→</span>
          <div className={`flex items-center gap-space-xs flex-1 px-space-sm py-space-xs rounded ${isOverridden ? 'bg-error-container/30' : 'bg-surface-container'}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${isOverridden ? 'bg-error' : 'bg-primary'}`}></span>
            <span className={`font-medium ${isOverridden ? 'text-error' : 'text-on-surface-variant'}`}>4. Guardrails</span>
            <span className={`ml-auto ${isOverridden ? 'text-error font-semibold' : 'text-primary'}`}>
              {isOverridden ? 'Overridden' : 'Passed'}
            </span>
          </div>
          <span className="px-space-xs text-outline">→</span>
          <div className={`flex items-center gap-space-xs flex-1 px-space-sm py-space-xs rounded ${isOverridden ? 'bg-error-container/40' : 'bg-primary/10'}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${isOverridden ? 'bg-error' : 'bg-primary'}`}></span>
            <span className={`font-semibold ${isOverridden ? 'text-error' : 'text-primary'}`}>5. Final Action</span>
            <span className="text-on-surface ml-auto font-medium">
              {evaluating ? 'Evaluating...' : currentResult?.final_action || 'RETRY_NUDGE'}
            </span>
          </div>
        </div>
      </div>

      {/* ── Main Workstation: Clean 2-Column Split ────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-space-xl items-start">
        {/* LEFT COLUMN: Payment Scenario */}
        <div className="lg:col-span-5 flex flex-col gap-space-lg">
          <div className="bg-surface-container-low rounded-xl p-space-lg shadow-sm border border-surface-container-high/60 flex flex-col gap-space-md">
            <div className="flex items-center justify-between">
              <h2 className="font-headline-sm text-headline-sm text-on-surface font-semibold">
                Payment Scenario
              </h2>
              <span className="font-label-caps text-label-caps text-on-surface-variant uppercase">
                Input Features
              </span>
            </div>

            {/* Presets */}
            <div className="flex items-center gap-1.5 sm:gap-space-xs flex-wrap">
              <button
                type="button"
                onClick={() => applyPreset('reliable')}
                className={`px-space-sm py-1 rounded-full font-mono-code-sm text-[11px] transition-all cursor-pointer ${
                  activeScenario === 'reliable'
                    ? 'bg-primary text-on-primary font-medium shadow-sm'
                    : 'bg-surface-container text-on-surface-variant hover:text-on-surface'
                }`}
              >
                Reliable Customer
              </button>
              <button
                type="button"
                onClick={() => applyPreset('chronic')}
                className={`px-space-sm py-1 rounded-full font-mono-code-sm text-[11px] transition-all cursor-pointer ${
                  activeScenario === 'chronic'
                    ? 'bg-primary text-on-primary font-medium shadow-sm'
                    : 'bg-surface-container text-on-surface-variant hover:text-on-surface'
                }`}
              >
                Chronic Failure
              </button>
              <button
                type="button"
                onClick={() => applyPreset('guardrail')}
                className={`px-space-sm py-1 rounded-full font-mono-code-sm text-[11px] transition-all cursor-pointer ${
                  activeScenario === 'guardrail'
                    ? 'bg-primary text-on-primary font-medium shadow-sm'
                    : 'bg-surface-container text-on-surface-variant hover:text-on-surface'
                }`}
              >
                Guardrail Override
              </button>
            </div>

            {/* Form Fields */}
            <div className="space-y-space-sm">
              {/* Payment ID & Amount */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-space-sm">
                <div className="flex flex-col gap-1">
                  <label className="font-label-caps text-[11px] text-on-surface-variant uppercase">Payment ID</label>
                  <div className="flex items-center justify-between bg-surface px-space-sm py-space-xs rounded border border-surface-container-high/40">
                    <span className="font-mono-code-sm text-[12px] text-on-surface truncate">{paymentId}</span>
                    <button
                      type="button"
                      onClick={handleCopyPaymentId}
                      className="text-on-surface-variant hover:text-primary transition-colors cursor-pointer"
                      title="Copy"
                    >
                      <span className="material-symbols-outlined text-[15px]">
                        {copied ? 'check' : 'content_copy'}
                      </span>
                    </button>
                  </div>
                </div>
                <div className="flex flex-col gap-1">
                  <label className="font-label-caps text-[11px] text-on-surface-variant uppercase">Amount (INR)</label>
                  <div className="flex items-center bg-surface px-space-sm py-space-xs rounded border border-surface-container-high/40">
                    <span className="font-mono-code-sm text-[12px] text-on-surface-variant mr-1">₹</span>
                    <input
                      type="number"
                      value={amount}
                      onChange={(e) => setAmount(parseFloat(e.target.value) || 0)}
                      className="w-full bg-transparent font-mono-code-sm text-[12px] text-on-surface focus:outline-none"
                    />
                  </div>
                </div>
              </div>

              {/* Attempt Number & Dynamic Success Rate */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-space-sm">
                <div className="flex flex-col gap-1">
                  <label className="font-label-caps text-[11px] text-on-surface-variant uppercase">Attempt Number</label>
                  <input
                    type="number"
                    min="1"
                    max="6"
                    value={attemptNumber}
                    onChange={(e) => setAttemptNumber(parseInt(e.target.value) || 1)}
                    className="bg-surface px-space-sm py-space-xs rounded border border-surface-container-high/40 font-mono-code-sm text-[12px] text-on-surface focus:outline-none"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between items-center">
                    <label className="font-label-caps text-[11px] text-on-surface-variant uppercase">Success Rate</label>
                    <span className="font-mono-code-sm text-[11px] text-primary">{Math.round(dynamicSuccessRate * 100)}%</span>
                  </div>
                  <div className="flex items-center h-8 px-space-xs bg-surface rounded border border-surface-container-high/40">
                    <input
                      type="range"
                      min="0.05"
                      max="0.95"
                      step="0.05"
                      value={dynamicSuccessRate}
                      onChange={(e) => setDynamicSuccessRate(parseFloat(e.target.value))}
                      className="w-full accent-primary cursor-pointer"
                    />
                  </div>
                </div>
              </div>

              {/* Cumulative Failures & Consecutive Failed Cycles */}
              <div className="grid grid-cols-2 gap-space-sm">
                <div className="flex flex-col gap-1">
                  <label className="font-label-caps text-[11px] text-on-surface-variant uppercase">Cumulative Failures</label>
                  <input
                    type="number"
                    min="0"
                    max="10"
                    value={cumulativeFailures}
                    onChange={(e) => setCumulativeFailures(parseInt(e.target.value) || 0)}
                    className="bg-surface px-space-sm py-space-xs rounded border border-surface-container-high/40 font-mono-code-sm text-[12px] text-on-surface focus:outline-none"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <label className="font-label-caps text-[11px] text-on-surface-variant uppercase">Consecutive Failed Cycles</label>
                  <input
                    type="number"
                    min="0"
                    max="6"
                    value={consecutiveFailedCycles}
                    onChange={(e) => setConsecutiveFailedCycles(parseInt(e.target.value) || 0)}
                    className={`bg-surface px-space-sm py-space-xs rounded border font-mono-code-sm text-[12px] focus:outline-none ${consecutiveFailedCycles >= 3 ? 'border-error/60 text-error' : 'border-surface-container-high/40 text-on-surface'}`}
                  />
                </div>
              </div>

              {/* Notification & Contact Scores */}
              <div className="grid grid-cols-2 gap-space-sm">
                <div className="flex flex-col gap-1">
                  <label className="font-label-caps text-[11px] text-on-surface-variant uppercase">Notification Score</label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    value={notificationScore}
                    onChange={(e) => setNotificationScore(parseFloat(e.target.value) || 0)}
                    className="bg-surface px-space-sm py-space-xs rounded border border-surface-container-high/40 font-mono-code-sm text-[12px] text-on-surface focus:outline-none"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <label className="font-label-caps text-[11px] text-on-surface-variant uppercase">Contact Score</label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    value={contactScore}
                    onChange={(e) => setContactScore(parseFloat(e.target.value) || 0)}
                    className="bg-surface px-space-sm py-space-xs rounded border border-surface-container-high/40 font-mono-code-sm text-[12px] text-on-surface focus:outline-none"
                  />
                </div>
              </div>

              {/* Payment Method */}
              <div className="flex flex-col gap-1">
                <label className="font-label-caps text-[11px] text-on-surface-variant uppercase">Payment Method</label>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-1 bg-surface p-1 rounded border border-surface-container-high/40">
                  {(['upi', 'card', 'netbanking', 'wallet'] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => setPaymentMethod(m)}
                      className={`py-1.5 sm:py-1 text-[11px] font-label-caps uppercase rounded transition-colors cursor-pointer text-center ${
                        paymentMethod === m
                          ? 'bg-surface-container-highest text-on-surface font-semibold shadow-sm'
                          : 'text-on-surface-variant hover:text-on-surface'
                      }`}
                    >
                      {m === 'wallet' ? 'eNACH' : m}
                    </button>
                  ))}
                </div>
              </div>

              {/* Failure Reason */}
              <div className="flex flex-col gap-1">
                <label className="font-label-caps text-[11px] text-on-surface-variant uppercase">Failure Reason</label>
                <select
                  value={failureReason}
                  onChange={(e) => setFailureReason(e.target.value)}
                  className="bg-surface px-space-sm py-space-xs rounded border border-surface-container-high/40 font-mono-code-sm text-[12px] text-on-surface focus:outline-none cursor-pointer"
                >
                  <option value="temporary_bank_issue">temporary_bank_issue</option>
                  <option value="insufficient_funds">insufficient_funds</option>
                  <option value="bank_decline">bank_decline</option>
                  <option value="network_error">network_error</option>
                  <option value="expired_card">expired_card</option>
                </select>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col gap-space-xs pt-space-xs">
              <button
                type="button"
                disabled={evaluating}
                onClick={() => handleEvaluate()}
                className="w-full py-space-sm bg-primary hover:bg-primary-container text-on-primary font-headline-sm text-headline-sm rounded flex items-center justify-center gap-space-xs shadow-md transition-all cursor-pointer disabled:opacity-50 font-semibold"
              >
                <span className={`material-symbols-outlined text-[18px] ${evaluating ? 'animate-spin' : ''}`}>
                  {evaluating ? 'sync' : 'bolt'}
                </span>
                <span>{evaluating ? 'Evaluating Causal Policy...' : 'Evaluate Payment'}</span>
              </button>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-space-xs">
                <button
                  type="button"
                  disabled={evaluating}
                  onClick={handleRepeatSameRequest}
                  className="py-1.5 px-space-sm bg-surface-container hover:bg-surface-container-high text-on-surface font-label-caps text-[11px] uppercase rounded text-center transition-colors cursor-pointer border border-surface-container-high/40"
                >
                  Repeat Same Request
                </button>
                <button
                  type="button"
                  disabled={evaluating}
                  onClick={handleChangeAmountTo7500}
                  className="py-1.5 px-space-sm bg-surface-container hover:bg-surface-container-high text-on-surface font-label-caps text-[11px] uppercase rounded text-center transition-colors cursor-pointer border border-surface-container-high/40"
                >
                  Change Amount → ₹7,500
                </button>
              </div>

              {apiError && (
                <div className="p-space-xs rounded bg-error-container/30 border border-error/40 text-error text-[12px] mt-1">
                  {apiError}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Decision Result & Why This Decision? */}
        <div className="lg:col-span-7 flex flex-col gap-space-md">
          {/* Main Decision Result Card (Visual Focal Point) */}
          <div className="bg-surface-container-low rounded-xl p-space-md sm:p-space-lg shadow-sm border border-surface-container-high/60 flex flex-col gap-space-md">
            {/* Top Row: Request ID & Meta */}
            <div className="flex items-center justify-between flex-wrap gap-space-xs">
              <div className="flex items-center gap-space-xs font-mono-code-sm text-[11px] sm:text-[12px] text-on-surface-variant">
                <span>REQ:</span>
                <span className="text-secondary font-medium truncate max-w-[160px] sm:max-w-[220px]">
                  {currentResult?.request_id || 'req_causal_8829fba1'}
                </span>
              </div>
              <div className="flex items-center gap-space-xs flex-wrap">
                <span className="px-space-xs py-0.5 rounded font-mono-code-sm text-[10px] sm:text-[11px] bg-surface-container text-on-surface-variant">
                  Source: {currentResult?.decision_source || (isCache ? 'cache' : 'llm')}
                </span>
                <span className={`px-space-xs py-0.5 rounded font-mono-code-sm text-[10px] sm:text-[11px] font-medium ${isOverridden ? 'bg-error-container/40 text-error' : 'bg-primary/20 text-primary'}`}>
                  Guardrails: {isOverridden ? 'OVERRIDDEN' : 'PASSED'}
                </span>
              </div>
            </div>

            {/* Prominent Action Banner */}
            <div className="bg-surface p-space-md rounded-xl border border-surface-container-high/40 flex flex-col sm:flex-row sm:items-center justify-between gap-space-sm">
              <div>
                <span className="font-label-caps text-[10px] sm:text-[11px] text-on-surface-variant uppercase tracking-wider block mb-1">
                  Recommended Recovery Action
                </span>
                <div className={`font-display-lg text-[20px] sm:text-[26px] font-bold tracking-tight flex items-center gap-space-xs flex-wrap ${isOverridden ? 'text-error' : 'text-primary'}`}>
                  <span>FINAL ACTION: {currentResult?.final_action || (isOverridden ? 'STOP' : 'RETRY_NUDGE')}</span>
                  <span className="material-symbols-outlined text-[22px] sm:text-[26px]">
                    {isOverridden ? 'cancel' : 'check_circle'}
                  </span>
                </div>
              </div>
              <div className="flex sm:flex-col items-center sm:items-end justify-between sm:justify-center pt-2 sm:pt-0 pl-0 sm:pl-space-md border-t sm:border-t-0 sm:border-l border-surface-container-high">
                <span className="font-mono-metric-lg text-[18px] sm:text-[22px] font-bold text-on-surface">
                  {confidencePct}%
                </span>
                <span className="font-label-caps text-[10px] text-on-surface-variant uppercase">
                  Confidence
                </span>
              </div>
            </div>

            {/* 3 Real Metric Boxes */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-space-sm">
              <div className="bg-surface p-space-sm rounded border border-surface-container-high/40">
                <span className="font-label-caps text-[10px] text-on-surface-variant uppercase block">Calculated Risk</span>
                <span className={`font-mono-code-sm text-[13px] font-semibold mt-1 block uppercase ${currentResult?.risk_level === 'high' ? 'text-error' : currentResult?.risk_level === 'low' ? 'text-primary' : 'text-secondary'}`}>
                  {currentResult?.risk_level || 'Medium'}
                </span>
              </div>
              <div className="bg-surface p-space-sm rounded border border-surface-container-high/40">
                <span className="font-label-caps text-[10px] text-on-surface-variant uppercase block">Expected Net Value</span>
                <span className="font-mono-metric-md text-[13px] text-primary font-semibold mt-1 block">
                  {displayedNetValue}
                </span>
              </div>
              <div className="bg-surface p-space-sm rounded border border-surface-container-high/40">
                <span className="font-label-caps text-[10px] text-on-surface-variant uppercase block">Decision Source</span>
                <span className="font-mono-code-sm text-[13px] text-on-surface mt-1 block capitalize">
                  {currentResult?.decision_source || 'LLM (Governed)'}
                </span>
              </div>
            </div>

            {/* Model Decision vs LLM Decision vs Guardrail Status */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-space-sm pt-space-xs border-t border-surface-container-high/40">
              <div className="bg-surface p-space-sm rounded border border-surface-container-high/40 flex flex-col justify-between">
                <span className="font-label-caps text-[10px] text-on-surface-variant uppercase">Causal Model Loss</span>
                <span className="font-mono-code-sm text-[13px] text-on-surface font-semibold mt-1">
                  {currentResult?.model_decision || 'RETRY'}
                </span>
              </div>
              <div className="bg-surface p-space-sm rounded border border-surface-container-high/40 flex flex-col justify-between">
                <span className="font-label-caps text-[10px] text-secondary uppercase">LLM Recommendation</span>
                <span className="font-mono-code-sm text-[13px] text-secondary font-semibold mt-1">
                  {currentResult?.llm_decision || 'RETRY_NUDGE'}
                </span>
              </div>
              <div className="bg-surface p-space-sm rounded border border-surface-container-high/40 flex flex-col justify-between sm:col-span-2">
                <div className="flex items-center justify-between">
                  <span className="font-label-caps text-[10px] text-on-surface-variant uppercase">Guardrail Verification</span>
                  <span className={`font-mono-code-sm text-[11px] font-semibold ${isOverridden ? 'text-error' : 'text-primary'}`}>
                    {isOverridden ? 'OVERRIDDEN' : 'PASSED'}
                  </span>
                </div>
                <p className="font-body-sm text-[12px] text-on-surface-variant mt-1">
                  {currentResult?.guardrail_reason ||
                    'Proposed action passed all deterministic guardrails (consecutive cycles and failure caps respected).'}
                </p>
              </div>
            </div>
          </div>

          {/* Why This Decision? Narrative */}
          <div className="bg-surface-container-low rounded-xl p-space-md shadow-sm border border-surface-container-high/60 flex flex-col gap-space-xs">
            <div className="flex items-center justify-between">
              <h3 className="font-headline-sm text-[14px] text-on-surface font-semibold flex items-center gap-space-xs">
                <span className="material-symbols-outlined text-secondary text-[16px]">psychology</span>
                <span>Why this decision?</span>
              </h3>
              <span className="font-mono-code-sm text-[11px] text-on-surface-variant">
                Causal Synthesis
              </span>
            </div>
            <p className="font-body-sm text-[12px] text-on-surface leading-relaxed mt-1">
              {currentResult?.reasoning ||
                `Among the permitted actions, RETRY_NUDGE yields the highest expected net value (${displayedNetValue}) balancing recovery probability against intervention cost. Customer has high historical engagement on UPI AutoPay with transient failure code, making proactive notification optimal prior to next execution.`}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
