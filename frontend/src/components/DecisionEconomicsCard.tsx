import React from 'react';
import { DollarSign, Percent, Lock, Info, TrendingUp } from 'lucide-react';
import { DecisionResponse } from '../types/api';

interface DecisionEconomicsCardProps {
  decision: DecisionResponse | null;
  amount: number;
}

export const DecisionEconomicsCard: React.FC<DecisionEconomicsCardProps> = ({ decision, amount }) => {
  if (!decision) {
    return (
      <div className="glass-panel rounded-xl p-5 border border-slate-800 text-slate-400 text-sm flex items-center justify-center min-h-[140px]">
        <div className="text-center">
          <DollarSign className="h-6 w-6 text-slate-600 mx-auto mb-1.5" />
          <p>Evaluate a payment to view decision recovery economics</p>
        </div>
      </div>
    );
  }

  // Extract reported net value from reasoning string if present (e.g. "INR 5830.31", "1979.63 INR", "INR 95.09")
  const reasoning = decision.reasoning || '';
  const netValueMatch =
    reasoning.match(/(?:net expected value|net value)\s*(?:of)?\s*(?:\()?\s*(?:INR\s*)?([\d,]+\.?\d*)\s*(?:INR)?/i) ||
    reasoning.match(/INR\s*([\d,]+\.?\d*)/i);
  const reportedNetValue = netValueMatch ? netValueMatch[1] : null;

  // Extract reported recovery probability from reasoning if present (e.g. "0.778", "79.8%", "0.134")
  const probMatch = reasoning.match(/(?:recovery probability|probability)\s*(?:of|approximately)?\s*([0-9.]+%?)/i);
  const reportedProbability = probMatch ? probMatch[1] : null;

  return (
    <div className="glass-panel rounded-xl p-5 border border-slate-800">
      <div className="flex items-center justify-between mb-3.5">
        <h3 className="text-sm font-semibold tracking-wide text-white uppercase flex items-center gap-2">
          <TrendingUp className="h-4 w-4 text-emerald-400" />
          Decision Economics
        </h3>
        <span className="text-[11px] font-medium px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700/60 flex items-center gap-1">
          <Lock className="h-3 w-3 text-slate-400" /> Server-side Causal Policy
        </span>
      </div>

      {reportedNetValue || reportedProbability ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-3.5">
          {reportedNetValue && (
            <div className="p-3 rounded-lg bg-emerald-950/30 border border-emerald-500/30">
              <div className="text-[11px] text-emerald-400/90 font-medium flex items-center gap-1 mb-0.5">
                <DollarSign className="h-3.5 w-3.5 text-emerald-400" />
                Reported Expected Net Value
              </div>
              <div className="text-xl font-extrabold text-emerald-300 font-mono">
                ₹{reportedNetValue}
              </div>
              <p className="text-[10px] text-slate-400 mt-1">
                Derived from causal treatment effect net of action intervention costs
              </p>
            </div>
          )}

          {reportedProbability && (
            <div className="p-3 rounded-lg bg-sky-950/30 border border-sky-500/30">
              <div className="text-[11px] text-sky-400/90 font-medium flex items-center gap-1 mb-0.5">
                <Percent className="h-3.5 w-3.5 text-sky-400" />
                Reported Recovery Probability
              </div>
              <div className="text-xl font-extrabold text-sky-300 font-mono">
                {reportedProbability.endsWith('%')
                  ? reportedProbability
                  : `${(parseFloat(reportedProbability) * 100).toFixed(1)}%`}
              </div>
              <p className="text-[10px] text-slate-400 mt-1">
                Predicted dynamic recovery probability under selected action
              </p>
            </div>
          )}
        </div>
      ) : (
        <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800 text-sm text-slate-300 mb-3.5 flex items-start gap-2.5">
          <Info className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
          <div>
            <p className="font-medium text-slate-200">
              Action selected from the trained causal policy using expected recovery economics.
            </p>
            <p className="text-xs text-slate-400 mt-1">
              Payment amount: <span className="text-emerald-400 font-mono font-semibold">₹{amount.toLocaleString('en-IN')}</span>. Optimal arm maximizes expected recovery value minus intervention cost.
            </p>
          </div>
        </div>
      )}

      <div className="rounded-lg bg-dark-950/60 p-3 border border-slate-800/80 text-xs text-slate-400 flex items-start gap-2">
        <Info className="h-4 w-4 text-slate-500 shrink-0 mt-0.5" />
        <p className="leading-relaxed">
          <strong className="text-slate-300">Contract Notice:</strong> Arm-by-arm net value vectors (<code className="text-slate-300">WAIT</code>, <code className="text-slate-300">RETRY</code>, <code className="text-slate-300">RETRY_NUDGE</code>, <code className="text-slate-300">ESCALATE</code>) are computed internally by the causal T-Learner and are not exposed in the public gateway contract. The card surfaces verified metrics returned in the decision payload without client-side fabrication.
        </p>
      </div>
    </div>
  );
};
