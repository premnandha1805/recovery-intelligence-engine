import React from 'react';
import { DecisionResponse } from '../types/api';

interface PipelineVisualizationProps {
  evaluating: boolean;
  hasResult: boolean;
  decision: DecisionResponse | null;
}

export const PipelineVisualization: React.FC<PipelineVisualizationProps> = ({
  evaluating,
  hasResult,
  decision,
}) => {
  const isOverridden = decision?.guardrail_overridden;
  const finalAction = decision?.final_action || 'RETRY_NUDGE';

  return (
    <div className="bg-surface-container-low rounded-lg p-space-sm shadow-md overflow-x-auto">
      <div className="flex items-center min-w-[860px] justify-between gap-space-xs">
        {/* Stage 1 */}
        <div className="flex items-center gap-space-xs flex-1 bg-surface-container px-space-sm py-space-xs rounded">
          <span className="material-symbols-outlined text-primary text-[18px]">data_object</span>
          <div className="flex flex-col min-w-0">
            <span className="font-label-caps text-label-caps text-on-surface-variant uppercase truncate">
              1. Features
            </span>
            <span className="font-mono-code-sm text-mono-code-sm text-primary truncate">
              {evaluating ? 'Loading...' : 'Active • Checked'}
            </span>
          </div>
        </div>

        <span className="text-outline-variant font-mono-code-sm text-mono-code-sm px-space-2xs">→</span>

        {/* Stage 2 */}
        <div className="flex items-center gap-space-xs flex-1 bg-surface-container px-space-sm py-space-xs rounded">
          <span className="material-symbols-outlined text-primary text-[18px]">schema</span>
          <div className="flex flex-col min-w-0">
            <span className="font-label-caps text-label-caps text-on-surface-variant uppercase truncate">
              2. Causal Policy
            </span>
            <span className="font-mono-code-sm text-mono-code-sm text-primary truncate">
              {evaluating ? 'Estimating...' : 'Active • Checked'}
            </span>
          </div>
        </div>

        <span className="text-outline-variant font-mono-code-sm text-mono-code-sm px-space-2xs">→</span>

        {/* Stage 3 */}
        <div className="flex items-center gap-space-xs flex-1 bg-surface-container px-space-sm py-space-xs rounded">
          <span className="material-symbols-outlined text-primary text-[18px]">trending_up</span>
          <div className="flex flex-col min-w-0">
            <span className="font-label-caps text-label-caps text-on-surface-variant uppercase truncate">
              3. Expected Net
            </span>
            <span className="font-mono-code-sm text-mono-code-sm text-primary truncate">
              {evaluating ? 'Computing...' : 'Active • ₹4,912'}
            </span>
          </div>
        </div>

        <span className="text-outline-variant font-mono-code-sm text-mono-code-sm px-space-2xs">→</span>

        {/* Stage 4 */}
        <div className="flex items-center gap-space-xs flex-1 bg-surface-container px-space-sm py-space-xs rounded">
          <span className="material-symbols-outlined text-secondary text-[18px]">auto_awesome</span>
          <div className="flex flex-col min-w-0">
            <span className="font-label-caps text-label-caps text-on-surface-variant uppercase truncate">
              4. LLM Reasoner
            </span>
            <span className="font-mono-code-sm text-mono-code-sm text-secondary truncate">
              {evaluating
                ? 'Synthesizing...'
                : decision?.decision_source === 'fallback_no_llm'
                ? 'Model Fallback'
                : 'Active • Synthesized'}
            </span>
          </div>
        </div>

        <span className="text-outline-variant font-mono-code-sm text-mono-code-sm px-space-2xs">→</span>

        {/* Stage 5 */}
        <div
          className={`flex items-center gap-space-xs flex-1 px-space-sm py-space-xs rounded ${
            isOverridden ? 'bg-error-container/40' : 'bg-surface-container'
          }`}
        >
          <span
            className={`material-symbols-outlined text-[18px] ${
              isOverridden ? 'text-error' : 'text-primary'
            }`}
          >
            {isOverridden ? 'gpp_bad' : 'verified_user'}
          </span>
          <div className="flex flex-col min-w-0">
            <span
              className={`font-label-caps text-label-caps uppercase truncate ${
                isOverridden ? 'text-error' : 'text-on-surface-variant'
              }`}
            >
              5. Guardrails
            </span>
            <span
              className={`font-mono-code-sm text-mono-code-sm truncate ${
                isOverridden ? 'text-error font-medium' : 'text-primary'
              }`}
            >
              {isOverridden ? 'Overridden (Rule #2)' : 'Passed (Rule #14)'}
            </span>
          </div>
        </div>

        <span className="text-outline-variant font-mono-code-sm text-mono-code-sm px-space-2xs">→</span>

        {/* Stage 6 */}
        <div
          className={`flex items-center gap-space-xs flex-1 px-space-sm py-space-xs rounded ${
            isOverridden ? 'bg-error-container/30' : 'bg-primary/10'
          }`}
        >
          <span
            className={`material-symbols-outlined text-[18px] ${
              isOverridden ? 'text-error' : 'text-primary'
            }`}
          >
            bolt
          </span>
          <div className="flex flex-col min-w-0">
            <span
              className={`font-label-caps text-label-caps uppercase truncate font-semibold ${
                isOverridden ? 'text-error' : 'text-primary'
              }`}
            >
              6. Action
            </span>
            <span className="font-mono-code-sm text-mono-code-sm text-on-surface truncate">
              {evaluating ? 'Pending...' : finalAction}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
