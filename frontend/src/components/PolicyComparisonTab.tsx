import React from 'react';

export const PolicyComparisonTab: React.FC = () => {
  // Exact data from ml/evaluation/rule_based_v2_comparison_summary.csv (Seed 777 holdout)
  const policies = [
    {
      name: 'CausalUpliftPolicy',
      tag: 'RECOMMENDED',
      netFormatted: '₹16.49M',
      netFull: '₹16,486,990.85',
      gross: '₹16,740,795.85',
      fees: '₹253,805.00',
      uplift: '+7.33%',
      widthPct: 100,
      isWinner: true,
    },
    {
      name: 'AlwaysRetryPolicy',
      tag: 'BASELINE',
      netFormatted: '₹15.36M',
      netFull: '₹15,360,307.59',
      gross: '₹15,508,902.59',
      fees: '₹148,595.00',
      uplift: 'Baseline',
      widthPct: 93,
      isWinner: false,
    },
    {
      name: 'RuleBasedPolicy (v1)',
      tag: 'Code Heuristic',
      netFormatted: '₹14.43M',
      netFull: '₹14,433,756.53',
      gross: '₹15,988,721.53',
      fees: '₹1,554,965.00',
      uplift: '-6.03%',
      widthPct: 87,
      isWinner: false,
    },
    {
      name: 'AlwaysNudgePolicy',
      tag: 'Proactive Alert',
      netFormatted: '₹13.84M',
      netFull: '₹13,842,651.32',
      gross: '₹14,288,436.32',
      fees: '₹445,785.00',
      uplift: '-9.88%',
      widthPct: 84,
      isWinner: false,
    },
    {
      name: 'RuleBasedPolicyV2 (v2)',
      tag: 'Time Decay',
      netFormatted: '₹12.88M',
      netFull: '₹12,879,882.01',
      gross: '₹14,123,362.01',
      fees: '₹1,243,480.00',
      uplift: '-16.15%',
      widthPct: 78,
      isWinner: false,
    },
    {
      name: 'WaitPolicy',
      tag: 'Null Action',
      netFormatted: '₹8.70M',
      netFull: '₹8,703,500.25',
      gross: '₹8,703,500.25',
      fees: '₹0.00',
      uplift: '-43.34%',
      widthPct: 53,
      isWinner: false,
    },
  ];

  return (
    <div className="flex flex-col w-full text-on-surface">
      {/* ── Page Header ──────────────────────────────────────────────────── */}
      <div className="flex flex-col mb-space-lg">
        <h1 className="font-display-lg text-[22px] sm:text-[28px] text-on-surface font-semibold tracking-tight">
          Policy Comparison
        </h1>
        <p className="font-body-md text-on-surface-variant mt-1 text-[13px] sm:text-[14px]">
          Offline evaluation of recovery policies
        </p>
      </div>

      {/* ── Small KPI Row ────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-space-md mb-space-xl">
        <div className="bg-surface-container-low rounded-xl p-space-md shadow-sm border border-surface-container-high/60 flex flex-col justify-between">
          <span className="font-label-caps text-[11px] text-on-surface-variant uppercase">Best Policy</span>
          <div className="mt-2">
            <div className="font-mono-metric-lg text-[18px] text-primary font-semibold truncate">
              CausalUpliftPolicy
            </div>
            <div className="font-body-sm text-[12px] text-on-surface-variant mt-0.5">
              Multi-Armed Causal Bandit (#1 Ranked)
            </div>
          </div>
        </div>

        <div className="bg-surface-container-low rounded-xl p-space-md shadow-sm border border-surface-container-high/60 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="font-label-caps text-[11px] text-on-surface-variant uppercase">Net Value</span>
            <span className="font-mono-code-sm text-[11px] text-primary font-medium">+7.33% vs Baseline</span>
          </div>
          <div className="mt-2">
            <div className="font-mono-metric-lg text-[18px] text-on-surface font-semibold">
              ₹16.49M
            </div>
            <div className="font-body-sm text-[12px] text-on-surface-variant mt-0.5">
              Net recovered post action costs
            </div>
          </div>
        </div>

        <div className="bg-surface-container-low rounded-xl p-space-md shadow-sm border border-surface-container-high/60 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="font-label-caps text-[11px] text-on-surface-variant uppercase">Recovery Rate</span>
            <span className="font-mono-code-sm text-[11px] text-secondary font-medium">+2.14% over Retry</span>
          </div>
          <div className="mt-2">
            <div className="font-mono-metric-lg text-[18px] text-on-surface font-semibold">
              27.63%
            </div>
            <div className="font-body-sm text-[12px] text-on-surface-variant mt-0.5">
              Top cohort recovery yield
            </div>
          </div>
        </div>

        <div className="bg-surface-container-low rounded-xl p-space-md shadow-sm border border-surface-container-high/60 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="font-label-caps text-[11px] text-on-surface-variant uppercase">Escalation Rate</span>
            <span className="font-mono-code-sm text-[11px] text-primary font-medium">Lowest Churn</span>
          </div>
          <div className="mt-2">
            <div className="font-mono-metric-lg text-[18px] text-on-surface font-semibold">
              0.13%
            </div>
            <div className="font-body-sm text-[12px] text-on-surface-variant mt-0.5">
              Low customer contact friction
            </div>
          </div>
        </div>
      </div>

      {/* ── Main Visualization: Net Value by Policy ──────────────────────── */}
      <div className="bg-surface-container-low rounded-xl p-space-md sm:p-space-lg shadow-sm border border-surface-container-high/60 flex flex-col gap-space-md mb-space-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 pb-space-xs border-b border-surface-container-high/40">
          <div>
            <h2 className="font-headline-sm text-[16px] text-on-surface font-semibold">
              Net Value by Policy
            </h2>
            <p className="font-body-sm text-[12px] text-on-surface-variant">
              Expected net revenue recovered across candidate policies (Seed 777 Holdout, 29,719 payments)
            </p>
          </div>
          <span className="font-mono-code-sm text-[11px] text-on-surface-variant bg-surface px-space-sm py-1 rounded border border-surface-container-high/40 self-start sm:self-auto">
            Unit: INR Millions (₹)
          </span>
        </div>

        {/* Clean Comparative Bars */}
        <div className="space-y-space-sm my-space-xs">
          {policies.map((p) => (
            <div key={p.name} className="flex flex-col gap-1">
              <div className="flex flex-col xs:flex-row xs:items-center justify-between font-mono-code-sm text-[12px] gap-0.5 xs:gap-0">
                <span className="font-medium text-on-surface flex items-center gap-space-xs flex-wrap">
                  <span>{p.name}</span>
                  {p.isWinner ? (
                    <span className="px-1.5 py-0.5 rounded bg-primary text-on-primary font-semibold text-[10px] uppercase">
                      Recommended
                    </span>
                  ) : (
                    <span className="text-on-surface-variant text-[11px]">({p.tag})</span>
                  )}
                </span>
                <span className={`font-semibold ${p.isWinner ? 'text-primary' : 'text-on-surface'}`}>
                  {p.netFormatted}
                </span>
              </div>
              <div className="w-full h-6 bg-surface-container rounded overflow-hidden flex">
                <div
                  className={`h-full rounded transition-all duration-300 ${
                    p.isWinner ? 'bg-primary' : 'bg-surface-bright'
                  }`}
                  style={{ width: `${p.widthPct}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>

        {/* Supporting Comparison Table */}
        <div className="pt-space-md mt-space-sm border-t border-surface-container-high/40 overflow-x-auto">
          <div className="font-label-caps text-[11px] text-on-surface-variant uppercase mb-space-xs font-semibold">
            Evaluation Metrics Breakdown
          </div>
          <table className="w-full text-left font-mono-code-sm text-[12px] min-w-[580px]">
            <thead>
              <tr className="text-on-surface-variant font-label-caps text-[11px] uppercase border-b border-surface-container-high">
                <th className="py-space-xs px-space-sm font-semibold">Policy Name</th>
                <th className="py-space-xs px-space-sm font-semibold text-right">Net Recovery</th>
                <th className="py-space-xs px-space-sm font-semibold text-right">Gross Yield</th>
                <th className="py-space-xs px-space-sm font-semibold text-right">Routing Fees</th>
                <th className="py-space-xs px-space-sm font-semibold text-right">Uplift vs Baseline</th>
              </tr>
            </thead>
            <tbody>
              {policies.map((p) => (
                <tr
                  key={p.name}
                  className={`border-b border-surface-container-high/30 transition-colors ${
                    p.isWinner ? 'bg-surface-container/40' : 'hover:bg-surface-container/20'
                  }`}
                >
                  <td className={`py-space-sm px-space-sm font-medium ${p.isWinner ? 'text-primary' : 'text-on-surface'}`}>
                    {p.name}
                  </td>
                  <td className="py-space-sm px-space-sm text-right text-on-surface font-semibold">{p.netFull}</td>
                  <td className="py-space-sm px-space-sm text-right text-on-surface-variant">{p.gross}</td>
                  <td className="py-space-sm px-space-sm text-right text-on-surface-variant">{p.fees}</td>
                  <td className={`py-space-sm px-space-sm text-right font-medium ${p.isWinner ? 'text-primary' : p.uplift === 'Baseline' ? 'text-on-surface-variant' : 'text-error'}`}>
                    {p.uplift}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Official Disclaimer ──────────────────────────────────── */}
      <div className="p-space-sm rounded-lg bg-surface-container-low border border-surface-container-high/60 text-on-surface-variant font-body-sm text-[12px] flex items-start sm:items-center gap-space-xs">
        <span className="material-symbols-outlined text-[16px] text-on-surface-variant shrink-0 mt-0.5 sm:mt-0">info</span>
        <span>
          Offline evaluation on the project's synthetic/evaluation data; not a claim of production causal uplift.
        </span>
      </div>
    </div>
  );
};
