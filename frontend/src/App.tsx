import React, { useState } from 'react';
import { Header } from './components/Header';
import { TabType } from './components/Tabs';
import { LiveDecisionTab } from './components/LiveDecisionTab';
import { PolicyComparisonTab } from './components/PolicyComparisonTab';
import { AuditTrailTab } from './components/AuditTrailTab';
import { ClientAuditRecord, DecisionResponse } from './types/api';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('live');
  const [auditRecords, setAuditRecords] = useState<ClientAuditRecord[]>([]);
  const [lastLatencyMs, setLastLatencyMs] = useState<number | undefined>(undefined);
  const [evaluating, setEvaluating] = useState<boolean>(false);
  const [currentResult, setCurrentResult] = useState<DecisionResponse | null>(null);

  const handleAddAuditRecord = (record: ClientAuditRecord) => {
    setAuditRecords((prev) => [record, ...prev]);
  };

  const handleClearAuditRecords = () => {
    setAuditRecords([]);
  };

  return (
    <div className="bg-surface font-body-md text-body-md text-on-surface antialiased min-h-screen flex flex-col">
      {/* Clean Fixed Header */}
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        latencyMs={lastLatencyMs}
      />

      {/* Main Content Area */}
      <main className="w-full pt-16 max-w-[1440px] mx-auto px-4 sm:px-gutter-desktop flex-1 flex flex-col py-5 sm:py-space-xl">
        {activeTab === 'live' && (
          <LiveDecisionTab
            onAddAuditRecord={handleAddAuditRecord}
            onLatencyUpdate={setLastLatencyMs}
            evaluating={evaluating}
            setEvaluating={setEvaluating}
            currentResult={currentResult}
            setCurrentResult={setCurrentResult}
          />
        )}

        {activeTab === 'policy' && <PolicyComparisonTab />}

        {activeTab === 'audit' && (
          <AuditTrailTab
            auditRecords={auditRecords}
            onClearRecords={handleClearAuditRecords}
          />
        )}
      </main>

      {/* Clean Minimal Footer */}
      <footer className="w-full border-t border-surface-container-high bg-surface-container-lowest py-space-md">
        <div className="max-w-[1440px] mx-auto px-4 sm:px-gutter-desktop flex flex-col sm:flex-row items-center justify-between gap-2 font-mono-code-sm text-[11px] sm:text-[12px] text-on-surface-variant text-center sm:text-left">
          <div>Recovery Intelligence Engine • Causal Payment Recovery</div>
          <div className="text-on-surface-variant/70">
            Offline Policy Evaluation &amp; Online Decision Gateway
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
