import React from 'react';
import { PlayCircle, BarChart3, Database } from 'lucide-react';

export type TabType = 'live' | 'policy' | 'audit';

interface TabsProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
  auditCount?: number;
}

export const Tabs: React.FC<TabsProps> = ({ activeTab, onTabChange, auditCount = 0 }) => {
  const tabs = [
    { id: 'live' as TabType, name: 'Live Decision', icon: PlayCircle, badge: null },
    { id: 'policy' as TabType, name: 'Policy Comparison', icon: BarChart3, badge: 'Verified ML' },
    { id: 'audit' as TabType, name: 'Audit Trail', icon: Database, badge: auditCount > 0 ? `${auditCount}` : null },
  ];

  return (
    <div className="flex items-center gap-2 border-b border-slate-800/80 pb-px mb-6 overflow-x-auto no-scrollbar">
      {tabs.map((tab) => {
        const Icon = tab.icon;
        const isActive = activeTab === tab.id;
        return (
          <button
            key={tab.id}
            type="button"
            onClick={() => onTabChange(tab.id)}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-t-xl text-xs sm:text-sm font-semibold transition-all border-b-2 cursor-pointer ${
              isActive
                ? 'bg-slate-900/90 text-white border-emerald-400 shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/40 border-transparent'
            }`}
          >
            <Icon className={`h-4 w-4 ${isActive ? 'text-emerald-400' : 'text-slate-500'}`} />
            <span>{tab.name}</span>
            {tab.badge && (
              <span
                className={`text-[10px] px-1.5 py-0.2 rounded font-bold ${
                  isActive
                    ? 'bg-emerald-500/20 text-emerald-300'
                    : 'bg-slate-800 text-slate-400'
                }`}
              >
                {tab.badge}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
};
