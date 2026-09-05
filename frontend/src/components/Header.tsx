import React, { useEffect, useState } from 'react';
import { checkSystemHealth } from '../services/api';
import { TabType } from './Tabs';

interface HeaderProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
  latencyMs?: number;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, onTabChange, latencyMs }) => {
  const [health, setHealth] = useState<{ ok: boolean; checking: boolean }>({
    ok: true,
    checking: false,
  });

  useEffect(() => {
    let mounted = true;
    checkSystemHealth().then((res) => {
      if (mounted) {
        setHealth({ ok: res.ok, checking: false });
      }
    });
    const interval = setInterval(() => {
      checkSystemHealth().then((res) => {
        if (mounted) {
          setHealth({ ok: res.ok, checking: false });
        }
      });
    }, 30000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="fixed top-0 left-0 w-full z-50 bg-surface/95 backdrop-blur-md border-b border-surface-container-high">
      <div className="h-16 max-w-[1440px] mx-auto px-3 sm:px-gutter-desktop flex items-center justify-between gap-1 sm:gap-space-md">
        {/* Brand & Logo */}
        <div className="flex items-center gap-2 sm:gap-space-md shrink-0">
          <img
            alt="Recovery Intelligence Engine Logo"
            className="h-6 sm:h-7 w-auto object-contain"
            src="https://lh3.googleusercontent.com/aida/AEtjO1X9tiajaVhbMErEH5uUblrS2G0u6M4FSxN3EZAfHOIV_Z7MFUSm2AGE_sfuuvOrPdk4jjqXj6oqR39RTnXmrM93fNjeoggvJCuLQZTFRpd_N80oorGmHdNp-y4WPuXA513-oMhBPy0GuBu9F4vbPAfRIlVJpv3IKgfhYDGYM4aiNpqWZclVcaIZvcD8NdHojS5PTILEQ2-0AAnGBcsEN5mL9N77nVx6KXZTIrSx_IXMWe9Anm_ATpKpRcU"
          />
          <div className="flex flex-col">
            <span className="font-headline-sm text-[13px] sm:text-[15px] tracking-tight text-on-surface font-semibold truncate max-w-[120px] xs:max-w-[180px] sm:max-w-none">
              Recovery Intelligence
            </span>
            <span className="font-body-sm text-[11px] sm:text-[12px] text-on-surface-variant hidden md:block">
              Causal decisions for failed payments
            </span>
          </div>
        </div>

        {/* Center Navigation Tabs */}
        <nav className="flex items-center gap-0.5 sm:gap-space-xs h-full shrink-0">
          <button
            type="button"
            onClick={() => onTabChange('live')}
            className={`h-full flex items-center px-2 sm:px-space-md font-label-caps uppercase transition-colors cursor-pointer text-[10px] sm:text-[11px] tracking-wider ${
              activeTab === 'live'
                ? 'bg-surface-container-high text-on-surface border-b-2 border-primary font-medium'
                : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container'
            }`}
          >
            <span className="sm:hidden">LIVE</span>
            <span className="hidden sm:inline">LIVE DECISION</span>
          </button>
          <button
            type="button"
            onClick={() => onTabChange('policy')}
            className={`h-full flex items-center px-2 sm:px-space-md font-label-caps uppercase transition-colors cursor-pointer text-[10px] sm:text-[11px] tracking-wider ${
              activeTab === 'policy'
                ? 'bg-surface-container-high text-on-surface border-b-2 border-primary font-medium'
                : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container'
            }`}
          >
            <span className="sm:hidden">POLICY</span>
            <span className="hidden sm:inline">POLICY COMPARISON</span>
          </button>
          <button
            type="button"
            onClick={() => onTabChange('audit')}
            className={`h-full flex items-center px-2 sm:px-space-md font-label-caps uppercase transition-colors cursor-pointer text-[10px] sm:text-[11px] tracking-wider ${
              activeTab === 'audit'
                ? 'bg-surface-container-high text-on-surface border-b-2 border-primary font-medium'
                : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container'
            }`}
          >
            <span className="sm:hidden">AUDIT</span>
            <span className="hidden sm:inline">AUDIT TRAIL</span>
          </button>
        </nav>

        {/* Right Status Pill */}
        <div className="flex items-center gap-space-xs shrink-0">
          <div className="flex items-center gap-1 sm:gap-space-xs px-2 sm:px-space-sm py-1 bg-surface-container-low rounded border border-surface-container-high">
            <span className="relative flex h-2 w-2">
              <span
                className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                  health.ok ? 'bg-primary' : 'bg-tertiary'
                }`}
              ></span>
              <span
                className={`relative inline-flex rounded-full h-2 w-2 ${
                  health.ok ? 'bg-primary' : 'bg-tertiary'
                }`}
              ></span>
            </span>
            <span className="font-label-caps text-[10px] sm:text-[11px] text-on-surface-variant uppercase hidden sm:inline">API:</span>
            <span className={`font-label-caps text-[10px] sm:text-[11px] font-medium ${health.ok ? 'text-primary' : 'text-tertiary'}`}>
              <span className="sm:hidden">{health.ok ? 'Live' : 'Off'}</span>
              <span className="hidden sm:inline">{health.ok ? 'Connected' : 'Connecting'}</span>
            </span>
            {latencyMs !== undefined && (
              <span className="font-mono-code-sm text-[10px] sm:text-[11px] text-on-surface-variant ml-1 pl-1 border-l border-surface-container-high hidden md:inline">
                {latencyMs}ms
              </span>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
