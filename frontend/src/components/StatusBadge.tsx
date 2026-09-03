import React from 'react';
import { Database, Sparkles, ShieldAlert, Cpu, AlertTriangle } from 'lucide-react';

interface StatusBadgeProps {
  type: 'cache' | 'llm' | 'fallback' | 'guardrail' | 'fresh' | 'action';
  label?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ type, label, size = 'md' }) => {
  const sizeClasses = {
    sm: 'text-[10px] px-2 py-0.5 gap-1',
    md: 'text-xs px-2.5 py-1 gap-1.5',
    lg: 'text-sm px-3.5 py-1.5 gap-2 font-bold tracking-wide',
  }[size];

  switch (type) {
    case 'cache':
      return (
        <span
          className={`inline-flex items-center rounded-md font-semibold bg-cyan-950/80 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-500/20 ${sizeClasses}`}
        >
          <Database className="h-3.5 w-3.5 text-cyan-400" />
          {label || 'CACHE HIT'}
        </span>
      );

    case 'llm':
      return (
        <span
          className={`inline-flex items-center rounded-md font-semibold bg-indigo-950/80 text-indigo-300 border border-indigo-500/40 shadow-sm shadow-indigo-500/20 ${sizeClasses}`}
        >
          <Cpu className="h-3.5 w-3.5 text-indigo-400" />
          {label || 'LLM REASONING'}
        </span>
      );

    case 'fallback':
      return (
        <span
          className={`inline-flex items-center rounded-md font-semibold bg-amber-950/80 text-amber-300 border border-amber-500/40 shadow-sm shadow-amber-500/20 ${sizeClasses}`}
        >
          <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
          {label || 'MODEL FALLBACK'}
        </span>
      );

    case 'guardrail':
      return (
        <span
          className={`inline-flex items-center rounded-md font-bold bg-rose-950/90 text-rose-200 border border-rose-500/60 shadow-lg shadow-rose-950/50 animate-pulse ${sizeClasses}`}
        >
          <ShieldAlert className="h-4 w-4 text-rose-400" />
          {label || 'GUARDRAIL OVERRIDE'}
        </span>
      );

    case 'fresh':
      return (
        <span
          className={`inline-flex items-center rounded-md font-semibold bg-teal-950/80 text-teal-300 border border-teal-500/40 ${sizeClasses}`}
        >
          <Sparkles className="h-3.5 w-3.5 text-teal-400" />
          {label || 'FRESH EVALUATION'}
        </span>
      );

    default:
      return null;
  }
};
