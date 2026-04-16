'use client';

export interface ProgressBarProps {
  value: number;           // current value (0..max)
  max?: number;            // default 100
  label?: string;          // left label
  valueLabel?: string;     // right label override
  status?: 'running' | 'done' | 'error' | 'idle';
  size?: 'xs' | 'sm' | 'md' | 'lg';
  showLabels?: boolean;
  indeterminate?: boolean; // use bouncing bar when total is unknown
}

const HEIGHT: Record<string, string> = {
  xs: 'h-1',
  sm: 'h-2',
  md: 'h-3',
  lg: 'h-5',
};

const BAR_BG: Record<string, string> = {
  running: 'bg-[#58a6ff] progress-bar-running',
  done:    'bg-[#3fb950]',
  error:   'bg-[#f85149]',
  idle:    'bg-[#58a6ff]/50',
};

const VALUE_COLOR: Record<string, string> = {
  running: 'text-[#8b949e]',
  done:    'text-[#3fb950]',
  error:   'text-[#f85149]',
  idle:    'text-[#8b949e]',
};

export default function ProgressBar({
  value,
  max = 100,
  label,
  valueLabel,
  status = 'idle',
  size = 'sm',
  showLabels = true,
  indeterminate = false,
}: ProgressBarProps) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  const h = HEIGHT[size] ?? HEIGHT.sm;
  const displayRight = valueLabel ?? `${pct < 10 ? pct.toFixed(1) : pct.toFixed(0)}%`;

  return (
    <div className="w-full">
      {showLabels && (
        <div className="flex justify-between items-center text-xs mb-1.5">
          <span className="text-[#8b949e]">{label ?? ''}</span>
          <span className={VALUE_COLOR[status] ?? 'text-[#8b949e]'}>{displayRight}</span>
        </div>
      )}
      <div className={`w-full bg-[#21262d] rounded-full overflow-hidden ${h}`}>
        {indeterminate ? (
          <div className={`${h} progress-bar-indeterminate w-full rounded-full`} />
        ) : (
          <div
            className={`${h} rounded-full transition-all duration-500 ${BAR_BG[status] ?? BAR_BG.idle}`}
            style={{ width: `${pct}%`, minWidth: pct > 0 ? '4px' : '0' }}
          />
        )}
      </div>
    </div>
  );
}
