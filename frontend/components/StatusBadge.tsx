type Status = 'running' | 'done' | 'error' | 'starting' | 'unknown';

interface StatusBadgeProps {
  status: Status | string;
  className?: string;
}

const STATUS_CONFIG: Record<Status, { dot: string; text: string; label: string }> = {
  done: {
    dot: 'bg-success',
    text: 'text-success',
    label: 'Gotowe',
  },
  running: {
    dot: 'bg-accent animate-pulse',
    text: 'text-accent',
    label: 'Uruchomiony',
  },
  error: {
    dot: 'bg-error',
    text: 'text-error',
    label: 'Błąd',
  },
  starting: {
    dot: 'bg-warning animate-pulse',
    text: 'text-warning',
    label: 'Startuje',
  },
  unknown: {
    dot: 'bg-text-muted',
    text: 'text-text-muted',
    label: 'Nieznany',
  },
};

export default function StatusBadge({ status, className = '' }: StatusBadgeProps) {
  const normalizedStatus = (status as Status) in STATUS_CONFIG ? (status as Status) : 'unknown';
  const config = STATUS_CONFIG[normalizedStatus];

  return (
    <span
      className={`
        inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full
        bg-bg-surface2 border border-border text-xs font-medium
        ${config.text} ${className}
      `}
    >
      <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${config.dot}`} />
      {config.label}
    </span>
  );
}
