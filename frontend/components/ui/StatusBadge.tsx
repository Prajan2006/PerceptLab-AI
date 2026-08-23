import type { ReactNode } from 'react';

export type StatusTone = 'neutral' | 'success' | 'warning' | 'danger' | 'info';

export interface StatusBadgeProps {
  tone?: StatusTone;
  pulse?: boolean;
  children: ReactNode;
}

/** Small status pill with a leading indicator dot. */
export function StatusBadge({ tone = 'neutral', pulse = false, children }: StatusBadgeProps) {
  const className = ['badge', `badge--${tone}`, pulse ? 'badge--pulse' : '']
    .filter(Boolean)
    .join(' ');
  return (
    <span className={className}>
      <span className="badge__dot" aria-hidden="true" />
      {children}
    </span>
  );
}
