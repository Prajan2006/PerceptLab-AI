export interface ProgressBarProps {
  value: number;
  max: number;
  label?: string;
}

/** Thin determinate progress indicator used for LOPO fold progress. */
export function ProgressBar({ value, max, label }: ProgressBarProps) {
  const safeMax = Math.max(max, 1);
  const clamped = Math.max(0, Math.min(value, safeMax));
  const percent = Math.round((clamped / safeMax) * 100);
  return (
    <div className="progress">
      <div
        className="progress__track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={safeMax}
        aria-valuenow={clamped}
        aria-label={label ?? 'Progress'}
      >
        <div className="progress__fill" style={{ width: `${percent}%` }} />
      </div>
      <span className="progress__value mono">
        {label ? `${label} · ` : ''}
        {clamped}/{safeMax}
      </span>
    </div>
  );
}
