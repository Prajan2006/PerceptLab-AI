import { EmptyState } from '@/components/ui/EmptyState';
import { ProgressBar } from '@/components/ui/ProgressBar';
import { StatusBadge } from '@/components/ui/StatusBadge';
import type { EvaluationResult } from '@/services/research/types';
import { formatMetricDeg } from '@/components/research/researchFormat';

export interface LatestMetricsCardProps {
  evaluation: EvaluationResult | null;
}

/** Headline metric summary for the dashboard / evaluation header. */
export function LatestMetricsCard({ evaluation }: LatestMetricsCardProps) {
  if (evaluation === null) {
    return (
      <EmptyState
        icon="chart"
        title="Not evaluated yet"
        description="Start a research run in the Experiment Workspace — mean angular error and per-subject results will appear here."
      />
    );
  }

  const entries = Object.entries(evaluation.perSubjectDeg);
  const best = entries.length
    ? entries.reduce((a, b) => (b[1] < a[1] ? b : a))
    : null;
  const worst = entries.length
    ? entries.reduce((a, b) => (b[1] > a[1] ? b : a))
    : null;
  const doneFolds = evaluation.foldProgress.filter((f) => f.status === 'done').length;

  return (
    <div className="metric-summary">
      <div className="metric-summary__headline">
        <span className="metric-summary__value mono">
          {formatMetricDeg(evaluation.meanDeg)}
        </span>
        <span className="metric-summary__label">mean 3D angular error</span>
        {evaluation.synthetic ? (
          <StatusBadge tone="info">Simulated</StatusBadge>
        ) : (
          <StatusBadge tone="success">Measured</StatusBadge>
        )}
      </div>

      <ProgressBar value={doneFolds} max={evaluation.foldProgress.length} label="LOPO folds" />

      <dl className="kv-grid">
        <div>
          <dt>Folds done</dt>
          <dd className="mono">
            {doneFolds}/{evaluation.foldProgress.length}
          </dd>
        </div>
        <div>
          <dt>Best subject</dt>
          <dd className="mono">
            {best ? `${best[0]} · ${formatMetricDeg(best[1])}` : '—'}
          </dd>
        </div>
        <div>
          <dt>Hardest subject</dt>
          <dd className="mono">
            {worst ? `${worst[0]} · ${formatMetricDeg(worst[1])}` : '—'}
          </dd>
        </div>
      </dl>

      {evaluation.meanDeg === null ? (
        <p className="hint">Running — final mean appears when all folds finish.</p>
      ) : null}
    </div>
  );
}
