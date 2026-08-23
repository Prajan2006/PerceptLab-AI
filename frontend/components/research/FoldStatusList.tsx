import { ProgressBar } from '@/components/ui/ProgressBar';
import { StatusBadge } from '@/components/ui/StatusBadge';
import type { FoldProgress } from '@/services/research/types';
import {
  FOLD_STATUS_LABELS,
  foldStatusTone,
  formatMetricDeg,
} from '@/components/research/researchFormat';

export interface FoldStatusListProps {
  folds: readonly FoldProgress[];
}

/** Leave-one-person-out fold status with per-subject errors. */
export function FoldStatusList({ folds }: FoldStatusListProps) {
  const done = folds.filter((fold) => fold.status === 'done').length;
  return (
    <div className="fold-list">
      <ProgressBar value={done} max={folds.length} label="LOPO folds" />
      <ul className="fold-list__rows">
        {folds.map((fold) => (
          <li key={fold.foldIndex} className="fold-list__row">
            <span className="mono muted">#{String(fold.foldIndex + 1).padStart(2, '0')}</span>
            <span className="mono">{fold.testSubject}</span>
            <span style={{ marginLeft: 'auto', display: 'flex', gap: '8px', alignItems: 'center' }}>
              <span className="mono">{formatMetricDeg(fold.errorDeg)}</span>
              <StatusBadge tone={foldStatusTone(fold.status)} pulse={fold.status === 'running'}>
                {FOLD_STATUS_LABELS[fold.status]}
              </StatusBadge>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
