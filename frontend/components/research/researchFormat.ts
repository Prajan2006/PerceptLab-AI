import type { StatusTone } from '@/components/ui/StatusBadge';
import type { FoldStatus, RunStatus } from '@/services/research/types';

export function runStatusTone(status: RunStatus): StatusTone {
  switch (status) {
    case 'completed':
      return 'success';
    case 'running':
      return 'warning';
    case 'failed':
      return 'danger';
    case 'queued':
    default:
      return 'info';
  }
}

export const RUN_STATUS_LABELS: Record<RunStatus, string> = {
  queued: 'Queued',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
};

export function foldStatusTone(status: FoldStatus): StatusTone {
  switch (status) {
    case 'done':
      return 'success';
    case 'running':
      return 'warning';
    case 'failed':
      return 'danger';
    case 'pending':
    default:
      return 'neutral';
  }
}

export const FOLD_STATUS_LABELS: Record<FoldStatus, string> = {
  pending: 'Pending',
  running: 'Running',
  done: 'Done',
  failed: 'Failed',
};

export function formatMetricDeg(deg: number | null | undefined): string {
  if (deg === null || deg === undefined || Number.isNaN(deg)) {
    return '—';
  }
  return `${deg.toFixed(2)}°`;
}

export function formatTimestamp(epochMs: number | null | undefined): string {
  if (!epochMs) {
    return '—';
  }
  const date = new Date(epochMs);
  return `${date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} ${date.toLocaleTimeString(
    undefined,
    { hour12: false },
  )}`;
}

export function formatDuration(startedAtEpochMs?: number | null, endedAtEpochMs?: number | null): string {
  if (!startedAtEpochMs) {
    return '—';
  }
  const end = endedAtEpochMs ?? Date.now();
  const totalSeconds = Math.max(0, Math.floor((end - startedAtEpochMs) / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}
