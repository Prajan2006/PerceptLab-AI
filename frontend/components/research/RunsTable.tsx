import { DataTable } from '@/components/ui/DataTable';
import type { DataColumn } from '@/components/ui/DataTable';
import { StatusBadge } from '@/components/ui/StatusBadge';
import type { ResearchRun } from '@/services/research/types';
import {
  RUN_STATUS_LABELS,
  formatDuration,
  formatMetricDeg,
  formatTimestamp,
  runStatusTone,
} from '@/components/research/researchFormat';

export interface RunsTableProps {
  runs: readonly ResearchRun[];
  onCancel?: (runId: string) => void;
  compact?: boolean;
}

/** Research run history — reproducibility metadata included. */
export function RunsTable({ runs, onCancel, compact = false }: RunsTableProps) {
  const columns: readonly DataColumn<ResearchRun>[] = [
    { key: 'id', header: 'Run ID', render: (run) => <span className="mono">{run.id}</span>, width: '90px' },
    { key: 'name', header: 'Experiment', render: (run) => run.experimentName },
    { key: 'model', header: 'Model', render: (run) => <span className="mono">{run.model}</span> },
    { key: 'dataset', header: 'Dataset', render: (run) => run.dataset },
    { key: 'protocol', header: 'Protocol', render: (run) => `${run.protocol} · seed ${run.seed}` },
    {
      key: 'status',
      header: 'Status',
      render: (run) => (
        <StatusBadge tone={runStatusTone(run.status)} pulse={run.status === 'running'}>
          {RUN_STATUS_LABELS[run.status]}
        </StatusBadge>
      ),
    },
    {
      key: 'metric',
      header: 'Mean error',
      align: 'right',
      render: (run) => (
        <span className="mono">
          {formatMetricDeg(run.metricValue)}
          {run.synthetic && run.metricValue !== null ? (
            <span className="synthetic-flag" title="Simulated value from the research mock service">
              *
            </span>
          ) : null}
        </span>
      ),
    },
    { key: 'started', header: 'Started', render: (run) => formatTimestamp(run.startedAtEpochMs) },
    {
      key: 'duration',
      header: 'Duration',
      align: 'right',
      render: (run) => formatDuration(run.startedAtEpochMs, run.endedAtEpochMs),
    },
    ...(compact
      ? []
      : [
          {
            key: 'output',
            header: 'Output directory',
            render: (run: ResearchRun) => (
              <span className="mono muted" title={run.outputDir}>
                {run.outputDir}
              </span>
            ),
          } satisfies DataColumn<ResearchRun>,
        ]),
    ...(!compact && onCancel
      ? [
          {
            key: 'actions',
            header: '',
            align: 'right' as const,
            render: (run: ResearchRun) =>
              run.status === 'queued' || run.status === 'running' ? (
                <button
                  type="button"
                  className="btn btn--ghost btn--sm"
                  onClick={() => onCancel(run.id)}
                >
                  Cancel
                </button>
              ) : null,
          } satisfies DataColumn<ResearchRun>,
        ]
      : []),
  ];

  return (
    <DataTable
      columns={columns}
      rows={runs}
      rowKey={(run) => run.id}
      emptyTitle="No research runs yet"
      emptyDescription="Configure an experiment in the workspace and start a run to populate history."
    />
  );
}
