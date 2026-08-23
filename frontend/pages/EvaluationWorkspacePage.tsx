import { LatestMetricsCard } from '@/components/research/LatestMetricsCard';
import {
  FOLD_STATUS_LABELS,
  foldStatusTone,
  formatMetricDeg,
} from '@/components/research/researchFormat';
import { Card } from '@/components/ui/Card';
import { DataTable } from '@/components/ui/DataTable';
import type { DataColumn } from '@/components/ui/DataTable';
import { EmptyState } from '@/components/ui/EmptyState';
import { StatusBadge } from '@/components/ui/StatusBadge';
import type { FoldProgress } from '@/services/research/types';
import { useResearch } from '@/services/hooks/useResearch';

interface SubjectRow {
  subjectId: string;
  errorDeg: number | null;
  status: FoldProgress['status'];
}

/**
 * Evaluation Workspace: mean angular error, per-subject breakdown, and
 * reserved comparison areas. Placeholder states are explicit — results
 * are never invented.
 */
export function EvaluationWorkspacePage() {
  const research = useResearch();
  const evaluation = research.evaluation;

  const rows: SubjectRow[] = evaluation
    ? evaluation.foldProgress.map((fold) => ({
        subjectId: fold.testSubject,
        errorDeg: fold.errorDeg ?? null,
        status: fold.status,
      }))
    : [];

  const columns: readonly DataColumn<SubjectRow>[] = [
    { key: 'fold', header: 'Fold', render: (row) => <span className="mono muted">{rows.indexOf(row) + 1}</span>, width: '70px' },
    { key: 'subject', header: 'Test subject', render: (row) => <span className="mono">{row.subjectId}</span> },
    {
      key: 'error',
      header: 'Angular error',
      align: 'right',
      render: (row) => <span className="mono">{formatMetricDeg(row.errorDeg)}</span>,
    },
    {
      key: 'status',
      header: 'Fold status',
      align: 'right',
      render: (row) => (
        <StatusBadge tone={foldStatusTone(row.status)} pulse={row.status === 'running'}>
          {FOLD_STATUS_LABELS[row.status]}
        </StatusBadge>
      ),
    },
  ];

  return (
    <div>
      <header className="page-header">
        <h1>Evaluation Workspace</h1>
        <p className="page-header__sub">
          Leave-one-person-out results on MPIIFaceGaze · metric: mean 3D angular error.
        </p>
      </header>

      <div className="workspace-grid">
        <div className="col-stack">
          <Card title="Latest Evaluation" subtitle="Most recent completed run">
            <LatestMetricsCard evaluation={evaluation} />
          </Card>

          <Card title="Per-Subject Results" subtitle="One row per held-out subject (fold)">
            <DataTable
              columns={columns}
              rows={rows}
              rowKey={(row) => row.subjectId}
              emptyTitle="No evaluation results yet"
              emptyDescription="Run an experiment to populate the per-subject LOPO breakdown."
            />
          </Card>
        </div>

        <div className="col-stack">
          <Card title="Model Comparison" subtitle="ResNet-50 baseline vs GazeTR-Hybrid">
            <EmptyState
              icon="chart"
              title="Reserved for comparison charts"
              description="Once both baselines have completed runs, mean-error comparisons will be visualized here."
            />
          </Card>

          <Card title="Experiment History" subtitle="Recent runs feeding this view">
            <ul className="mini-history">
              {research.runs.slice(0, 5).map((run) => (
                <li key={run.id}>
                  <span className="mono">{run.id}</span>
                  <span>{run.experimentName}</span>
                </li>
              ))}
              {research.runs.length === 0 ? (
                <li className="muted">No runs recorded yet.</li>
              ) : null}
            </ul>
          </Card>

          <Card title="Failure-Case Analysis">
            <EmptyState
              icon="cpu"
              title="Reserved"
              description="Worst-fold qualitative samples will be listed here for inspection."
            />
          </Card>
        </div>
      </div>
    </div>
  );
}
