import { useState } from 'react';

import { ExperimentConfigPanel } from '@/components/research/ExperimentConfigPanel';
import { FoldStatusList } from '@/components/research/FoldStatusList';
import {
  RUN_STATUS_LABELS,
  formatDuration,
  runStatusTone,
} from '@/components/research/researchFormat';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { useResearch } from '@/services/hooks/useResearch';
import { isResearchSimulated } from '@/services/registry';

/**
 * Experiment Workspace: define dataset → preprocessing → model → protocol,
 * launch runs, and watch LOPO fold progression. All research logic lives
 * behind the ResearchService contract.
 */
export function WorkspacePage() {
  const research = useResearch();
  const [submitError, setSubmitError] = useState<string | null>(null);

  const activeRun =
    research.runs.find((run) => run.status === 'running' || run.status === 'queued') ?? null;
  const activeEvaluation = activeRun ? research.evaluation : null;

  async function handleSubmit(spec: Parameters<typeof research.submitExperiment>[0]) {
    setSubmitError(null);
    try {
      await research.submitExperiment(spec);
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : String(error));
    }
  }

  return (
    <div>
      <header className="page-header">
        <h1>Experiment Workspace</h1>
        <p className="page-header__sub">
          Configure reproducible gaze-estimation runs on MPIIFaceGaze.
        </p>
      </header>

      <div className="workspace-grid">
        <div className="col-stack">
          <Card title="Experiment Definition">
            <ExperimentConfigPanel
              datasets={research.datasets}
              models={research.models}
              simulated={isResearchSimulated}
              busy={activeRun !== null}
              submitError={submitError}
              onSubmit={(spec) => void handleSubmit(spec)}
            />
          </Card>

          <Card title="Run Notes" subtitle="Reproducibility metadata attached to every run">
            <dl className="kv-grid">
              <div>
                <dt>Metric</dt>
                <dd className="mono">mean_angular_error_deg</dd>
              </div>
              <div>
                <dt>Splits</dt>
                <dd className="mono">deterministic LOPO · sorted subjects</dd>
              </div>
              <div className="kv-span">
                <dt>Output directory</dt>
                <dd className="mono muted small">
                  data/experiments/&lt;experiment-name&gt;
                </dd>
              </div>
            </dl>
          </Card>
        </div>

        <div className="col-stack">
          <Card
            title="Active Run"
            actions={
              activeRun ? (
                <div className="card__actions-inline">
                  <StatusBadge tone={runStatusTone(activeRun.status)} pulse={activeRun.status === 'running'}>
                    {RUN_STATUS_LABELS[activeRun.status]}
                  </StatusBadge>
                  <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    disabled={activeRun.status !== 'queued' && activeRun.status !== 'running'}
                    onClick={() => void research.cancelRun(activeRun.id)}
                  >
                    Cancel
                  </button>
                </div>
              ) : null
            }
          >
            {activeRun === null ? (
              <EmptyState
                icon="flask"
                title="No run in progress"
                description="Define an experiment on the left, then start a run. Completed runs appear under Run History."
              />
            ) : (
              <dl className="kv-grid">
                <div>
                  <dt>Run ID</dt>
                  <dd className="mono">{activeRun.id}</dd>
                </div>
                <div>
                  <dt>Elapsed</dt>
                  <dd className="mono">
                    {formatDuration(activeRun.startedAtEpochMs, activeRun.endedAtEpochMs)}
                  </dd>
                </div>
              </dl>
            )}
          </Card>

          <Card title="LOPO Fold Progress">
            {activeEvaluation && activeEvaluation.foldProgress.length > 0 ? (
              <FoldStatusList folds={activeEvaluation.foldProgress} />
            ) : (
              <EmptyState
                icon="check"
                title="No folds running"
                description="Fold-by-fold status for the 15 leave-one-person-out splits will stream here during a run."
              />
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
