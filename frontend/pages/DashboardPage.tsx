import { DatasetStatusCard, ModelStatusCard } from '@/components/research/DatasetModelCards';
import { LatestMetricsCard } from '@/components/research/LatestMetricsCard';
import { RunsTable } from '@/components/research/RunsTable';
import { Icon } from '@/components/ui/Icon';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { useCamera } from '@/services/hooks/useCamera';
import { useExperiment } from '@/services/hooks/useExperiment';
import { useResearch } from '@/services/hooks/useResearch';

const CAMERA_STATE_LABELS: Record<string, string> = {
  disconnected: 'Offline',
  connecting: 'Linking…',
  connected: 'Online',
  error: 'Error',
};

/**
 * Research Dashboard — single glance across experiments, dataset, model,
 * evaluation status, and the capture link.
 */
export function DashboardPage() {
  const research = useResearch();
  const camera = useCamera();
  const experiment = useExperiment();

  const activeRuns = research.runs.filter(
    (run) => run.status === 'running' || run.status === 'queued',
  );
  const primaryDataset = research.datasets.find((d) => d.id === 'mpii_facegaze');
  const baseline = research.models.find((m) => m.name === 'resnet50');
  const latestMeanDeg =
    research.evaluation !== null && research.evaluation.meanDeg !== null
      ? research.evaluation.meanDeg
      : null;

  return (
    <div>
      <header className="page-header">
        <h1>Research Dashboard</h1>
        <p className="page-header__sub">
          Monocular RGB gaze estimation · MPIIFaceGaze · ResNet-50 baseline vs GazeTR-Hybrid
        </p>
      </header>

      <div className="grid-tiles">
        <article className="tile">
          <span className="tile__icon tile__icon--accent" aria-hidden="true">
            <Icon name="flask" size={19} />
          </span>
          <div>
            <p className="tile__value mono">{activeRuns.length}</p>
            <p className="tile__label">Active runs</p>
            <p className="tile__hint">{activeRuns[0]?.id ?? 'Idle'}</p>
          </div>
        </article>

        <article className="tile">
          <span className="tile__icon tile__icon--success" aria-hidden="true">
            <Icon name="check" size={19} />
          </span>
          <div>
            <p className="tile__value mono">
              {latestMeanDeg !== null ? `${latestMeanDeg.toFixed(2)}°` : '—'}
            </p>
            <p className="tile__label">Latest mean error</p>
            <p className="tile__hint">mean 3D angular error · LOPO</p>
          </div>
        </article>

        <article className="tile">
          <span className="tile__icon tile__icon--info" aria-hidden="true">
            <Icon name="database" size={19} />
          </span>
          <div>
            <p className="tile__value">{primaryDataset ? primaryDataset.label : '—'}</p>
            <p className="tile__label">Primary dataset</p>
            <p className="tile__hint">15 subjects · leave-one-person-out</p>
          </div>
        </article>

        <article className="tile">
          <span className={`tile__icon ${camera.connectionState === 'connected' ? 'tile__icon--success' : 'tile__icon--warning'}`} aria-hidden="true">
            <Icon name="camera" size={19} />
          </span>
          <div>
            <p className="tile__value">{CAMERA_STATE_LABELS[camera.connectionState]}</p>
            <p className="tile__label">Capture link</p>
            <p className="tile__hint">
              {experiment.snapshot.phase === 'recording'
                ? `Recording · ${experiment.snapshot.sessionId}`
                : 'Session idle'}
            </p>
          </div>
        </article>
      </div>

      <div className="dashboard-grid">
        <div className="col-stack">
          <Card title="Evaluation Status" subtitle="Most recent LOPO evaluation">
            <LatestMetricsCard evaluation={research.evaluation} />
          </Card>

          <Card title="Recent Runs" subtitle="Last five research runs">
            <RunsTable runs={research.runs.slice(0, 5)} compact />
          </Card>
        </div>

        <div className="col-stack">
          <Card title="Current Model" subtitle="Locked research direction">
            <ModelStatusCard model={baseline} />
          </Card>

          <Card title="Dataset Status" subtitle="Configured via config/datasets.json">
            <DatasetStatusCard dataset={primaryDataset} />
          </Card>

          <Card title="Head-Pose Analysis">
            <EmptyState
              icon="cpu"
              title="Reserved"
              description="Future head-pose breakdowns per subject will appear here."
            />
          </Card>
        </div>
      </div>
    </div>
  );
}