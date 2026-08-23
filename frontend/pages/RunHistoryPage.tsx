import { RunsTable } from '@/components/research/RunsTable';
import { Card } from '@/components/ui/Card';
import { useResearch } from '@/services/hooks/useResearch';

/** Reproducible research run history. */
export function RunHistoryPage() {
  const research = useResearch();

  return (
    <div>
      <header className="page-header">
        <h1>Run History</h1>
        <p className="page-header__sub">
          Every experiment with model, dataset, protocol, metric, and reproducibility metadata.
        </p>
      </header>

      <Card
        title="Research Runs"
        subtitle={`${research.runs.length} run(s) recorded`}
      >
        <RunsTable
          runs={research.runs}
          onCancel={(runId) => void research.cancelRun(runId)}
        />
      </Card>
    </div>
  );
}
