import type {
  DatasetInfo,
  EvaluationResult,
  ExperimentSpec,
  FoldProgress,
  ModelInfo,
  ResearchRun,
  ResearchService,
  RunStatus,
  RunsListener,
} from '@/services/research/types';
import type { Unsubscribe } from '@/services/core/types';

/**
 * Deterministic research simulator.
 *
 * Lifecycle, fold progression, and state transitions mirror the future
 * training backend so the workstation is fully exercisable today. Metric
 * numbers produced here are clearly flagged `synthetic` and are derived
 * from a documented deterministic formula — never presented as real
 * scientific results.
 */

const SIMULATED_FOLD_MS = 900;
const QUEUED_MS = 700;

const DATASETS: readonly DatasetInfo[] = [
  {
    id: 'mpii_facegaze',
    label: 'MPIIFaceGaze',
    root: 'data/raw/MPIIFaceGaze (configure via PERCEPTLAB_MPIIFACE_GAZE_ROOT)',
    status: 'unknown', // mock cannot inspect the local disk
    numSubjects: 15,
    numSamples: 45000,
    protocolNote: 'GazeHub preprocessing · 15-subject leave-one-person-out',
  },
];

const MODELS: readonly ModelInfo[] = [
  {
    name: 'resnet50',
    label: 'ResNet-50 (baseline)',
    description: 'Appearance-based gaze estimation baseline — primary comparison anchor.',
    inputs: { face: [3, 224, 224] },
    implemented: false,
  },
  {
    name: 'gazetr_hybrid',
    label: 'GazeTR-Hybrid (comparison)',
    description: 'CNN eye encoder + transformer; measures architecture headroom over ResNet-50.',
    inputs: { left_eye: [3, 36, 60], right_eye: [3, 36, 60], face: [3, 224, 224] },
    implemented: false,
  },
];

/** Deterministic pseudo-metric for the simulator (flagged synthetic in UI). */
function syntheticMeanDeg(experimentName: string, seed: number): number {
  let hash = seed;
  for (let i = 0; i < experimentName.length; i += 1) {
    hash = (hash * 31 + experimentName.charCodeAt(i)) >>> 0;
  }
  return Math.round((4.6 + (hash % 24) / 10) * 100) / 100; // 4.60°–6.90°
}

function syntheticPerSubject(meanDeg: number, seed: number): Record<string, number> {
  const perSubject: Record<string, number> = {};
  for (let i = 0; i < 15; i += 1) {
    const subject = `p${String(i).padStart(2, '0')}`;
    const offset = (((seed + i * 37) % 9) - 4) * 0.12;
    perSubject[subject] = Math.round((meanDeg + offset) * 100) / 100;
  }
  return perSubject;
}

interface MutableRun {
  id: string;
  experimentName: string;
  model: string;
  dataset: string;
  seed: number;
  status: RunStatus;
  startedAtEpochMs: number | null;
  endedAtEpochMs: number | null;
  outputDir: string;
  timers: ReturnType<typeof setTimeout>[];
}

export class MockResearchService implements ResearchService {
  private runs: MutableRun[] = [];
  private evaluations = new Map<string, EvaluationResult>();
  private listeners = new Set<RunsListener>();
  private counter = 0;

  async listDatasets(): Promise<readonly DatasetInfo[]> {
    return DATASETS;
  }

  async listModels(): Promise<readonly ModelInfo[]> {
    return MODELS;
  }

  async listRuns(): Promise<readonly ResearchRun[]> {
    return this.listRunsSync();
  }

  listRunsSync(): readonly ResearchRun[] {
    return [...this.runs]
      .sort((a, b) => (b.startedAtEpochMs ?? 0) - (a.startedAtEpochMs ?? 0))
      .map((run) => this.toPublic(run));
  }

  getActiveEvaluationSync(): EvaluationResult | null {
    const completed = this.runs.filter((run) => run.status === 'completed');
    if (completed.length === 0) {
      return null;
    }
    return this.evaluations.get(completed[completed.length - 1].id) ?? null;
  }

  onRunsChanged(listener: RunsListener): Unsubscribe {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  async submitExperiment(spec: ExperimentSpec): Promise<ResearchRun> {
    this.counter += 1;
    const now = Date.now();
    const run: MutableRun = {
      id: `run-${String(this.counter).padStart(4, '0')}`,
      experimentName: spec.name || `experiment-${this.counter}`,
      model: spec.modelName,
      dataset: spec.datasetId,
      seed: spec.protocol.seed,
      status: 'queued',
      startedAtEpochMs: now,
      endedAtEpochMs: null,
      outputDir: `data/experiments/${spec.name || `experiment-${this.counter}`}`,
      timers: [],
    };
    this.runs.push(run);
    this.notify();

    run.timers.push(
      setTimeout(() => {
        if (run.status !== 'queued') return;
        run.status = 'running';
        this.startFolds(run, spec);
        this.notify();
      }, QUEUED_MS),
    );

    return this.toPublic(run);
  }

  async cancelRun(runId: string): Promise<void> {
    const run = this.runs.find((candidate) => candidate.id === runId);
    if (!run || (run.status !== 'queued' && run.status !== 'running')) {
      return;
    }
    run.timers.forEach(clearTimeout);
    run.status = 'failed';
    run.endedAtEpochMs = Date.now();
    this.notify();
  }

  private startFolds(run: MutableRun, spec: ExperimentSpec): void {
    const subjects = Array.from({ length: 15 }, (_, i) => `p${String(i).padStart(2, '0')}`);
    let evaluation: EvaluationResult = {
      runId: run.id,
      metric: spec.metric,
      meanDeg: null,
      perSubjectDeg: {},
      foldProgress: subjects.map((subject, index) => ({
        foldIndex: index,
        testSubject: subject,
        status: 'pending' as const,
      })),
      completedAtEpochMs: null,
      synthetic: true,
    };
    this.evaluations.set(run.id, evaluation);

    subjects.forEach((subject, index) => {
      run.timers.push(
        setTimeout(() => {
          if (run.status !== 'running') return;
          const meanDeg = syntheticMeanDeg(run.experimentName, run.seed);
          const perSubject = syntheticPerSubject(meanDeg, run.seed);

          const folds: FoldProgress[] = subjects.map((entry, position) => ({
            foldIndex: position,
            testSubject: entry,
            status:
              position < index ? ('done' as const)
              : position === index ? ('running' as const)
              : ('pending' as const),
          }));
          folds[index] = { ...folds[index], status: 'done', errorDeg: perSubject[subject] };

          evaluation = {
            ...evaluation,
            meanDeg: null, // only final at completion
            perSubjectDeg: Object.fromEntries(
              folds.filter((f) => f.status === 'done').map((f) => [f.testSubject, f.errorDeg as number]),
            ),
            foldProgress: folds,
          };

          if (index === subjects.length - 1) {
            run.status = 'completed';
            run.endedAtEpochMs = Date.now();
            evaluation = {
              ...evaluation,
              meanDeg,
              completedAtEpochMs: Date.now(),
            };
          }
          this.evaluations.set(run.id, evaluation);
          this.notify();
        }, SIMULATED_FOLD_MS * (index + 1)),
      );
    });
  }

  private toPublic(run: MutableRun): ResearchRun {
    return {
      id: run.id,
      experimentName: run.experimentName,
      model: run.model,
      dataset: run.dataset,
      protocol: 'lopo',
      seed: run.seed,
      status: run.status,
      metricValue: this.evaluations.get(run.id)?.meanDeg ?? null,
      startedAtEpochMs: run.startedAtEpochMs,
      endedAtEpochMs: run.endedAtEpochMs,
      outputDir: run.outputDir,
      synthetic: true,
    };
  }

  private notify(): void {
    for (const listener of this.listeners) {
      try {
        listener();
      } catch {
        /* listener isolation */
      }
    }
  }
}
