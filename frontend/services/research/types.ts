/* ============================================================
   Research domain types (dataset → experiment → evaluation).
   These contracts keep research logic out of React components.
   ============================================================ */

import type { Unsubscribe } from '@/services/core/types';

/** Disk/backend presence of a dataset — distinct from run state. */
export type Availability = 'available' | 'missing' | 'unknown';

export interface DatasetInfo {
  readonly id: string;
  readonly label: string;
  /** Configured root (may not exist on this machine). */
  readonly root: string;
  readonly status: Availability;
  readonly numSubjects: number | null;
  readonly numSamples: number | null;
  readonly protocolNote: string;
}

export interface ModelInfo {
  readonly name: string;
  readonly label: string;
  readonly description: string;
  readonly inputs: Record<string, number[]>;
  readonly implemented: boolean;
}

export interface PreprocessingProfile {
  readonly recipe: 'gazehub';
  readonly faceSize: number;
  readonly expandRatio: number;
  readonly imagenetNormalization: boolean;
}

export type ProtocolType = 'lopo';

export interface ExperimentSpec {
  readonly name: string;
  readonly datasetId: string;
  readonly modelName: string;
  readonly preprocessing: PreprocessingProfile;
  readonly protocol: { readonly type: ProtocolType; readonly seed: number };
  readonly metric: 'mean_angular_error_deg';
}

export type RunStatus = 'queued' | 'running' | 'completed' | 'failed';
export type FoldStatus = 'pending' | 'running' | 'done' | 'failed';

export interface FoldProgress {
  readonly foldIndex: number;
  readonly testSubject: string;
  readonly status: FoldStatus;
  readonly errorDeg?: number;
}

export interface EvaluationResult {
  readonly runId: string;
  readonly metric: 'mean_angular_error_deg';
  /** Null while the run has not produced a final value. */
  readonly meanDeg: number | null;
  readonly perSubjectDeg: Readonly<Record<string, number>>;
  readonly foldProgress: readonly FoldProgress[];
  readonly completedAtEpochMs: number | null;
  /** True when numbers come from the deterministic simulator, not real training. */
  readonly synthetic: boolean;
}

export interface ResearchRun {
  readonly id: string;
  readonly experimentName: string;
  readonly model: string;
  readonly dataset: string;
  readonly protocol: string;
  readonly seed: number;
  readonly status: RunStatus;
  readonly metricValue: number | null;
  readonly startedAtEpochMs: number | null;
  readonly endedAtEpochMs: number | null;
  readonly outputDir: string;
  readonly synthetic: boolean;
}

export type RunsListener = () => void;

/**
 * Application-facing research contract. The mock implementation keeps the
 * workstation fully usable before/without the training backend; an HTTP
 * adapter will implement this same interface later without UI changes.
 */
export interface ResearchService {
  listDatasets(): Promise<readonly DatasetInfo[]>;
  listModels(): Promise<readonly ModelInfo[]>;
  listRuns(): Promise<readonly ResearchRun[]>;
  /** Convenience snapshot for subscription callbacks. */
  listRunsSync(): readonly ResearchRun[];
  getActiveEvaluationSync(): EvaluationResult | null;
  submitExperiment(spec: ExperimentSpec): Promise<ResearchRun>;
  cancelRun(runId: string): Promise<void>;
  onRunsChanged(listener: RunsListener): Unsubscribe;
}
