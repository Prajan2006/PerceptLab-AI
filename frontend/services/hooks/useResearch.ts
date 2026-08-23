import { useCallback, useEffect, useState } from 'react';

import type {
  DatasetInfo,
  EvaluationResult,
  ExperimentSpec,
  ModelInfo,
  ResearchRun,
} from '@/services/research/types';
import { researchService } from '@/services/registry';

export interface UseResearchResult {
  datasets: readonly DatasetInfo[];
  models: readonly ModelInfo[];
  runs: readonly ResearchRun[];
  evaluation: EvaluationResult | null;
  submitExperiment(spec: ExperimentSpec): Promise<void>;
  cancelRun(runId: string): Promise<void>;
}

/** React binding for the ResearchService interface. */
export function useResearch(): UseResearchResult {
  const [datasets, setDatasets] = useState<readonly DatasetInfo[]>([]);
  const [models, setModels] = useState<readonly ModelInfo[]>([]);
  const [runs, setRuns] = useState<readonly ResearchRun[]>([]);
  const [evaluation, setEvaluation] = useState<EvaluationResult | null>(null);

  useEffect(() => {
    let cancelled = false;
    void researchService.listDatasets().then((result) => !cancelled && setDatasets(result));
    void researchService.listModels().then((result) => !cancelled && setModels(result));

    const refresh = () => {
      setRuns(researchService.listRunsSync());
      setEvaluation(researchService.getActiveEvaluationSync());
    };
    refresh();
    const unsubscribe = researchService.onRunsChanged(refresh);

    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, []);

  const submitExperiment = useCallback(async (spec: ExperimentSpec) => {
    await researchService.submitExperiment(spec);
  }, []);

  const cancelRun = useCallback(async (runId: string) => {
    await researchService.cancelRun(runId);
  }, []);

  return { datasets, models, runs, evaluation, submitExperiment, cancelRun };
}
