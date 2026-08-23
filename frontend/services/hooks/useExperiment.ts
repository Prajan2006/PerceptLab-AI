import { useCallback, useEffect, useState } from 'react';

import type {
  ExperimentSnapshot,
  StartExperimentRequest,
} from '@/services/core/types';
import { experimentService } from '@/services/registry';

export interface UseExperimentResult {
  snapshot: ExperimentSnapshot;
  starting: boolean;
  stopping: boolean;
  error: string | null;
  start(request?: StartExperimentRequest): Promise<void>;
  stop(): Promise<void>;
}

function toMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** React binding for the ExperimentService interface. */
export function useExperiment(): UseExperimentResult {
  const [snapshot, setSnapshot] = useState<ExperimentSnapshot>(() =>
    experimentService.getSnapshot(),
  );
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => experimentService.onSnapshot(setSnapshot), []);

  const start = useCallback(async (request?: StartExperimentRequest) => {
    setStarting(true);
    setError(null);
    try {
      await experimentService.start(request);
    } catch (cause) {
      setError(toMessage(cause));
    } finally {
      setStarting(false);
    }
  }, []);

  const stop = useCallback(async () => {
    setStopping(true);
    setError(null);
    try {
      await experimentService.stop();
    } catch (cause) {
      setError(toMessage(cause));
    } finally {
      setStopping(false);
    }
  }, []);

  return { snapshot, starting, stopping, error, start, stop };
}
