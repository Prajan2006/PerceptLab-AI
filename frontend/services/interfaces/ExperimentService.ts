import type {
  ExperimentSnapshot,
  StartExperimentRequest,
  Unsubscribe,
} from '@/services/core/types';

export type SnapshotListener = (snapshot: ExperimentSnapshot) => void;

/**
 * Experiment lifecycle contract, independent of camera and analysis
 * implementations. Future backends (REST, WebSocket, gRPC gateway)
 * implement this same interface.
 */
export interface ExperimentService {
  start(request?: StartExperimentRequest): Promise<ExperimentSnapshot>;
  stop(): Promise<ExperimentSnapshot>;
  getSnapshot(): ExperimentSnapshot;
  onSnapshot(listener: SnapshotListener): Unsubscribe;
}
