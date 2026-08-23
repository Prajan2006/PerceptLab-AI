import type {
  ExperimentSnapshot,
  StartExperimentRequest,
} from '@/services/core/types';
import type {
  ExperimentService,
  SnapshotListener,
} from '@/services/interfaces/ExperimentService';

const SNAPSHOT_TICK_MS = 250;

/**
 * Simulated experiment lifecycle manager.
 *
 * Tracks one active session at a time, ticks elapsed time while
 * recording, and counts completed sessions. Persistence arrives with
 * the storage-adapter integration; the interface stays unchanged.
 */
export class MockExperimentService implements ExperimentService {
  private snapshot: ExperimentSnapshot = {
    phase: 'idle',
    sessionId: null,
    startedAtEpochMs: null,
    elapsedMs: 0,
    sessionsCompleted: 0,
  };
  private readonly listeners = new Set<SnapshotListener>();
  private ticker: ReturnType<typeof setInterval> | null = null;
  private sessionCounter = 0;

  async start(_request?: StartExperimentRequest): Promise<ExperimentSnapshot> {
    if (this.snapshot.phase === 'recording') {
      throw new Error('An experiment session is already running.');
    }

    this.sessionCounter += 1;
    this.setSnapshot({
      phase: 'recording',
      sessionId: `sess-${String(this.sessionCounter).padStart(4, '0')}`,
      startedAtEpochMs: Date.now(),
      elapsedMs: 0,
      sessionsCompleted: this.snapshot.sessionsCompleted,
    });

    this.ticker = setInterval(() => {
      if (this.snapshot.startedAtEpochMs !== null) {
        this.setSnapshot({
          ...this.snapshot,
          elapsedMs: Date.now() - this.snapshot.startedAtEpochMs,
        });
      }
    }, SNAPSHOT_TICK_MS);

    return this.snapshot;
  }

  async stop(): Promise<ExperimentSnapshot> {
    if (this.snapshot.phase !== 'recording') {
      throw new Error('No experiment session is currently running.');
    }

    if (this.ticker !== null) {
      clearInterval(this.ticker);
      this.ticker = null;
    }

    this.setSnapshot({
      phase: 'idle',
      sessionId: null,
      startedAtEpochMs: null,
      elapsedMs: 0,
      sessionsCompleted: this.snapshot.sessionsCompleted + 1,
    });

    return this.snapshot;
  }

  getSnapshot(): ExperimentSnapshot {
    return this.snapshot;
  }

  onSnapshot(listener: SnapshotListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private setSnapshot(snapshot: ExperimentSnapshot): void {
    this.snapshot = snapshot;
    for (const listener of this.listeners) {
      listener(snapshot);
    }
  }
}
