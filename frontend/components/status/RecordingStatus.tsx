import { Card } from '@/components/ui/Card';
import type { ExperimentSnapshot } from '@/services/core/types';
import { formatElapsed } from '@/services/core/format';

export interface RecordingStatusProps {
  snapshot: ExperimentSnapshot;
}

/** Live recording indicator with session id and elapsed time. */
export function RecordingStatus({ snapshot }: RecordingStatusProps) {
  const recording = snapshot.phase === 'recording' && snapshot.sessionId !== null;

  return (
    <Card title="Recording Status">
      <section className="recording-status" aria-live="polite">
        {recording ? (
          <>
            <span className="rec-dot" aria-hidden="true" />
            <div>
              <p className="recording-status__title">Recording</p>
              <p className="recording-status__meta mono">
                Session {snapshot.sessionId} · {formatElapsed(snapshot.elapsedMs)} elapsed
              </p>
            </div>
          </>
        ) : (
          <p className="recording-status__idle">Standby — no active session.</p>
        )}
      </section>
    </Card>
  );
}
