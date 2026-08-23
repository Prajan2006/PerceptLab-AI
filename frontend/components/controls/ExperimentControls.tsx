import { Card } from '@/components/ui/Card';
import { Icon } from '@/components/ui/Icon';
import type { SessionPhase } from '@/services/core/types';

export interface ExperimentControlsProps {
  phase: SessionPhase;
  starting: boolean;
  stopping: boolean;
  /** Non-null disables Start and explains why (e.g., camera offline). */
  startDisabledReason?: string | null;
  error?: string | null;
  onStart: () => void;
  onStop: () => void;
}

/** Start/stop controls for experiment sessions. */
export function ExperimentControls({
  phase,
  starting,
  stopping,
  startDisabledReason,
  error,
  onStart,
  onStop,
}: ExperimentControlsProps) {
  const recording = phase === 'recording';
  const startDisabled = !recording && starting === false && startDisabledReason != null;

  return (
    <Card title="Session Controls">
      <div className="controls-stack">
        {!recording ? (
          <button
            type="button"
            className="btn btn--primary btn--lg"
            disabled={startDisabled}
            onClick={onStart}
          >
            <Icon name="play" size={18} />
            {starting ? 'Starting…' : 'Start Experiment'}
          </button>
        ) : (
          <button
            type="button"
            className="btn btn--danger btn--lg"
            disabled={stopping}
            onClick={onStop}
          >
            <Icon name="stop" size={18} />
            {stopping ? 'Stopping…' : 'Stop Experiment'}
          </button>
        )}
      </div>

      {!recording && startDisabledReason != null ? (
        <p className="hint">{startDisabledReason}</p>
      ) : null}

      {error !== null && error !== undefined && error.length > 0 ? (
        <p className="form-error" role="alert">
          {error}
        </p>
      ) : null}
    </Card>
  );
}
