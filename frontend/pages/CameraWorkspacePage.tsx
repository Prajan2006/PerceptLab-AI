import { CameraPreview } from '@/components/camera/CameraPreview';
import { CameraStatusBar } from '@/components/camera/CameraStatusBar';
import { ExperimentControls } from '@/components/controls/ExperimentControls';
import { RecordingStatus } from '@/components/status/RecordingStatus';
import { Card } from '@/components/ui/Card';
import { EmptyState } from '@/components/ui/EmptyState';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { useCamera } from '@/services/hooks/useCamera';
import { useExperiment } from '@/services/hooks/useExperiment';

/**
 * Live capture workspace. The camera stack is unchanged from the verified
 * integration; the right column reserves space for future gaze inference
 * outputs without coupling to them.
 */
export function CameraWorkspacePage() {
  const camera = useCamera();
  const experiment = useExperiment();
  const recording = experiment.snapshot.phase === 'recording';

  function handleStart() {
    void experiment.start({
      label: 'Manual capture',
      cameraDeviceId: camera.activeDeviceId ?? undefined,
    });
  }

  function handleStop() {
    void experiment.stop();
  }

  return (
    <div>
      <header className="page-header">
        <h1>Live Camera Workspace</h1>
        <p className="page-header__sub">
          Real-time capture monitoring — independent of research experiments.
        </p>
      </header>

      <div className="workspace-grid">
        <div className="col-stack">
          <Card
            className="card--flush"
            title="Camera Preview"
            actions={
              <StatusBadge tone={camera.connectionState === 'connected' ? 'success' : 'neutral'}>
                {camera.connectionState === 'connected' ? 'Live' : 'Standby'}
              </StatusBadge>
            }
          >
            <CameraPreview
              connectionState={camera.connectionState}
              latestFrame={camera.latestFrame}
            />
          </Card>

          <CameraStatusBar
            devices={camera.devices}
            activeDeviceId={camera.activeDeviceId}
            connectionState={camera.connectionState}
            connect={(deviceId) => void camera.connect(deviceId)}
            disconnect={() => camera.disconnect()}
            busy={camera.connectionState === 'connecting'}
            error={camera.error}
          />

          <ExperimentControls
            phase={experiment.snapshot.phase}
            starting={experiment.starting}
            stopping={experiment.stopping}
            startDisabledReason={
              recording || camera.connectionState === 'connected'
                ? null
                : 'Connect a capture device to enable session controls.'
            }
            error={experiment.error}
            onStart={handleStart}
            onStop={handleStop}
          />

          <RecordingStatus snapshot={experiment.snapshot} />
        </div>

        <div className="col-stack">
          <Card title="Capture Metadata" subtitle="Live stream statistics">
            <dl className="kv-grid">
              <div>
                <dt>Connection</dt>
                <dd className="mono">{camera.connectionState}</dd>
              </div>
              <div>
                <dt>Device</dt>
                <dd className="mono small">{camera.activeDeviceId ?? '—'}</dd>
              </div>
              <div>
                <dt>Frames received</dt>
                <dd className="mono">{camera.latestFrame?.sequence ?? 0}</dd>
              </div>
              <div>
                <dt>Measured FPS</dt>
                <dd className="mono">
                  {camera.latestFrame ? camera.latestFrame.fps.toFixed(1) : '—'}
                </dd>
              </div>
              <div>
                <dt>Resolution</dt>
                <dd className="mono">
                  {camera.latestFrame?.width && camera.latestFrame?.height
                    ? `${camera.latestFrame.width}×${camera.latestFrame.height}`
                    : '—'}
                </dd>
              </div>
            </dl>
          </Card>

          <Card title="Gaze Overlay">
            <EmptyState
              icon="pulse"
              title="Reserved for live gaze overlay"
              description="Once model inference is connected, predicted gaze vectors will be drawn over this preview in real time."
            />
          </Card>

          <Card title="Gaze Visualization">
            <EmptyState
              icon="chart"
              title="Reserved for gaze-vector plots"
              description="Future area for yaw/pitch time-series and angular-error traces during live sessions."
            />
          </Card>
        </div>
      </div>
    </div>
  );
}
