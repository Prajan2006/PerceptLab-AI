import type { ConnectionState, FrameTick } from '@/services/core/types';
import { formatSequence, formatTimestampWithMs } from '@/services/core/format';

export interface CameraPreviewProps {
  connectionState: ConnectionState;
  latestFrame: FrameTick | null;
}

const STATE_ANNOUNCEMENTS: Record<ConnectionState, string> = {
  disconnected: 'Camera preview: no signal',
  connecting: 'Camera preview: establishing connection',
  connected: 'Camera preview: live',
  error: 'Camera preview: error',
};

/**
 * Camera viewport. Renders a stylized placeholder per connection state;
 * once the capture service streams real frames, the same component will
 * receive them through its props without structural change.
 */
export function CameraPreview({ connectionState, latestFrame }: CameraPreviewProps) {
  return (
    <div className={`camera-preview camera-preview--${connectionState}`}>
      <div className="camera-preview__surface" aria-hidden="true" />

      {connectionState === 'connected' ? (
        <>
          <span className="camera-preview__corner camera-preview__corner--tl" aria-hidden="true" />
          <span className="camera-preview__corner camera-preview__corner--tr" aria-hidden="true" />
          <span className="camera-preview__corner camera-preview__corner--bl" aria-hidden="true" />
          <span className="camera-preview__corner camera-preview__corner--br" aria-hidden="true" />
          <span className="camera-preview__watermark">SIMULATED SOURCE</span>
          {latestFrame !== null ? (
            <div className="camera-preview__hud mono">
              FRAME {formatSequence(latestFrame.sequence)} ·{' '}
              {formatTimestampWithMs(latestFrame.capturedAtEpochMs)} ·{' '}
              {latestFrame.fps.toFixed(1)} FPS
            </div>
          ) : null}
        </>
      ) : (
        <div className="camera-preview__center">
          {connectionState === 'connecting' ? (
            <div>
              <div className="camera-preview__spinner" aria-hidden="true" />
              <p className="camera-preview__center-title">ESTABLISHING LINK</p>
              <p className="camera-preview__center-sub">Connecting to the capture service…</p>
            </div>
          ) : connectionState === 'error' ? (
            <p className="camera-preview__center-title">SIGNAL ERROR</p>
          ) : (
            <div>
              <p className="camera-preview__center-title">NO SIGNAL</p>
              <p className="camera-preview__center-sub">
                Connect a capture device to begin monitoring.
              </p>
            </div>
          )}
        </div>
      )}

      <p className="visually-hidden" role="status">
        {STATE_ANNOUNCEMENTS[connectionState]}
      </p>
    </div>
  );
}
