import { useEffect, useState } from 'react';

import { Card } from '@/components/ui/Card';
import { StatusBadge } from '@/components/ui/StatusBadge';
import type { StatusTone } from '@/components/ui/StatusBadge';
import type { CaptureDeviceInfo, ConnectionState } from '@/services/core/types';

export interface CameraStatusBarProps {
  devices: readonly CaptureDeviceInfo[];
  activeDeviceId: string | null;
  connectionState: ConnectionState;
  connect: (deviceId?: string) => void;
  disconnect: () => void;
  busy: boolean;
  error?: string | null;
}

const STATE_META: Record<ConnectionState, { tone: StatusTone; label: string }> = {
  disconnected: { tone: 'neutral', label: 'Offline' },
  connecting: { tone: 'warning', label: 'Connecting…' },
  connected: { tone: 'success', label: 'Online' },
  error: { tone: 'danger', label: 'Error' },
};

/** Device selection + connection control panel (self-contained card). */
export function CameraStatusBar({
  devices,
  activeDeviceId,
  connectionState,
  connect,
  disconnect,
  busy,
  error,
}: CameraStatusBarProps) {
  const [pendingDeviceId, setPendingDeviceId] = useState<string | null>(null);

  // Clear any staged selection once a connection has been established.
  useEffect(() => {
    if (connectionState === 'connected') {
      setPendingDeviceId(null);
    }
  }, [connectionState]);

  const connected = connectionState === 'connected';
  const selectedId =
    pendingDeviceId ??
    activeDeviceId ??
    (devices.length > 0 ? devices[0].id : '');
  const selectedDevice = devices.find((device) => device.id === selectedId);
  const stateMeta = STATE_META[connectionState];

  return (
    <Card
      title="Capture Control"
      subtitle="Connection is brokered by the platform service layer."
      actions={
        <StatusBadge tone={stateMeta.tone} pulse={connectionState === 'connecting'}>
          {stateMeta.label}
        </StatusBadge>
      }
    >
      <div className="field-row">
        <label className="field">
          <span className="field__label">Capture device</span>
          <select
            className="select"
            value={selectedId}
            disabled={busy || devices.length === 0 || connected}
            onChange={(event) => setPendingDeviceId(event.target.value)}
          >
            {devices.length === 0 ? <option value="">No devices available</option> : null}
            {devices.map((device) => (
              <option key={device.id} value={device.id}>
                {device.label}
              </option>
            ))}
          </select>
        </label>

        <div className="btn-row">
          {connected || connectionState === 'connecting' ? (
            <button
              type="button"
              className="btn btn--ghost"
              disabled={busy}
              onClick={disconnect}
            >
              Disconnect
            </button>
          ) : (
            <button
              type="button"
              className="btn btn--primary"
              disabled={busy || devices.length === 0}
              onClick={() => connect(pendingDeviceId ?? undefined)}
            >
              Connect
            </button>
          )}
        </div>
      </div>

      {selectedDevice?.resolution !== undefined ? (
        <p className="device-meta mono">
          {selectedDevice.resolution.width}×{selectedDevice.resolution.height}
          {selectedDevice.maxFps !== undefined
            ? ` · up to ${selectedDevice.maxFps} fps`
            : ''}
        </p>
      ) : null}

      {connected ? <p className="hint">Disconnect before switching devices.</p> : null}

      {error !== null && error !== undefined && error.length > 0 ? (
        <p className="form-error" role="alert">
          {error}
        </p>
      ) : null}
    </Card>
  );
}
