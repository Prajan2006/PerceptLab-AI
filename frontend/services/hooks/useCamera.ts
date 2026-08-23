import { useCallback, useEffect, useState } from 'react';

import type {
  CaptureDeviceInfo,
  ConnectionState,
  FrameTick,
} from '@/services/core/types';
import { cameraService } from '@/services/registry';

export interface UseCameraResult {
  connectionState: ConnectionState;
  devices: readonly CaptureDeviceInfo[];
  activeDeviceId: string | null;
  latestFrame: FrameTick | null;
  error: string | null;
  connect(deviceId?: string): Promise<void>;
  disconnect(): void;
}

function toMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** React binding for the CameraService interface. */
export function useCamera(): UseCameraResult {
  const [connectionState, setConnectionState] = useState<ConnectionState>(() =>
    cameraService.getState(),
  );
  const [devices, setDevices] = useState<readonly CaptureDeviceInfo[]>([]);
  const [activeDeviceId, setActiveDeviceId] = useState<string | null>(() =>
    cameraService.getActiveDeviceId(),
  );
  const [latestFrame, setLatestFrame] = useState<FrameTick | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const offStateChange = cameraService.onStateChange(setConnectionState);
    const offFrame = cameraService.onFrame(setLatestFrame);

    let cancelled = false;
    cameraService
      .listDevices()
      .then((result) => {
        if (!cancelled) {
          setDevices(result);
        }
      })
      .catch((cause) => {
        if (!cancelled) {
          setError(toMessage(cause));
        }
      });

    return () => {
      cancelled = true;
      offStateChange();
      offFrame();
    };
  }, []);

  const connect = useCallback(async (deviceId?: string) => {
    setError(null);
    try {
      await cameraService.connect(deviceId);
      setActiveDeviceId(cameraService.getActiveDeviceId());
    } catch (cause) {
      setError(toMessage(cause));
    }
  }, []);

  const disconnect = useCallback(() => {
    setError(null);
    cameraService.disconnect();
    setActiveDeviceId(cameraService.getActiveDeviceId());
  }, []);

  return {
    connectionState,
    devices,
    activeDeviceId,
    latestFrame,
    error,
    connect,
    disconnect,
  };
}
