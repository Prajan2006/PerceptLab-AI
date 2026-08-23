import type {
  CaptureDeviceInfo,
  ConnectionState,
  FrameTick,
  Unsubscribe,
} from '@/services/core/types';

export type ConnectionStateListener = (state: ConnectionState) => void;
export type FrameListener = (frame: FrameTick) => void;

/**
 * Transport-agnostic capture contract.
 *
 * The UI depends only on this interface — never on browser device APIs or
 * vendor SDKs. Implementations may wrap the platform backend's capture
 * relay, a local simulator, or (later) additional sensor transports.
 */
export interface CameraService {
  listDevices(): Promise<readonly CaptureDeviceInfo[]>;
  connect(deviceId?: string): Promise<void>;
  disconnect(): void;
  getState(): ConnectionState;
  getActiveDeviceId(): string | null;
  onStateChange(listener: ConnectionStateListener): Unsubscribe;
  onFrame(listener: FrameListener): Unsubscribe;
}
