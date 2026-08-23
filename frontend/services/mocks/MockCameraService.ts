import type {
  CaptureDeviceInfo,
  ConnectionState,
  FrameTick,
} from '@/services/core/types';
import type {
  CameraService,
  ConnectionStateListener,
  FrameListener,
} from '@/services/interfaces/CameraService';

const MOCK_DEVICES: readonly CaptureDeviceInfo[] = [
  {
    id: 'mock-camera-0',
    label: 'Mock Camera 0 · Simulated source',
    kind: 'camera',
    resolution: { width: 1280, height: 720 },
    maxFps: 30,
  },
  {
    id: 'mock-camera-1',
    label: 'Mock Camera 1 · Simulated source',
    kind: 'camera',
    resolution: { width: 1920, height: 1080 },
    maxFps: 60,
  },
];

const CONNECT_DELAY_MS = 900;
const FRAME_INTERVAL_MS = 66; // ~15 fps simulated cadence
const FPS_SMOOTHING_FACTOR = 0.9;

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

/**
 * Simulated capture source used until the platform backend provides a
 * camera relay. Emits synthetic frame ticks so preview, status, and
 * session flows are fully exercisable without hardware. Never touches
 * physical devices — Phase-1 rule: no direct hardware access from UI.
 */
export class MockCameraService implements CameraService {
  private state: ConnectionState = 'disconnected';
  private activeDeviceId: string | null = null;
  private connectToken = 0;
  private frameTimer: ReturnType<typeof setInterval> | null = null;
  private sequence = 0;
  private lastFrameAt: number | null = null;
  private smoothedFps = 0;
  private readonly stateListeners = new Set<ConnectionStateListener>();
  private readonly frameListeners = new Set<FrameListener>();

  async listDevices(): Promise<readonly CaptureDeviceInfo[]> {
    await delay(120); // simulate transport latency
    return MOCK_DEVICES;
  }

  async connect(deviceId?: string): Promise<void> {
    if (this.state !== 'disconnected') {
      this.disconnect();
    }
    const targetDeviceId = deviceId ?? MOCK_DEVICES[0].id;
    const token = ++this.connectToken;

    this.activeDeviceId = targetDeviceId;
    this.setState('connecting');
    await delay(CONNECT_DELAY_MS);

    if (token !== this.connectToken) {
      return; // superseded by a newer connect/disconnect call
    }

    this.sequence = 0;
    this.lastFrameAt = null;
    this.smoothedFps = 0;
    this.setState('connected');
    this.emitFrame();
    this.frameTimer = setInterval(() => this.emitFrame(), FRAME_INTERVAL_MS);
  }

  disconnect(): void {
    this.connectToken += 1;
    if (this.frameTimer !== null) {
      clearInterval(this.frameTimer);
      this.frameTimer = null;
    }
    this.activeDeviceId = null;
    this.setState('disconnected');
  }

  getState(): ConnectionState {
    return this.state;
  }

  getActiveDeviceId(): string | null {
    return this.activeDeviceId;
  }

  onStateChange(listener: ConnectionStateListener): () => void {
    this.stateListeners.add(listener);
    return () => {
      this.stateListeners.delete(listener);
    };
  }

  onFrame(listener: FrameListener): () => void {
    this.frameListeners.add(listener);
    return () => {
      this.frameListeners.delete(listener);
    };
  }

  private setState(state: ConnectionState): void {
    if (this.state === state) {
      return;
    }
    this.state = state;
    for (const listener of this.stateListeners) {
      listener(state);
    }
  }

  private emitFrame(): void {
    const capturedAt = Date.now();
    if (this.lastFrameAt !== null) {
      const instantFps = 1000 / Math.max(capturedAt - this.lastFrameAt, 1);
      this.smoothedFps =
        this.smoothedFps === 0
          ? instantFps
          : this.smoothedFps * FPS_SMOOTHING_FACTOR +
            instantFps * (1 - FPS_SMOOTHING_FACTOR);
    }
    this.lastFrameAt = capturedAt;

    const frame: FrameTick = {
      sequence: ++this.sequence,
      capturedAtEpochMs: capturedAt,
      fps: Number((this.smoothedFps || 1000 / FRAME_INTERVAL_MS).toFixed(1)),
    };

    for (const listener of this.frameListeners) {
      listener(frame);
    }
  }
}
