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
import {
  PROTOCOL_VERSION,
  parseFrameEnvelope,
} from './protocol.ts';
import type {
  CameraStateMessage,
  DeviceDto,
  ServerControlMessage,
} from './protocol';

const REQUEST_TIMEOUT_MS = 10_000;
const CONNECT_TIMEOUT_MS = 20_000;
const RECONNECT_BASE_MS = 300;
const RECONNECT_MAX_MS = 5_000;

function defaultWebSocketUrl(): string {
  // Works in Vite (import.meta.env) and under plain Node (env undefined).
  const env = (import.meta as unknown as { env?: Record<string, string | undefined> }).env;
  return env?.VITE_WS_URL ?? 'ws://127.0.0.1:8000/api/camera/ws';
}

interface PendingDevicesRequest {
  resolve: (devices: readonly CaptureDeviceInfo[]) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

interface PendingConnect {
  resolve: () => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

/**
 * Application-facing camera adapter backed by the platform WebSocket
 * transport. Implements the existing `CameraService` contract exactly, so
 * React components and hooks remain transport-agnostic.
 *
 * - Control messages travel as JSON text; frames as binary envelopes
 *   preserving sequence, monotonic ns, wall-clock ns, fps, resolution.
 * - Unexpected socket drops trigger exponential-backoff reconnects; the
 *   server's hello/state snapshot reconciles the authoritative state, and
 *   an interrupted link is transparently re-established.
 */
export class HttpCameraService implements CameraService {
  private readonly url: string;

  private socket: WebSocket | null = null;
  private opening: Promise<WebSocket> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private disposed = false;
  private usedAtLeastOnce = false;
  /** True while the user expects a live link (auto-restored on reconnect). */
  private linkWantedDeviceId: string | null = null;

  private state: ConnectionState = 'disconnected';
  private activeDeviceId: string | null = null;
  private lastErrorMessage: string | null = null;

  private readonly stateListeners = new Set<ConnectionStateListener>();
  private readonly frameListeners = new Set<FrameListener>();
  private pendingDevices: PendingDevicesRequest | null = null;
  private pendingConnects = new Set<PendingConnect>();

  constructor(url: string = defaultWebSocketUrl()) {
    this.url = url;
  }

  // ------------------------------------------------------------------
  // CameraService contract
  // ------------------------------------------------------------------

  async listDevices(): Promise<readonly CaptureDeviceInfo[]> {
    const socket = await this.ensureSocket();
    return new Promise<readonly CaptureDeviceInfo[]>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingDevices = null;
        reject(new Error('devices.list timed out.'));
      }, REQUEST_TIMEOUT_MS);
      this.pendingDevices = { resolve, reject, timer };
      socket.send(JSON.stringify({ type: 'devices.list' } satisfies { type: 'devices.list' }));
    });
  }

  async connect(deviceId?: string): Promise<void> {
    const socket = await this.ensureSocket();

    return new Promise<void>((resolve, reject) => {
      const pending: PendingConnect = {
        resolve: () => resolve(),
        reject: (error) => reject(error),
        timer: setTimeout(() => {
          this.pendingConnects.delete(pending);
          reject(
            new Error(
              this.lastErrorMessage
                ? `Capture error: ${this.lastErrorMessage}`
                : 'camera.connect timed out.',
            ),
          );
        }, CONNECT_TIMEOUT_MS),
      };
      this.pendingConnects.add(pending);
      socket.send(JSON.stringify({ type: 'camera.connect', deviceId: deviceId ?? null }));
    });
  }

  disconnect(): void {
    if (this.socket !== null && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type: 'camera.disconnect' }));
    }
    // Optimistic local transition; server snapshot will confirm.
    this.applyState('disconnected', null);
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

  /** Non-contract utility for tests/scripts: close the transport for good. */
  dispose(): void {
    this.disposed = true;
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
    }
    this.socket?.close();
    this.socket = null;
  }

  // ------------------------------------------------------------------
  // Transport management
  // ------------------------------------------------------------------

  private ensureSocket(): Promise<WebSocket> {
    this.usedAtLeastOnce = true;
    if (this.socket !== null && this.socket.readyState === WebSocket.OPEN) {
      return Promise.resolve(this.socket);
    }
    if (this.opening !== null) {
      return this.opening;
    }
    this.opening = this.openSocket()
      .then((socket) => {
        this.opening = null;
        return socket;
      })
      .catch((error) => {
        this.opening = null;
        throw error;
      });
    return this.opening;
  }

  private openSocket(): Promise<WebSocket> {
    return new Promise<WebSocket>((resolve, reject) => {
      let settled = false;
      let helloTimer: ReturnType<typeof setTimeout>;

      const socket = new WebSocket(this.url);
      socket.binaryType = 'arraybuffer';

      const fail = (error: Error) => {
        if (settled || this.disposed) {
          return;
        }
        settled = true;
        clearTimeout(helloTimer);
        socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null;
        try {
          socket.close();
        } catch {
          /* already closed */
        }
        reject(error);
      };

      helloTimer = setTimeout(() => {
        fail(new Error(`WebSocket handshake with ${this.url} timed out.`));
      }, REQUEST_TIMEOUT_MS);

      socket.onopen = () => {
        socket.send(JSON.stringify({ type: 'hello', client: 'perceptlab-frontend' }));
      };

      socket.onclose = () => {
        this.handleSocketLoss();
        fail(new Error('WebSocket closed during handshake.'));
      };

      socket.onerror = () => {
        fail(new Error(`Unable to reach the capture service at ${this.url}.`));
      };

      socket.onmessage = (event: MessageEvent) => {
        if (typeof event.data === 'string') {
          const message = JSON.parse(event.data) as ServerControlMessage;
          if (message.type === 'hello.ack') {
            if (settled) {
              return;
            }
            if (message.protocolVersion !== PROTOCOL_VERSION) {
              fail(new Error(`Unsupported protocol v${message.protocolVersion}.`));
              return;
            }
            settled = true;
            clearTimeout(helloTimer);
            this.reconnectAttempts = 0;
            this.socket = socket;
            // The server follows the ack with an authoritative snapshot;
            // restore a previously requested link if the socket had died.
            if (this.linkWantedDeviceId !== null && this.state === 'disconnected') {
              socket.send(
                JSON.stringify({
                  type: 'camera.connect',
                  deviceId: this.linkWantedDeviceId,
                }),
              );
            } else {
              socket.send(JSON.stringify({ type: 'camera.disconnect' }));
            }
            resolve(socket);
            return;
          }
          this.handleControlMessage(message);
          return;
        }

        if (event.data instanceof ArrayBuffer) {
          this.handleFrameBuffer(event.data);
        }
      };
    });
  }

  private handleSocketLoss(): void {
    this.socket = null;
    if (this.disposed || !this.usedAtLeastOnce) {
      return;
    }
    if (this.reconnectTimer !== null) {
      return; // a retry is already scheduled
    }
    const wasLinked =
      this.state === 'connected' || this.state === 'connecting' || this.linkWantedDeviceId !== null;
    if (wasLinked) {
      this.applyState('connecting', this.activeDeviceId);
    }
    const delay = Math.min(
      RECONNECT_MAX_MS,
      RECONNECT_BASE_MS * 2 ** Math.min(this.reconnectAttempts, 6),
    );
    this.reconnectAttempts += 1;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      void this.ensureSocket().catch(() => this.handleSocketLoss());
    }, delay + Math.random() * 150);
  }

  // ------------------------------------------------------------------
  // Inbound dispatch
  // ------------------------------------------------------------------

  private handleControlMessage(message: ServerControlMessage): void {
    switch (message.type) {
      case 'camera.state': {
        const stateMessage = message as CameraStateMessage;
        this.lastErrorMessage = stateMessage.error ?? (stateMessage.state === 'error' ? 'capture error' : null);
        this.applyState(stateMessage.state, stateMessage.activeDeviceId);

        if (stateMessage.state === 'connected') {
          this.linkWantedDeviceId = stateMessage.activeDeviceId;
          this.settlePendingConnects(null);
        } else if (stateMessage.state === 'error') {
          this.linkWantedDeviceId = null;
          this.settlePendingConnects(
            new Error(stateMessage.error ?? 'Capture link entered the error state.'),
          );
        } else if (stateMessage.state === 'disconnected') {
          this.linkWantedDeviceId = null;
        }
        break;
      }
      case 'devices.list.result': {
        const pending = this.pendingDevices;
        this.pendingDevices = null;
        if (pending !== null) {
          clearTimeout(pending.timer);
          pending.resolve(message.devices.map(toDeviceInfo));
        }
        break;
      }
      case 'error': {
        this.lastErrorMessage = `${message.code}: ${message.message}`;
        this.settlePendingConnects(new Error(`${message.code}: ${message.message}`));
        break;
      }
      default:
        break; // hello.ack outside handshake is ignored
    }
  }

  private settlePendingConnects(error: Error | null): void {
    const pending = [...this.pendingConnects];
    this.pendingConnects.clear();
    for (const item of pending) {
      clearTimeout(item.timer);
      if (error === null) {
        item.resolve();
      } else {
        item.reject(error);
      }
    }
  }

  private handleFrameBuffer(data: ArrayBuffer): void {
    let tick: FrameTick;
    try {
      const { header } = parseFrameEnvelope(data);
      tick = {
        sequence: header.seq,
        capturedAtEpochMs: Math.floor(header.wallNs / 1_000_000),
        fps: header.fps,
        monotonicNs: header.monoNs,
        width: header.w,
        height: header.h,
      };
    } catch {
      return; // malformed frame — drop rather than disturb consumers
    }
    for (const listener of this.frameListeners) {
      listener(tick);
    }
  }

  private applyState(state: ConnectionState, activeDeviceId: string | null): void {
    this.state = state;
    this.activeDeviceId = activeDeviceId;
    for (const listener of this.stateListeners) {
      listener(state);
    }
  }
}

function toDeviceInfo(dto: DeviceDto): CaptureDeviceInfo {
  return {
    id: dto.id,
    label: dto.label,
    kind: dto.kind === 'video_file' ? 'video_file' : 'camera',
    resolution:
      dto.resolution ? { width: dto.resolution.width, height: dto.resolution.height } : undefined,
    maxFps: dto.maxFps ?? undefined,
  };
}
