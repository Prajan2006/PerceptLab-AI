/* ============================================================
   PerceptLab AI — Shared domain types for the service layer.
   UI components depend on these contracts, never on concrete
   transports (HTTP/WebSocket) or hardware implementations.
   ============================================================ */

/** Lifecycle of a capture link. Extensible for future states. */
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface Resolution {
  readonly width: number;
  readonly height: number;
}

/**
 * Kind of capture source. `camera` today; `video_file` arrives through the
 * live transport; future sensors extend this union without changing consumers.
 */
export type CaptureSourceKind = 'camera' | 'video_file';

export interface CaptureDeviceInfo {
  readonly id: string;
  readonly label: string;
  readonly kind: CaptureSourceKind;
  readonly resolution?: Resolution;
  readonly maxFps?: number;
}

/** Metadata describing one captured frame. */
export interface FrameTick {
  readonly sequence: number;
  readonly capturedAtEpochMs: number;
  readonly fps: number;
  /** High-resolution monotonic stamp (ns) — present on the live transport. */
  readonly monotonicNs?: number;
  /** Frame geometry — present on the live transport. */
  readonly width?: number;
  readonly height?: number;
}

export type SessionPhase = 'idle' | 'recording';

export interface ExperimentSnapshot {
  readonly phase: SessionPhase;
  readonly sessionId: string | null;
  readonly startedAtEpochMs: number | null;
  readonly elapsedMs: number;
  readonly sessionsCompleted: number;
}

/**
 * Generic start request. Optional fields let future experiment types
 * (protocols, sensor sets, algorithm selections) join without breaking
 * existing callers.
 */
export interface StartExperimentRequest {
  readonly label?: string;
  readonly protocolId?: string;
  readonly cameraDeviceId?: string;
  readonly sensorIds?: readonly string[];
}

export type Unsubscribe = () => void;
