/**
 * Versioned WebSocket protocol (v1) — TypeScript mirror of
 * backend/models/messages.py. Keep both files in sync.
 */

export const PROTOCOL_VERSION = 1;

// ---------------------------------------------------------------------------
// Client → Server
// ---------------------------------------------------------------------------

export interface HelloMessage {
  type: 'hello';
  client?: string;
}

export interface DeviceListRequest {
  type: 'devices.list';
}

export interface CameraConnectMessage {
  type: 'camera.connect';
  deviceId?: string | null;
}

export interface CameraDisconnectMessage {
  type: 'camera.disconnect';
}

export type ClientMessage =
  | HelloMessage
  | DeviceListRequest
  | CameraConnectMessage
  | CameraDisconnectMessage;

// ---------------------------------------------------------------------------
// Server → Client (control)
// ---------------------------------------------------------------------------

export type ConnectionStateValue = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface ResolutionDto {
  width: number;
  height: number;
}

export interface DeviceDto {
  id: string;
  label: string;
  kind: string;
  resolution?: ResolutionDto | null;
  maxFps?: number | null;
  backend?: string | null;
}

export interface HelloAckMessage {
  type: 'hello.ack';
  protocolVersion: number;
  serverVersion: string;
}

export interface CameraStateMessage {
  type: 'camera.state';
  state: ConnectionStateValue;
  activeDeviceId: string | null;
  error?: string | null;
}

export interface DeviceListResultMessage {
  type: 'devices.list.result';
  devices: DeviceDto[];
}

export interface ProtocolErrorMessage {
  type: 'error';
  code: string;
  message: string;
}

export type ServerControlMessage =
  | HelloAckMessage
  | CameraStateMessage
  | DeviceListResultMessage
  | ProtocolErrorMessage;

// ---------------------------------------------------------------------------
// Server → Client (frames)
// ---------------------------------------------------------------------------
// One binary message per frame:
//   [uint32 LE header length][UTF-8 JSON header][JPEG payload]

export interface FrameHeader {
  seq: number;
  monoNs: number;
  wallNs: number;
  fps: number;
  w: number;
  h: number;
  enc: 'jpeg' | 'none';
}

const textDecoder = new TextDecoder();

/** Splits a binary frame message into its header (payload excluded here). */
export function parseFrameEnvelope(data: ArrayBuffer): { header: FrameHeader; payloadByteOffset: number } {
  if (data.byteLength < 4) {
    throw new Error('Frame message truncated: missing header length.');
  }
  const view = new DataView(data);
  const headerLength = view.getUint32(0, true);
  if (4 + headerLength > data.byteLength) {
    throw new Error('Frame message truncated: incomplete header.');
  }
  const headerJson = textDecoder.decode(new Uint8Array(data, 4, headerLength));
  return { header: JSON.parse(headerJson) as FrameHeader, payloadByteOffset: 4 + headerLength };
}
