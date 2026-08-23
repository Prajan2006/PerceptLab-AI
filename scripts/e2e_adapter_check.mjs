/**
 * End-to-end integration check: the REAL frontend HttpCameraService
 * (unmodified production code) against the LIVE FastAPI transport and the
 * existing OpenCV camera service driven by a synthetic AVI file source.
 *
 * Env:
 *   PL_WS_URL  optional (default ws://127.0.0.1:8000/api/camera/ws)
 *   PL_CLIP    required — path to synthetic AVI clip
 */

import { existsSync, writeFileSync } from 'node:fs';

import { HttpCameraService } from '../frontend/services/http/HttpCameraService.ts';

const url = process.env.PL_WS_URL ?? 'ws://127.0.0.1:8000/api/camera/ws';
const clip = process.env.PL_CLIP;

function fail(message) {
  console.error(`E2E FAIL: ${message}`);
  process.exit(1);
}

if (!clip || !existsSync(clip)) {
  fail('PL_CLIP not set or file does not exist.');
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitFor(predicate, timeoutMs, label) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await sleep(25);
  }
  fail(`timeout waiting for: ${label}`);
}

console.log(`E2E: connecting adapter to ${url}`);
const service = new HttpCameraService(url);

// --- discovery through the API -------------------------------------------
let devices;
try {
  devices = await service.listDevices();
} catch (error) {
  fail(`listDevices threw: ${error.message}`);
}
if (!Array.isArray(devices)) fail('devices result must be an array');
console.log(`E2E: discovery ok (${devices.length} physical device(s) visible)`);

// --- lifecycle + frame streaming ------------------------------------------
const states = [];
const frames = [];
service.onStateChange((state) => states.push(state));
service.onFrame((tick) => frames.push(tick));

await service.connect(clip);
await waitFor(
  () => service.getState() === 'connected',
  15_000,
  'camera link reaching connected',
);

await waitFor(() => frames.length >= 12, 20_000, 'frames arriving');
await sleep(100); // let any in-flight frames settle

if (frames.length < 12) fail(`only ${frames.length} frames`);
const sequences = frames.slice(0, 24).map((f) => f.sequence);
for (let i = 1; i < sequences.length; i += 1) {
  if (sequences[i] !== sequences[i - 1] + 1) {
    fail(`sequence ordering broken at ${i}: ${sequences.join(',')}`);
  }
}

const monotonics = frames.map((f) => f.monotonicNs);
for (let i = 1; i < monotonics.length; i += 1) {
  if (!(monotonics[i] > monotonics[i - 1])) {
    fail('monotonic timestamps not strictly increasing');
  }
}

const walls = frames.map((f) => f.capturedAtEpochMs);
if (!walls.every((value, i) => i === 0 || value >= walls[i - 1])) {
  fail('wall-clock timestamps regressed');
}
if (walls[0] < Date.now() - 60_000 || walls[0] > Date.now() + 60_000) {
  fail('wall-clock timestamps not near current epoch');
}

if (!frames.some((f) => f.fps > 0 && f.fps < 1000)) {
  fail('fps metadata never became positive/sane');
}
const geometryOk = frames.every((f) => f.width === 320 && f.height === 240);
if (!geometryOk) fail('resolution metadata not preserved (expected 320x240)');

console.log(
  `E2E: ${frames.length} frames verified — ordering/timestamps/fps/resolution OK`,
);
console.log(`E2E: states seen: ${states.join(' -> ')}`);

// --- disconnect ------------------------------------------------------------
service.disconnect();
await waitFor(
  () => service.getState() === 'disconnected',
  10_000,
  'disconnect returning link to idle',
);
console.log('E2E: clean disconnect ok');

// --- error propagation ------------------------------------------------------
let rejected = null;
try {
  await service.connect('Z:/definitely/missing/nope.avi');
} catch (error) {
  rejected = error;
}
await waitFor(() => service.getState() === 'error', 10_000, 'error state surfacing');
if (rejected === null) fail('connect() to an invalid source should reject');
console.log(`E2E: error propagated to frontend correctly (${rejected.message})`);

// --- reconnect after error ---------------------------------------------------
await service.connect(clip);
await waitFor(
  () => service.getState() === 'connected',
  15_000,
  'reconnect after error',
);
const before = frames.length;
await waitFor(() => frames.length >= before + 8, 20_000, 'frames after reconnect');
console.log(`E2E: reconnect ok (+${frames.length - before} more frames)`);

service.disconnect();
service.dispose();

// --- raw-socket frame capture: prove decodable pixels on the wire ----------
// A plain WebSocket (same runtime React would use) receives the binary
// envelope; slice the JPEG payload per protocol v1 and save it for
// independent pixel-level decoding.
const outPath = process.env.PL_PAYLOAD_OUT ?? 'pl_frame_sample.jpg';
await sleep(600); // let the disposed adapter's server-side teardown complete
const raw = new WebSocket(url);
raw.binaryType = 'arraybuffer'; // Node default is 'blob' — must match adapter
let captured = null;
const controlTexts = [];
raw.onmessage = (event) => {
  if (event.data instanceof ArrayBuffer) {
    if (captured === null) captured = event.data;
    return;
  }
  try {
    controlTexts.push(JSON.parse(event.data));
  } catch {
    /* ignore malformed */
  }
};
await new Promise((resolve, reject) => {
  raw.onopen = resolve;
  raw.onerror = () => reject(new Error('raw socket failed to open'));
});

const rawSendConnect = () =>
  raw.send(JSON.stringify({ type: 'camera.connect', deviceId: clip }));
rawSendConnect();

const rawDeadline = Date.now() + 30_000;
while (captured === null && Date.now() < rawDeadline) {
  // If our claim raced with the previous owner's release, retry politely.
  const busyIndex = controlTexts.findIndex(
    (m) => m.type === 'error' && m.code === 'camera_busy',
  );
  if (busyIndex >= 0) {
    controlTexts.length = 0;
    rawSendConnect();
  }
  await sleep(100);
}
if (!captured) fail('raw binary frame');
const view = new DataView(captured);
const headerLength = view.getUint32(0, true);
const header = JSON.parse(
  new TextDecoder().decode(new Uint8Array(captured, 4, headerLength)),
);
const payload = Buffer.from(new Uint8Array(captured, 4 + headerLength));
if (header.enc !== 'jpeg') fail(`expected jpeg payload, got ${header.enc}`);
writeFileSync(outPath, payload);
console.log(
  `E2E: raw frame captured — seq=${header.seq} ${header.w}x${header.h} fps=${header.fps} monoNs=${header.monoNs} wallNs=${header.wallNs} bytes=${payload.length} -> ${outPath}`,
);
raw.close();

console.log('E2E PASS: full adapter round-trip verified.');
process.exit(0);
