function pad(value: number, length = 2): string {
  return String(value).padStart(length, '0');
}

/** HH:MM:SS in local time. */
export function formatClock(date: Date): string {
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

/** HH:MM:SS.mmm — frame-level timestamps. */
export function formatTimestampWithMs(epochMs: number): string {
  const date = new Date(epochMs);
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}.${pad(
    date.getMilliseconds(),
    3,
  )}`;
}

/** MM:SS (or H:MM:SS beyond one hour). */
export function formatElapsed(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}:${pad(minutes)}:${pad(seconds)}`;
  }
  return `${pad(minutes)}:${pad(seconds)}`;
}

/** Zero-padded sequence counter, e.g. 000123. */
export function formatSequence(sequence: number, length = 6): string {
  return String(sequence).padStart(length, '0');
}
