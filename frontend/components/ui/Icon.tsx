import type { ReactElement } from 'react';

export type IconName =
  | 'dashboard'
  | 'flask'
  | 'pulse'
  | 'menu'
  | 'camera'
  | 'play'
  | 'stop'
  | 'check'
  | 'chart'
  | 'database'
  | 'cpu'
  | 'log'
  | 'sun'
  | 'moon';

const GLYPHS: Record<IconName, ReactElement> = {
  dashboard: (
    <>
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" />
    </>
  ),
  flask: (
    <>
      <path d="M10 2h4" />
      <path d="M10 3v6.3L4.7 18a2 2 0 0 0 1.8 3h11a2 2 0 0 0 1.8-3L14 9.3V3" />
      <path d="M7.5 15h9" />
    </>
  ),
  pulse: <path d="M3 12h4l2.5-6.5L14 18l2-6h5" />,
  menu: <path d="M4 6h16M4 12h16M4 18h16" />,
  camera: (
    <>
      <rect x="3" y="6" width="12" height="12" rx="2" />
      <path d="m15 10 5-3v10l-5-3" />
    </>
  ),
  play: <path d="M8 5.5v13l11-6.5z" />,
  stop: <rect x="7" y="7" width="10" height="10" rx="1.5" />,
  check: <path d="m4 12.5 5 5L20 6.5" />,
  chart: (
    <>
      <path d="M4 20v-6" />
      <path d="M9.5 20V9" />
      <path d="M15 20v-8" />
      <path d="M20.5 20V5" />
    </>
  ),
  database: (
    <>
      <ellipse cx="12" cy="5" rx="8" ry="3" />
      <path d="M4 5v14c0 1.66 3.58 3 8 3s8-1.34 8-3V5" />
      <path d="M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3" />
    </>
  ),
  cpu: (
    <>
      <rect x="7" y="7" width="10" height="10" rx="1.5" />
      <path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" />
    </>
  ),
  log: (
    <>
      <path d="M8.5 6H20M8.5 12H20M8.5 18H20" />
      <path d="M4 6h.01M4 12h.01M4 18h.01" />
    </>
  ),
  sun: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </>
  ),
  moon: <path d="M21 12.79A9 9 0 1 1 11.21 3a7 7 0 0 0 9.79 9.79z" />,
};

interface IconProps {
  name: IconName;
  size?: number;
  className?: string;
}

/** Inline SVG icon set — zero external icon dependencies. */
export function Icon({ name, size = 18, className }: IconProps) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {GLYPHS[name]}
    </svg>
  );
}
