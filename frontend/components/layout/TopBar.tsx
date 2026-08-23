import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';

import { ThemeToggle } from '@/components/layout/ThemeToggle';
import { Icon } from '@/components/ui/Icon';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { formatClock } from '@/services/core/format';
import { routes } from '@/pages/routes';
import { isMockMode } from '@/services/registry';

export interface TopBarProps {
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
}

function pageTitle(pathname: string): string {
  let title = '';
  for (const route of routes) {
    const matches = route.path === '/' ? pathname === '/' : pathname.startsWith(route.path);
    if (matches) {
      title = route.title;
    }
  }
  return title;
}

/** Top bar: navigation toggle, page title, service-mode badge, live clock. */
export function TopBar({ sidebarOpen, onToggleSidebar }: TopBarProps) {
  const { pathname } = useLocation();
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => {
      window.clearInterval(timer);
    };
  }, []);

  return (
    <header className="topbar">
      <button
        type="button"
        className="btn--icon menu-btn"
        aria-label={sidebarOpen ? 'Close navigation' : 'Open navigation'}
        aria-expanded={sidebarOpen}
        aria-controls="app-sidebar"
        onClick={onToggleSidebar}
      >
        <Icon name="menu" size={20} />
      </button>

      <div className="topbar__title">{pageTitle(pathname)}</div>

      <div className="topbar__meta">
        <StatusBadge tone={isMockMode ? 'info' : 'success'}>
          {isMockMode ? 'Simulated data' : 'Live backend'}
        </StatusBadge>
        <time className="topbar__clock mono" dateTime={now.toISOString()}>
          {formatClock(now)}
        </time>
        <ThemeToggle />
      </div>
    </header>
  );
}
