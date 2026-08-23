import { NavLink } from 'react-router-dom';

import { Icon } from '@/components/ui/Icon';
import { NAV_GROUP_ORDER, routes } from '@/pages/routes';
import { isMockMode } from '@/services/registry';

export interface SidebarProps {
  /** Called after any navigation so the mobile drawer can close. */
  onNavigate?: () => void;
}

function BrandMark() {
  return (
    <svg width="36" height="36" viewBox="0 0 40 40" aria-hidden="true" focusable="false">
      <defs>
        <linearGradient id="pl-brand-gradient" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#2b64e3" />
          <stop offset="100%" stopColor="#12306e" />
        </linearGradient>
      </defs>
      <rect width="40" height="40" rx="10" fill="url(#pl-brand-gradient)" />
      <circle cx="20" cy="20" r="8.5" fill="none" stroke="#ffffff" strokeWidth="2.6" opacity="0.92" />
      <circle cx="27.5" cy="12.5" r="3" fill="#67e8f9" />
    </svg>
  );
}

/** Primary navigation rail. Items derive from the shared route config. */
export function Sidebar({ onNavigate }: SidebarProps) {
  return (
    <aside className="sidebar" id="app-sidebar">
      <div className="sidebar__brand">
        <BrandMark />
        <div>
          <span className="sidebar__brand-name">PerceptLab AI</span>
          <span className="sidebar__brand-tagline">Research Platform</span>
        </div>
      </div>

      <nav aria-label="Primary">
        {NAV_GROUP_ORDER.map((group) => {
          const items = routes.filter((route) => route.group === group);
          if (items.length === 0) {
            return null;
          }
          return (
            <div key={group} className="sidebar__group">
              <p className="sidebar__nav-label">{group}</p>
              <ul className="sidebar__nav-list">
                {items.map((route) => (
                  <li key={route.path}>
                    <NavLink
                      to={route.path}
                      end={route.end}
                      onClick={onNavigate}
                      className={({ isActive }) =>
                        isActive ? 'sidebar__link sidebar__link--active' : 'sidebar__link'
                      }
                    >
                      <Icon name={route.icon} size={17} />
                      {route.navLabel}
                    </NavLink>
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
      </nav>

      <footer className="sidebar__footer">
        <span>Phase 2 · Research Workstation</span>
        <span>{isMockMode ? 'Simulated services active' : 'Backend services connected'}</span>
      </footer>
    </aside>
  );
}
